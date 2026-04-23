from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from evals.contracts import ReviewQueueItem
from model.contracts import (
    AlertStatus,
    FeedbackEventType,
    OnlineAlertEvaluation,
    OnlineAlertPolicy,
    OnlineMetricGroup,
    OnlineMetricsSummary,
    OnlineSummarySnapshot,
    Outcome,
    RiskBand,
    RunRecord,
    RuntimeFeedbackEvent,
    RuntimeFeedbackRequest,
    TriggeredAlert,
)
from services.run_service import RunService
from services.storage_service import StorageService
from services.human_review_router import HumanReviewRouter, ReviewRouteRequest


CORE_ROLLBACK_METRICS = {
    "invalid_output_rate",
    "supported_answer_rate",
    "review_recommended_rate",
    "thumbs_down_rate",
    "high_risk_run_rate",
}

RISK_ORDER = {
    RiskBand.LOW: 0,
    RiskBand.MEDIUM: 1,
    RiskBand.HIGH: 2,
}


class OnlineControlPlaneService:
    def __init__(
        self,
        *,
        run_service: RunService,
        storage: StorageService,
        review_router: HumanReviewRouter,
        alert_policy_path: str = "data/online_alert_policy.json",
    ) -> None:
        self.run_service = run_service
        self.storage = storage
        self.review_router = review_router
        self.alert_policy_path = Path(alert_policy_path)

    def record_feedback(self, request: RuntimeFeedbackRequest) -> RuntimeFeedbackEvent:
        run = self.run_service.get_run(request.run_id)
        if run is None:
            raise LookupError(f"Run {request.run_id} was not found.")

        event = RuntimeFeedbackEvent(
            event_id=str(uuid4()),
            run_id=request.run_id,
            session_id=request.session_id,
            event_type=FeedbackEventType(request.event_type.value),
            event_value=request.event_value,
            created_at=datetime.now(UTC),
            model_backend=run.model_backend,
            prompt_version=run.prompt_version,
            response_outcome=run.outcome,
            risk_band=run.risk_band,
        )
        self.storage.write_feedback_event(event)

        if run.risk_band in {RiskBand.MEDIUM, RiskBand.HIGH} and event.event_type == FeedbackEventType.THUMBS_DOWN:
            self.review_router.ensure_review_item(
                ReviewRouteRequest(
                    run_id=run.run_id,
                    source="thumbs_down_feedback",
                    reason="Medium/high-risk run received a thumbs-down signal.",
                    reason_codes=["feedback_thumbs_down_risky_run"],
                    metadata={
                        "event_id": event.event_id,
                        "session_id": event.session_id,
                        "event_value": event.event_value,
                    },
                )
            )
        return event

    def list_feedback(self, *, limit: int = 50, run_id: str | None = None) -> list[RuntimeFeedbackEvent]:
        return self.storage.list_feedback_events(limit=limit, run_id=run_id)

    def build_live_summary(self, *, limit: int = 50) -> OnlineSummarySnapshot:
        run_records = self.run_service.list_run_records(limit=limit)
        feedback_events = (
            self.storage.list_feedback_events(run_ids=[record.run_id for record in run_records], limit=0)
            if run_records
            else []
        )
        summary = self._build_metrics_summary(run_records, feedback_events)
        alert_evaluation = self.evaluate_summary(summary)
        worst_runs = self._rank_live_runs(run_records, feedback_events)[:3]

        review_items: list[ReviewQueueItem] = []
        if alert_evaluation.status == AlertStatus.ACTION_REQUIRED:
            for run in worst_runs:
                review_items.append(
                    self.review_router.ensure_review_item(
                        ReviewRouteRequest(
                            run_id=run.run_id,
                            source="online_summary_alert",
                            reason="Online metrics summary reached action_required.",
                            reason_codes=["online_summary_action_required"],
                            metadata={
                                "alert_status": alert_evaluation.status.value,
                                "worst_run_ranked": True,
                            },
                        )
                    )
                )
        return OnlineSummarySnapshot(
            summary=summary,
            alert_evaluation=alert_evaluation,
            worst_run_ids=[run.run_id for run in worst_runs],
            review_queue_item_ids=[item.review_queue_item_id for item in review_items],
        )

    def evaluate_summary(self, summary: OnlineMetricsSummary) -> OnlineAlertEvaluation:
        if summary.total_runs == 0:
            return OnlineAlertEvaluation(status=AlertStatus.OK)

        policy = self._load_alert_policy()
        triggered_alerts = [
            alert
            for alert in [
                self._max_rate_alert(
                    metric_name="invalid_output_rate",
                    observed_value=summary.invalid_output_rate,
                    threshold_value=policy.max_invalid_output_rate,
                    label="Invalid output rate",
                ),
                self._max_rate_alert(
                    metric_name="review_recommended_rate",
                    observed_value=summary.review_recommended_rate,
                    threshold_value=policy.max_review_recommended_rate,
                    label="Review recommended rate",
                ),
                self._max_rate_alert(
                    metric_name="thumbs_down_rate",
                    observed_value=summary.thumbs_down_rate,
                    threshold_value=policy.max_thumbs_down_rate,
                    label="Thumbs down rate",
                ),
                self._min_rate_alert(
                    metric_name="supported_answer_rate",
                    observed_value=summary.supported_answer_rate,
                    threshold_value=policy.min_supported_answer_rate,
                    label="Supported answer rate",
                ),
                self._max_rate_alert(
                    metric_name="high_risk_run_rate",
                    observed_value=self._high_risk_run_rate(summary),
                    threshold_value=policy.max_high_risk_run_rate,
                    label="High-risk run rate",
                ),
            ]
            if alert is not None
        ]

        status = AlertStatus.OK
        if any(alert.severity == AlertStatus.ACTION_REQUIRED for alert in triggered_alerts):
            status = AlertStatus.ACTION_REQUIRED
        elif triggered_alerts:
            status = AlertStatus.WATCH
        if status == AlertStatus.WATCH and len(triggered_alerts) >= 2:
            status = AlertStatus.ACTION_REQUIRED

        rollback_reasons = [
            alert.message
            for alert in triggered_alerts
            if alert.metric_name in CORE_ROLLBACK_METRICS and status == AlertStatus.ACTION_REQUIRED
        ]
        return OnlineAlertEvaluation(
            status=status,
            triggered_alerts=sorted(
                triggered_alerts,
                key=lambda alert: (0 if alert.severity == AlertStatus.ACTION_REQUIRED else 1, alert.metric_name),
            ),
            rollback_recommended=bool(rollback_reasons),
            rollback_reasons=rollback_reasons,
        )

    def _build_metrics_summary(
        self,
        run_records: list[RunRecord],
        feedback_events: list[RuntimeFeedbackEvent],
    ) -> OnlineMetricsSummary:
        now = datetime.now(UTC)
        feedback_by_run = self._feedback_by_run(feedback_events)
        window_start = min((record.created_at for record in run_records), default=now)
        window_end = max((record.created_at for record in run_records), default=now)
        metrics = self._metric_counts(run_records, feedback_by_run)
        return OnlineMetricsSummary(
            window_start=window_start,
            window_end=window_end,
            total_runs=len(run_records),
            total_feedback_events=len(feedback_events),
            outcome_counts=metrics["outcome_counts"],
            risk_band_counts=metrics["risk_band_counts"],
            avg_groundedness_proxy_score=metrics["avg_groundedness_proxy_score"],
            avg_citation_validity_score=metrics["avg_citation_validity_score"],
            avg_policy_adherence_score=metrics["avg_policy_adherence_score"],
            avg_format_validity_score=metrics["avg_format_validity_score"],
            avg_retrieval_support_score=metrics["avg_retrieval_support_score"],
            avg_brand_voice_score=metrics["avg_brand_voice_score"],
            avg_tone_appropriateness_score=metrics["avg_tone_appropriateness_score"],
            avg_online_score_total=metrics["avg_online_score_total"],
            review_recommended_rate=metrics["review_recommended_rate"],
            supported_answer_rate=metrics["supported_answer_rate"],
            invalid_output_rate=metrics["invalid_output_rate"],
            thumbs_up_rate=metrics["thumbs_up_rate"],
            thumbs_down_rate=metrics["thumbs_down_rate"],
            high_risk_run_count=metrics["high_risk_run_count"],
            by_prompt_version=self._group_metrics(run_records, feedback_by_run, lambda run: run.prompt_version),
            by_model_backend=self._group_metrics(run_records, feedback_by_run, lambda run: run.model_backend.value),
            by_response_outcome=self._group_metrics(run_records, feedback_by_run, lambda run: run.outcome.value),
            by_risk_band=self._group_metrics(run_records, feedback_by_run, lambda run: run.risk_band.value),
        )

    def _group_metrics(
        self,
        run_records: list[RunRecord],
        feedback_by_run: dict[str, list[RuntimeFeedbackEvent]],
        key_func: Callable[[RunRecord], str],
    ) -> list[OnlineMetricGroup]:
        grouped_runs: dict[str, list[RunRecord]] = defaultdict(list)
        for run in run_records:
            grouped_runs[key_func(run)].append(run)

        results: list[OnlineMetricGroup] = []
        for group_key in sorted(grouped_runs):
            group_runs = grouped_runs[group_key]
            group_feedback = {
                run_id: events
                for run_id, events in feedback_by_run.items()
                if any(run.run_id == run_id for run in group_runs)
            }
            metrics = self._metric_counts(group_runs, group_feedback)
            results.append(
                OnlineMetricGroup(
                    group_key=group_key,
                    total_runs=len(group_runs),
                    total_feedback_events=sum(len(events) for events in group_feedback.values()),
                    outcome_counts=metrics["outcome_counts"],
                    risk_band_counts=metrics["risk_band_counts"],
                    avg_groundedness_proxy_score=metrics["avg_groundedness_proxy_score"],
                    avg_citation_validity_score=metrics["avg_citation_validity_score"],
                    avg_policy_adherence_score=metrics["avg_policy_adherence_score"],
                    avg_format_validity_score=metrics["avg_format_validity_score"],
                    avg_retrieval_support_score=metrics["avg_retrieval_support_score"],
                    avg_brand_voice_score=metrics["avg_brand_voice_score"],
                    avg_tone_appropriateness_score=metrics["avg_tone_appropriateness_score"],
                    avg_online_score_total=metrics["avg_online_score_total"],
                    review_recommended_rate=metrics["review_recommended_rate"],
                    supported_answer_rate=metrics["supported_answer_rate"],
                    invalid_output_rate=metrics["invalid_output_rate"],
                    thumbs_up_rate=metrics["thumbs_up_rate"],
                    thumbs_down_rate=metrics["thumbs_down_rate"],
                    high_risk_run_count=metrics["high_risk_run_count"],
                )
            )
        return results

    def _metric_counts(
        self,
        run_records: list[RunRecord],
        feedback_by_run: dict[str, list[RuntimeFeedbackEvent]],
    ) -> dict[str, int | float | dict[str, int]]:
        total_runs = len(run_records)
        outcome_counts = Counter(run.outcome.value for run in run_records)
        risk_band_counts = Counter(run.risk_band.value for run in run_records)

        def rate(count: int) -> float:
            return round(count / total_runs, 4) if total_runs else 0.0

        def avg(values: list[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        thumbs_up_runs = self._runs_with_feedback(feedback_by_run, FeedbackEventType.THUMBS_UP)
        thumbs_down_runs = self._runs_with_feedback(feedback_by_run, FeedbackEventType.THUMBS_DOWN)

        return {
            "outcome_counts": dict(outcome_counts),
            "risk_band_counts": dict(risk_band_counts),
            "avg_groundedness_proxy_score": avg(
                [run.score_breakdown.groundedness_proxy_score for run in run_records]
            ),
            "avg_citation_validity_score": avg(
                [run.score_breakdown.citation_validity_score for run in run_records]
            ),
            "avg_policy_adherence_score": avg(
                [run.score_breakdown.policy_adherence_score for run in run_records]
            ),
            "avg_format_validity_score": avg(
                [run.score_breakdown.format_validity_score for run in run_records]
            ),
            "avg_retrieval_support_score": avg(
                [run.score_breakdown.retrieval_support_score for run in run_records]
            ),
            "avg_brand_voice_score": avg(
                [run.score_breakdown.brand_voice_score for run in run_records]
            ),
            "avg_tone_appropriateness_score": avg(
                [run.score_breakdown.tone_appropriateness_score for run in run_records]
            ),
            "avg_online_score_total": avg([run.online_score_total for run in run_records]),
            "review_recommended_rate": rate(sum(1 for run in run_records if run.review_required)),
            "supported_answer_rate": rate(sum(1 for run in run_records if run.outcome == Outcome.SUPPORTED_ANSWER)),
            "invalid_output_rate": rate(sum(1 for run in run_records if self._is_invalid_output(run))),
            "thumbs_up_rate": rate(len(thumbs_up_runs)),
            "thumbs_down_rate": rate(len(thumbs_down_runs)),
            "high_risk_run_count": sum(1 for run in run_records if run.risk_band == RiskBand.HIGH),
        }

    def _load_alert_policy(self) -> OnlineAlertPolicy:
        with self.alert_policy_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return OnlineAlertPolicy.model_validate(payload)

    def _max_rate_alert(
        self,
        *,
        metric_name: str,
        observed_value: float,
        threshold_value: float,
        label: str,
    ) -> TriggeredAlert | None:
        if observed_value <= threshold_value:
            return None
        severity = self._severity_for_max(observed_value, threshold_value)
        return TriggeredAlert(
            metric_name=metric_name,
            severity=severity,
            observed_value=round(observed_value, 4),
            threshold_value=round(threshold_value, 4),
            comparator="<=",
            message=f"{label} is {round(observed_value, 4)} which is above the allowed maximum of {round(threshold_value, 4)}.",
        )

    def _min_rate_alert(
        self,
        *,
        metric_name: str,
        observed_value: float,
        threshold_value: float,
        label: str,
    ) -> TriggeredAlert | None:
        if observed_value >= threshold_value:
            return None
        severity = self._severity_for_min(observed_value, threshold_value)
        return TriggeredAlert(
            metric_name=metric_name,
            severity=severity,
            observed_value=round(observed_value, 4),
            threshold_value=round(threshold_value, 4),
            comparator=">=",
            message=f"{label} is {round(observed_value, 4)} which is below the required minimum of {round(threshold_value, 4)}.",
        )

    def _severity_for_max(self, observed_value: float, threshold_value: float) -> AlertStatus:
        margin = observed_value - threshold_value
        return AlertStatus.ACTION_REQUIRED if margin >= max(0.05, threshold_value * 0.5) else AlertStatus.WATCH

    def _severity_for_min(self, observed_value: float, threshold_value: float) -> AlertStatus:
        margin = threshold_value - observed_value
        return AlertStatus.ACTION_REQUIRED if margin >= max(0.05, threshold_value * 0.25) else AlertStatus.WATCH

    def _high_risk_run_rate(self, summary: OnlineMetricsSummary) -> float:
        if summary.total_runs == 0:
            return 0.0
        return round(summary.high_risk_run_count / summary.total_runs, 4)

    def _feedback_by_run(self, feedback_events: list[RuntimeFeedbackEvent]) -> dict[str, list[RuntimeFeedbackEvent]]:
        grouped: dict[str, list[RuntimeFeedbackEvent]] = defaultdict(list)
        for event in feedback_events:
            grouped[event.run_id].append(event)
        return grouped

    def _runs_with_feedback(
        self,
        feedback_by_run: dict[str, list[RuntimeFeedbackEvent]],
        *event_types: FeedbackEventType,
    ) -> set[str]:
        target_types = set(event_types)
        return {
            run_id
            for run_id, events in feedback_by_run.items()
            if any(event.event_type in target_types for event in events)
        }

    def _is_invalid_output(self, run: RunRecord) -> bool:
        validation = run.validation_result
        return (not validation.structure_valid) or (not validation.citation_valid) or validation.repair_attempted

    def _rank_live_runs(
        self,
        run_records: list[RunRecord],
        feedback_events: list[RuntimeFeedbackEvent],
    ) -> list[RunRecord]:
        feedback_by_run = self._feedback_by_run(feedback_events)
        return sorted(
            run_records,
            key=lambda run: (
                -RISK_ORDER[run.risk_band],
                -sum(1 for event in feedback_by_run.get(run.run_id, []) if event.event_type == FeedbackEventType.THUMBS_DOWN),
                -int(run.review_required),
                -int(self._is_invalid_output(run)),
                run.online_score_total,
                -run.created_at.timestamp(),
            ),
        )
