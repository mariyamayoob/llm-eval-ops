from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from evals.contracts import ReviewQueueItem
from model.contracts import ReviewPriority, RiskBand, RunRecord, SuspiciousFlag
from services.run_service import RunService


@dataclass(frozen=True)
class RunReviewContext:
    run_id: str
    trace_id: str
    online_score_total: float
    risk_band: RiskBand
    suspicious_flags: list[SuspiciousFlag]


@dataclass(frozen=True)
class ReviewRouteRequest:
    run_id: str
    source: str
    run_context: RunReviewContext | None = None
    reason: str | None = None
    reason_codes: list[str] | None = None
    priority: ReviewPriority | None = None
    metadata: dict[str, Any] | None = None


class HumanReviewRouter:
    """Single place to ensure/dedupe review queue items.

    This is intentionally a small "router" service: it centralizes the rules
    for when a review item exists and how multiple signals are merged.
    """

    def __init__(self, *, run_service: RunService) -> None:
        self.run_service = run_service

    def ensure_review_item(self, request: ReviewRouteRequest) -> ReviewQueueItem:
        run: RunRecord | None = None
        context = request.run_context
        if context is None:
            run = self._require_run(request.run_id)
            context = RunReviewContext(
                run_id=run.run_id,
                trace_id=run.trace_id,
                online_score_total=run.online_score_total,
                risk_band=run.risk_band,
                suspicious_flags=run.suspicious_flags,
            )

        existing = self.run_service.find_review_item_by_run(request.run_id)
        if existing is not None:
            updated = self._merge(existing, request)
            if updated != existing:
                self.run_service.enqueue_review(updated)
            if run is not None:
                self._mark_run_review_required(run, review_queue_item_id=existing.review_queue_item_id)
            return updated

        item = ReviewQueueItem(
            review_queue_item_id=str(uuid4()),
            run_id=context.run_id,
            trace_id=context.trace_id,
            online_score_total=context.online_score_total,
            review_priority=request.priority or self._default_priority(context.risk_band),
            suspicious_flags=context.suspicious_flags,
            review_source=request.source,
            review_sources=[request.source],
            review_reason=request.reason,
            review_reason_codes=sorted(set(request.reason_codes or [])),
            review_metadata=request.metadata or {},
        )
        self.run_service.enqueue_review(item)
        if run is not None:
            self._mark_run_review_required(run, review_queue_item_id=item.review_queue_item_id)
        return item

    def _merge(self, item: ReviewQueueItem, request: ReviewRouteRequest) -> ReviewQueueItem:
        sources = list(dict.fromkeys([*(item.review_sources or [item.review_source]), request.source]))
        reason_codes = sorted(set([*(item.review_reason_codes or []), *(request.reason_codes or [])]))
        metadata = dict(item.review_metadata or {})
        if request.metadata:
            metadata.update(request.metadata)

        priority = item.review_priority
        if request.priority is not None:
            priority = self._max_priority(priority, request.priority)

        reason = item.review_reason
        if request.reason and request.reason != reason:
            reason = self._merge_reasons(reason, request.reason)

        if (
            sources == (item.review_sources or [item.review_source])
            and reason_codes == (item.review_reason_codes or [])
            and metadata == (item.review_metadata or {})
            and priority == item.review_priority
            and reason == item.review_reason
        ):
            return item

        updated = item.model_copy(deep=True)
        updated.review_sources = sources
        updated.review_reason_codes = reason_codes
        updated.review_metadata = metadata
        updated.review_priority = priority
        updated.review_reason = reason
        return updated

    def _mark_run_review_required(self, run: RunRecord, *, review_queue_item_id: str) -> None:
        if run.review_required and run.review_queue_item_id == review_queue_item_id:
            return

        run.review_required = True
        run.review_queue_item_id = review_queue_item_id

        # Keep the persisted response payload consistent for the UI explorer.
        payload = run.response_payload
        payload.review_required = True
        payload.review_queue_item_id = review_queue_item_id
        run.response_payload = payload

        self.run_service.write_run(run)

    def _require_run(self, run_id: str) -> RunRecord:
        run = self.run_service.get_run(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} was not found.")
        return run

    def _default_priority(self, risk_band: RiskBand) -> ReviewPriority:
        return ReviewPriority.HIGH if risk_band == RiskBand.HIGH else ReviewPriority.MEDIUM

    def _max_priority(self, left: ReviewPriority, right: ReviewPriority) -> ReviewPriority:
        if left == ReviewPriority.HIGH or right == ReviewPriority.HIGH:
            return ReviewPriority.HIGH
        return ReviewPriority.MEDIUM

    def _merge_reasons(self, existing: str | None, incoming: str) -> str:
        if not existing:
            return incoming
        existing_norm = existing.strip()
        incoming_norm = incoming.strip()
        if existing_norm == incoming_norm:
            return existing_norm
        return f"{existing_norm} | {incoming_norm}"
