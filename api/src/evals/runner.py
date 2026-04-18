from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from evals.contracts import (
    CaseSet,
    EvalCase,
    OfflineComparisonBucketDelta,
    OfflineComparisonCaseDelta,
    OfflineComparisonRequest,
    OfflineComparisonRun,
    OfflineComparisonSummary,
    OfflineEvalConfig,
    OfflineEvalRun,
    ReleaseDecision,
    SkippedEvalCase,
    parse_eval_cases,
    summarize_offline_results,
)
from evals.gates import (
    GatePolicy,
    apply_gate_policy,
    comparison_hard_fail_reasons,
    load_gate_policy,
    resolve_thresholds,
)
from evals.scorers import score_offline_case
from model.contracts import ModelBackend, PolicyDeskAssistantRequest, ScenarioName
from services.qa_service import PolicyDeskService
from services.storage_service import StorageService


OPENAI_SUPPORTED_SCENARIOS = {
    ScenarioName.NORMAL,
    ScenarioName.RETRIEVAL_MISS,
    ScenarioName.CONFLICTING_EVIDENCE,
    ScenarioName.SLOW_RESPONSE,
}


def load_eval_cases(
    dataset_path: str = "data/eval_dataset.json",
    case_set: CaseSet = CaseSet.FULL,
) -> list[EvalCase]:
    with Path(dataset_path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    cases = parse_eval_cases(payload)
    if case_set == CaseSet.PORTABLE:
        return [case for case in cases if case.case_kind == "portable"]
    return cases


class OfflineEvalRunner:
    def __init__(
        self,
        policy_service: PolicyDeskService,
        storage: StorageService,
        dataset_path: str = "data/eval_dataset.json",
        gate_policy_path: str = "data/offline_gate_policy.json",
    ) -> None:
        self.policy_service = policy_service
        self.storage = storage
        self.dataset_path = dataset_path
        self.gate_policy_path = gate_policy_path

    def run(
        self,
        *,
        model_backend,
        prompt_version: str,
        retrieval_config_version: str = "retrieval-config:v1",
        source_snapshot_id: str = "kb-snapshot:current",
        case_set: CaseSet = CaseSet.FULL,
    ) -> OfflineEvalRun:
        return self.run_config(
            OfflineEvalConfig(
                model_backend=model_backend,
                prompt_version=prompt_version,
                retrieval_config_version=retrieval_config_version,
                source_snapshot_id=source_snapshot_id,
                case_set=case_set,
            )
        )

    def run_config(self, config: OfflineEvalConfig) -> OfflineEvalRun:
        return self._run_config(config, cases=None)

    def _run_config(self, config: OfflineEvalConfig, cases: list[EvalCase] | None) -> OfflineEvalRun:
        eval_run_id = str(uuid4())
        gate_policy = self._load_gate_policy()
        loaded_cases = cases or load_eval_cases(self.dataset_path, case_set=config.case_set)
        case_results = []
        skipped_cases: list[SkippedEvalCase] = []
        model_name = "mock-policy-model"

        for case in loaded_cases:
            if not self._case_supported_by_backend(case, config.model_backend):
                skipped_cases.append(
                    SkippedEvalCase(
                        case_id=case.case_id,
                        scenario=case.scenario,
                        reason="Case is not configured for this backend.",
                        reason_code="backend_unsupported",
                    )
                )
                continue
            if config.model_backend == ModelBackend.OPENAI and case.scenario not in OPENAI_SUPPORTED_SCENARIOS:
                skipped_cases.append(
                    SkippedEvalCase(
                        case_id=case.case_id,
                        scenario=case.scenario,
                        reason="Scenario is supported only on the mock backend.",
                        reason_code="scenario_unsupported",
                    )
                )
                continue

            retrieval_config_version = case.retrieval_config_version or config.retrieval_config_version
            source_snapshot_id = case.source_snapshot_id or config.source_snapshot_id
            request = PolicyDeskAssistantRequest(
                question=case.question,
                scenario=case.scenario,
                model_backend=config.model_backend,
                prompt_version=config.prompt_version,
                retrieval_config_version=retrieval_config_version,
                source_snapshot_id=source_snapshot_id,
            )
            response = self.policy_service.respond(request)
            run_record = self.storage.get_run(response.run_id)
            assert run_record is not None
            model_name = run_record.model_name
            result = score_offline_case(
                eval_run_id=eval_run_id,
                raw_case=case,
                response=response,
                run_record=run_record,
                retrieval_config_version=retrieval_config_version,
                source_snapshot_id=source_snapshot_id,
            )
            case_results.append(result)

        gate_result = apply_gate_policy(case_results=case_results, cases=loaded_cases, policy=gate_policy)
        summary = summarize_offline_results(
            eval_run_id=eval_run_id,
            model_backend=config.model_backend,
            model_name=model_name,
            prompt_version=config.prompt_version,
            case_results=case_results,
            skipped_cases=skipped_cases,
            retrieval_config_version=config.retrieval_config_version,
            source_snapshot_id=config.source_snapshot_id,
            gate_policy_version=gate_policy.policy_version,
            by_bucket_breakdown=gate_result.bucket_summaries,
            release_decision=gate_result.release_decision,
            decision_reasons=gate_result.decision_reasons,
            failed_buckets=gate_result.failed_buckets,
        )
        self.storage.write_offline_eval(summary, case_results)
        return OfflineEvalRun(summary=summary, case_results=case_results)

    def compare(self, request: OfflineComparisonRequest) -> OfflineComparisonRun:
        comparison_id = str(uuid4())
        all_cases = load_eval_cases(self.dataset_path, case_set=request.baseline.case_set)
        gate_policy = self._load_gate_policy()
        comparable_cases, excluded_case_ids = self._comparable_cases(
            all_cases,
            baseline_backend=request.baseline.model_backend,
            candidate_backend=request.candidate.model_backend,
        )

        baseline_run = self._run_config(request.baseline, comparable_cases)
        candidate_run = self._run_config(request.candidate, comparable_cases)

        baseline_results = {item.case_id: item for item in baseline_run.case_results}
        candidate_results = {item.case_id: item for item in candidate_run.case_results}
        baseline_skipped = {item.case_id: item for item in baseline_run.summary.skipped_cases}
        candidate_skipped = {item.case_id: item for item in candidate_run.summary.skipped_cases}

        case_deltas: list[OfflineComparisonCaseDelta] = []
        new_failures: list[str] = []
        new_blocking_failures: list[str] = []
        fixed_failures: list[str] = []
        decision_reasons = list(candidate_run.summary.decision_reasons)
        failed_buckets = list(candidate_run.summary.failed_buckets)
        if excluded_case_ids:
            decision_reasons.append(
                f"Comparison evaluated {len(comparable_cases)} common supported cases and excluded {len(excluded_case_ids)} backend-specific stress cases."
            )

        for case in comparable_cases:
            baseline_result = baseline_results.get(case.case_id)
            candidate_result = candidate_results.get(case.case_id)
            introduced: list[str] = []
            fixed: list[str] = []
            new_blockers: list[str] = []
            resolved_blockers: list[str] = []
            new_blocking_regressions: list[str] = []
            resolved_blocking_regressions: list[str] = []
            thresholds = resolve_thresholds(gate_policy, case)

            if baseline_result and candidate_result:
                if baseline_result.passed and not candidate_result.passed:
                    introduced.append("candidate_case_failed")
                    new_failures.append(case.case_id)
                if not baseline_result.passed and candidate_result.passed:
                    fixed.append("candidate_case_fixed")
                    fixed_failures.append(case.case_id)
                new_blockers = sorted(set(candidate_result.regression_blockers) - set(baseline_result.regression_blockers))
                resolved_blockers = sorted(set(baseline_result.regression_blockers) - set(candidate_result.regression_blockers))
                new_blocking_regressions = _blocking_regressions(new_blockers, thresholds.blocker_dimensions)
                resolved_blocking_regressions = _blocking_regressions(
                    resolved_blockers,
                    thresholds.blocker_dimensions,
                )
                for blocker in new_blocking_regressions:
                    new_blocking_failures.append(f"{case.case_id}:{blocker}")
                if resolved_blocking_regressions:
                    fixed.extend(f"resolved_blocker:{blocker}" for blocker in resolved_blocking_regressions)
            elif baseline_result and case.case_id in candidate_skipped:
                introduced.append("candidate_skipped_case")
                new_blocking_regressions.append("candidate_skipped")
                new_failures.append(case.case_id)
                new_blocking_failures.append(f"{case.case_id}:candidate_skipped")
            elif candidate_result and case.case_id in baseline_skipped:
                fixed.append("baseline_skipped_candidate_executed")
                fixed_failures.append(case.case_id)

            delta = OfflineComparisonCaseDelta(
                case_id=case.case_id,
                bucket_id=case.bucket_id,
                bucket_name=case.bucket_name,
                risk_tier=case.risk_tier,
                baseline_score=baseline_result.score_breakdown.total if baseline_result else None,
                candidate_score=candidate_result.score_breakdown.total if candidate_result else None,
                score_delta=_delta(
                    candidate_result.score_breakdown.total if candidate_result else None,
                    baseline_result.score_breakdown.total if baseline_result else None,
                ),
                baseline_weighted_score=baseline_result.weighted_score if baseline_result else None,
                candidate_weighted_score=candidate_result.weighted_score if candidate_result else None,
                weighted_score_delta=_delta(
                    candidate_result.weighted_score if candidate_result else None,
                    baseline_result.weighted_score if baseline_result else None,
                ),
                baseline_passed=baseline_result.passed if baseline_result else None,
                candidate_passed=candidate_result.passed if candidate_result else None,
                baseline_behavior=baseline_result.actual_behavior if baseline_result else None,
                candidate_behavior=candidate_result.actual_behavior if candidate_result else None,
                behavior_changed=(
                    bool(baseline_result and candidate_result)
                    and baseline_result.actual_behavior != candidate_result.actual_behavior
                ),
                new_failures=sorted(dict.fromkeys(introduced)),
                fixed_failures=sorted(dict.fromkeys(fixed)),
                new_blocking_failures=sorted(dict.fromkeys(new_blocking_regressions)),
                resolved_blockers=sorted(dict.fromkeys(resolved_blocking_regressions)),
            )
            case_deltas.append(delta)

            if new_blocking_regressions:
                hard_fail_reasons = comparison_hard_fail_reasons(gate_policy, case, new_blocking_regressions)
                if hard_fail_reasons:
                    decision_reasons.extend(f"{case.case_id}: {reason}" for reason in hard_fail_reasons)

        bucket_deltas = self._bucket_deltas(
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            case_deltas=case_deltas,
            cases=comparable_cases,
        )
        aggregate_deltas = self._aggregate_deltas(baseline_run, candidate_run)
        decision, decision_reasons, failed_buckets = self._comparison_decision(
            gate_policy=gate_policy,
            case_deltas=case_deltas,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            decision_reasons=decision_reasons,
            failed_buckets=failed_buckets,
            new_failures=new_failures,
            new_blocking_failures=new_blocking_failures,
        )

        summary = OfflineComparisonSummary(
            comparison_id=comparison_id,
            created_at=candidate_run.summary.created_at,
            gate_policy_version=gate_policy.policy_version,
            baseline_config=request.baseline,
            candidate_config=request.candidate,
            baseline_summary=baseline_run.summary,
            candidate_summary=candidate_run.summary,
            aggregate_deltas=aggregate_deltas,
            bucket_deltas=bucket_deltas,
            release_decision=decision,
            decision_reasons=sorted(dict.fromkeys(decision_reasons)),
            failed_buckets=sorted(dict.fromkeys(failed_buckets)),
            new_failures=sorted(dict.fromkeys(new_failures)),
            new_blocking_failures=sorted(dict.fromkeys(new_blocking_failures)),
            fixed_failures=sorted(dict.fromkeys(fixed_failures)),
            compared_case_ids=[case.case_id for case in comparable_cases],
            excluded_case_ids=excluded_case_ids,
        )
        comparison_run = OfflineComparisonRun(summary=summary, case_deltas=case_deltas)
        self.storage.write_offline_comparison(summary, case_deltas)
        return comparison_run

    def _aggregate_deltas(self, baseline_run: OfflineEvalRun, candidate_run: OfflineEvalRun) -> dict[str, float]:
        keys = sorted(set(baseline_run.summary.aggregate_metrics) | set(candidate_run.summary.aggregate_metrics))
        deltas: dict[str, float] = {}
        for key in keys:
            deltas[key] = round(
                candidate_run.summary.aggregate_metrics.get(key, 0.0) - baseline_run.summary.aggregate_metrics.get(key, 0.0),
                4,
            )
        deltas["skipped_case_delta"] = float(len(candidate_run.summary.skipped_cases) - len(baseline_run.summary.skipped_cases))
        return deltas

    def _bucket_deltas(
        self,
        *,
        baseline_run: OfflineEvalRun,
        candidate_run: OfflineEvalRun,
        case_deltas: list[OfflineComparisonCaseDelta],
        cases: list[EvalCase],
    ) -> list[OfflineComparisonBucketDelta]:
        case_map = {case.case_id: case for case in cases}
        baseline_buckets = {item.bucket_id: item for item in baseline_run.summary.by_bucket_breakdown}
        candidate_buckets = {item.bucket_id: item for item in candidate_run.summary.by_bucket_breakdown}
        bucket_case_deltas: dict[str, list[OfflineComparisonCaseDelta]] = {}
        for delta in case_deltas:
            bucket_case_deltas.setdefault(delta.bucket_id, []).append(delta)

        deltas: list[OfflineComparisonBucketDelta] = []
        for bucket_id in sorted(set(baseline_buckets) | set(candidate_buckets) | set(bucket_case_deltas)):
            baseline_bucket = baseline_buckets.get(bucket_id)
            candidate_bucket = candidate_buckets.get(bucket_id)
            sample_case = next((case_map[delta.case_id] for delta in bucket_case_deltas.get(bucket_id, []) if delta.case_id in case_map), None)
            risk_tier = (
                candidate_bucket.risk_tier
                if candidate_bucket
                else baseline_bucket.risk_tier
                if baseline_bucket
                else sample_case.risk_tier
            )
            bucket_deltas = bucket_case_deltas.get(bucket_id, [])
            deltas.append(
                OfflineComparisonBucketDelta(
                    bucket_id=bucket_id,
                    bucket_name=(
                        candidate_bucket.bucket_name
                        if candidate_bucket
                        else baseline_bucket.bucket_name
                        if baseline_bucket
                        else sample_case.bucket_name
                    ),
                    risk_tier=risk_tier,
                    baseline_avg_score=baseline_bucket.weighted_avg_score if baseline_bucket else None,
                    candidate_avg_score=candidate_bucket.weighted_avg_score if candidate_bucket else None,
                    avg_score_delta=_delta(
                        candidate_bucket.weighted_avg_score if candidate_bucket else None,
                        baseline_bucket.weighted_avg_score if baseline_bucket else None,
                    ),
                    baseline_pass_rate=baseline_bucket.pass_rate if baseline_bucket else None,
                    candidate_pass_rate=candidate_bucket.pass_rate if candidate_bucket else None,
                    pass_rate_delta=_delta(
                        candidate_bucket.pass_rate if candidate_bucket else None,
                        baseline_bucket.pass_rate if baseline_bucket else None,
                    ),
                    new_failures=sum(1 for delta in bucket_deltas if any(item == "candidate_case_failed" for item in delta.new_failures)),
                    fixed_failures=sum(1 for delta in bucket_deltas if any(item == "candidate_case_fixed" for item in delta.fixed_failures)),
                    new_blocking_failures=sum(len(delta.new_blocking_failures) for delta in bucket_deltas),
                )
            )
        return deltas

    def _comparison_decision(
        self,
        *,
        gate_policy: GatePolicy,
        case_deltas: list[OfflineComparisonCaseDelta],
        baseline_run: OfflineEvalRun,
        candidate_run: OfflineEvalRun,
        decision_reasons: list[str],
        failed_buckets: list[str],
        new_failures: list[str],
        new_blocking_failures: list[str],
    ) -> tuple[ReleaseDecision, list[str], list[str]]:
        decision = candidate_run.summary.release_decision
        global_thresholds = gate_policy.global_defaults
        max_new_failures = global_thresholds.max_new_failures if global_thresholds.max_new_failures is not None else 1
        max_new_blocking_failures = (
            global_thresholds.max_new_blocking_failures if global_thresholds.max_new_blocking_failures is not None else 0
        )

        if len(new_blocking_failures) > max_new_blocking_failures:
            decision = _max_decision(decision, ReleaseDecision.FAIL)
            decision_reasons.append(
                f"Candidate introduced {len(new_blocking_failures)} new blocking failures which exceeds the allowed maximum of {max_new_blocking_failures}."
            )
        elif new_blocking_failures:
            decision = _max_decision(decision, ReleaseDecision.WARN)
            decision_reasons.append(f"Candidate introduced {len(new_blocking_failures)} new blocking failures.")

        if len(new_failures) > max_new_failures:
            decision = _max_decision(decision, ReleaseDecision.FAIL)
            decision_reasons.append(
                f"Candidate introduced {len(new_failures)} new failures which exceeds the allowed maximum of {max_new_failures}."
            )
        elif new_failures:
            decision = _max_decision(decision, ReleaseDecision.WARN)
            decision_reasons.append(f"Candidate introduced {len(new_failures)} new failures.")

        critical_regressions = [
            delta.case_id
            for delta in case_deltas
            if delta.risk_tier.value == "critical"
            and (
                delta.new_blocking_failures
                or "candidate_case_failed" in delta.new_failures
                or (delta.score_delta is not None and delta.score_delta < 0)
            )
        ]
        if critical_regressions:
            decision = _max_decision(decision, ReleaseDecision.FAIL)
            failed_buckets.extend(
                delta.bucket_id
                for delta in case_deltas
                if delta.case_id in critical_regressions
            )
            decision_reasons.append(
                f"Critical bucket regressions were introduced in cases: {', '.join(sorted(critical_regressions))}."
            )

        high_risk_regressions = [
            delta.case_id
            for delta in case_deltas
            if delta.risk_tier.value == "high"
            and delta.score_delta is not None
            and delta.score_delta <= -0.10
        ]
        if high_risk_regressions:
            decision = _max_decision(decision, ReleaseDecision.WARN)
            decision_reasons.append(
                f"High-risk cases regressed by at least 0.10 weighted points: {', '.join(sorted(high_risk_regressions))}."
            )

        if len(candidate_run.summary.skipped_cases) > len(baseline_run.summary.skipped_cases):
            decision = _max_decision(decision, ReleaseDecision.WARN)
            decision_reasons.append("Candidate skipped more eval cases than the baseline.")

        return decision, sorted(dict.fromkeys(decision_reasons)), sorted(dict.fromkeys(failed_buckets))

    def _load_gate_policy(self) -> GatePolicy:
        return load_gate_policy(self.gate_policy_path)

    def _case_supported_by_backend(self, case: EvalCase, model_backend: ModelBackend) -> bool:
        return model_backend in case.supported_backends

    def _comparable_cases(
        self,
        cases: list[EvalCase],
        *,
        baseline_backend: ModelBackend,
        candidate_backend: ModelBackend,
    ) -> tuple[list[EvalCase], list[str]]:
        comparable: list[EvalCase] = []
        excluded: list[str] = []
        for case in cases:
            if self._case_supported_by_backend(case, baseline_backend) and self._case_supported_by_backend(case, candidate_backend):
                comparable.append(case)
            else:
                excluded.append(case.case_id)
        return comparable, excluded


def _delta(candidate_value: float | None, baseline_value: float | None) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    return round(candidate_value - baseline_value, 4)


def _max_decision(left: ReleaseDecision, right: ReleaseDecision) -> ReleaseDecision:
    order = {
        ReleaseDecision.PASS: 0,
        ReleaseDecision.WARN: 1,
        ReleaseDecision.FAIL: 2,
    }
    return left if order[left] >= order[right] else right


def _blocking_regressions(blockers: list[str], blocker_dimensions: list[str]) -> list[str]:
    return sorted(set(blockers) & set(blocker_dimensions))
