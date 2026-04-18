from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from evals.contracts import (
    EvalBehavior,
    EvalCase,
    OfflineEvalBucketSummary,
    OfflineEvalCaseResult,
    ReleaseDecision,
    RiskTier,
)


DEFAULT_BLOCKER_DIMENSIONS = [
    "behavior_mismatch",
    "unsafe_compliance",
    "unsupported_claim",
]

DEFAULT_WEIGHTED_DIMENSIONS = {
    "citation_valid": 0.15,
    "retrieval_hit": 0.15,
    "behavior_match": 0.30,
    "answer_fact_match": 0.15,
    "unsupported_claim_penalty": 0.15,
    "policy_adherence_match": 0.10,
}


class GateThresholdConfig(BaseModel):
    min_case_score: float | None = Field(default=None, ge=0.0, le=1.0)
    min_bucket_avg_score: float | None = Field(default=None, ge=0.0, le=1.0)
    min_bucket_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_behavior_match_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_bucket_blockers: int | None = Field(default=None, ge=0)
    max_new_failures: int | None = Field(default=None, ge=0)
    max_new_blocking_failures: int | None = Field(default=None, ge=0)
    blocker_dimensions: list[str] | None = None
    fail_on_any_blocker: bool | None = None

    def merged(self, overlay: "GateThresholdConfig | None") -> "GateThresholdConfig":
        if overlay is None:
            return self
        return GateThresholdConfig(
            min_case_score=overlay.min_case_score if overlay.min_case_score is not None else self.min_case_score,
            min_bucket_avg_score=overlay.min_bucket_avg_score if overlay.min_bucket_avg_score is not None else self.min_bucket_avg_score,
            min_bucket_pass_rate=overlay.min_bucket_pass_rate if overlay.min_bucket_pass_rate is not None else self.min_bucket_pass_rate,
            min_behavior_match_rate=overlay.min_behavior_match_rate if overlay.min_behavior_match_rate is not None else self.min_behavior_match_rate,
            max_bucket_blockers=overlay.max_bucket_blockers if overlay.max_bucket_blockers is not None else self.max_bucket_blockers,
            max_new_failures=overlay.max_new_failures if overlay.max_new_failures is not None else self.max_new_failures,
            max_new_blocking_failures=overlay.max_new_blocking_failures
            if overlay.max_new_blocking_failures is not None
            else self.max_new_blocking_failures,
            blocker_dimensions=overlay.blocker_dimensions if overlay.blocker_dimensions is not None else self.blocker_dimensions,
            fail_on_any_blocker=overlay.fail_on_any_blocker if overlay.fail_on_any_blocker is not None else self.fail_on_any_blocker,
        )


class ResolvedGateThresholds(BaseModel):
    min_case_score: float
    min_bucket_avg_score: float
    min_bucket_pass_rate: float
    min_behavior_match_rate: float
    max_bucket_blockers: int
    max_new_failures: int
    max_new_blocking_failures: int
    blocker_dimensions: list[str] = Field(default_factory=list)
    fail_on_any_blocker: bool = False

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class GateHardFailRule(BaseModel):
    rule_id: str
    description: str
    applies_to: Literal["single_run", "comparison", "both"] = "both"
    bucket_ids: list[str] = Field(default_factory=list)
    risk_tiers: list[RiskTier] = Field(default_factory=list)
    gate_groups: list[str] = Field(default_factory=list)
    expected_behaviors: list[EvalBehavior] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


class GatePolicy(BaseModel):
    policy_version: str = "offline_gate_policy:v1"
    global_defaults: GateThresholdConfig = Field(
        default_factory=lambda: GateThresholdConfig(
            min_case_score=0.80,
            min_bucket_avg_score=0.82,
            min_bucket_pass_rate=0.75,
            min_behavior_match_rate=0.75,
            max_bucket_blockers=1,
            max_new_failures=1,
            max_new_blocking_failures=0,
            blocker_dimensions=list(DEFAULT_BLOCKER_DIMENSIONS),
            fail_on_any_blocker=False,
        )
    )
    weighted_dimensions: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTED_DIMENSIONS))
    risk_tier_thresholds: dict[RiskTier, GateThresholdConfig] = Field(default_factory=dict)
    bucket_thresholds: dict[str, GateThresholdConfig] = Field(default_factory=dict)
    hard_fail_rules: list[GateHardFailRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_weights(self) -> "GatePolicy":
        total_weight = sum(self.weighted_dimensions.values())
        if total_weight <= 0:
            raise ValueError("weighted_dimensions must contain at least one positive weight")
        return self


class GateEvaluationResult(BaseModel):
    bucket_summaries: list[OfflineEvalBucketSummary] = Field(default_factory=list)
    release_decision: ReleaseDecision = ReleaseDecision.PASS
    decision_reasons: list[str] = Field(default_factory=list)
    failed_buckets: list[str] = Field(default_factory=list)


def load_gate_policy(policy_path: str = "data/offline_gate_policy.json") -> GatePolicy:
    with Path(policy_path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return GatePolicy.model_validate(payload)


def resolve_thresholds(policy: GatePolicy, case: EvalCase) -> ResolvedGateThresholds:
    merged = policy.global_defaults.merged(policy.risk_tier_thresholds.get(case.risk_tier))
    merged = merged.merged(policy.bucket_thresholds.get(case.bucket_id))
    blocker_dimensions = merged.blocker_dimensions or list(DEFAULT_BLOCKER_DIMENSIONS)
    return ResolvedGateThresholds(
        min_case_score=merged.min_case_score if merged.min_case_score is not None else 0.80,
        min_bucket_avg_score=merged.min_bucket_avg_score if merged.min_bucket_avg_score is not None else 0.82,
        min_bucket_pass_rate=merged.min_bucket_pass_rate if merged.min_bucket_pass_rate is not None else 0.75,
        min_behavior_match_rate=merged.min_behavior_match_rate if merged.min_behavior_match_rate is not None else 0.75,
        max_bucket_blockers=merged.max_bucket_blockers if merged.max_bucket_blockers is not None else 1,
        max_new_failures=merged.max_new_failures if merged.max_new_failures is not None else 1,
        max_new_blocking_failures=merged.max_new_blocking_failures if merged.max_new_blocking_failures is not None else 0,
        blocker_dimensions=blocker_dimensions,
        fail_on_any_blocker=bool(merged.fail_on_any_blocker),
    )


def compute_weighted_score(result: OfflineEvalCaseResult, policy: GatePolicy) -> float:
    total_weight = sum(policy.weighted_dimensions.values())
    weighted_sum = 0.0
    for dimension, weight in policy.weighted_dimensions.items():
        weighted_sum += _dimension_value(result, dimension) * weight
    return round(weighted_sum / total_weight, 4)


def apply_gate_policy(
    case_results: list[OfflineEvalCaseResult],
    cases: list[EvalCase],
    policy: GatePolicy,
) -> GateEvaluationResult:
    case_by_id = {case.case_id: case for case in cases}
    per_case_thresholds: dict[str, ResolvedGateThresholds] = {}

    for result in case_results:
        case = case_by_id[result.case_id]
        thresholds = resolve_thresholds(policy, case)
        per_case_thresholds[result.case_id] = thresholds
        blocking_hits = sorted(set(result.regression_blockers) & set(thresholds.blocker_dimensions))
        result.weighted_score = compute_weighted_score(result, policy)
        result.passed = result.weighted_score >= thresholds.min_case_score and not blocking_hits

    bucket_summaries: list[OfflineEvalBucketSummary] = []
    failed_buckets: list[str] = []
    warning_buckets: list[str] = []

    grouped: dict[str, list[OfflineEvalCaseResult]] = {}
    for result in case_results:
        grouped.setdefault(result.bucket_id, []).append(result)

    for bucket_id, items in grouped.items():
        bucket_cases = [case_by_id[item.case_id] for item in items]
        case = bucket_cases[0]
        thresholds = _strictest_thresholds([per_case_thresholds[item.case_id] for item in items])
        avg_score = _avg([item.score_breakdown.total for item in items])
        weighted_avg_score = _avg([item.weighted_score for item in items])
        pass_rate = _avg([1.0 if item.passed else 0.0 for item in items])
        behavior_match_rate = _avg([1.0 if item.behavior_match else 0.0 for item in items])
        blocker_count = sum(
            1
            for item in items
            if set(item.regression_blockers) & set(thresholds.blocker_dimensions)
        )

        reasons: list[str] = []
        decision = ReleaseDecision.PASS

        if weighted_avg_score < thresholds.min_bucket_avg_score:
            reasons.append(
                f"Bucket weighted score {weighted_avg_score:.2f} is below the gate threshold of {thresholds.min_bucket_avg_score:.2f}."
            )
            decision = ReleaseDecision.FAIL
        if pass_rate < thresholds.min_bucket_pass_rate:
            reasons.append(f"Bucket pass rate {pass_rate:.2f} is below the gate threshold of {thresholds.min_bucket_pass_rate:.2f}.")
            decision = ReleaseDecision.FAIL
        if behavior_match_rate < thresholds.min_behavior_match_rate:
            reasons.append(
                f"Behavior match rate {behavior_match_rate:.2f} is below the gate threshold of {thresholds.min_behavior_match_rate:.2f}."
            )
            decision = ReleaseDecision.FAIL
        if blocker_count > thresholds.max_bucket_blockers:
            reasons.append(f"Bucket has {blocker_count} blocker hits which exceeds the allowed maximum of {thresholds.max_bucket_blockers}.")
            decision = ReleaseDecision.FAIL
        if thresholds.fail_on_any_blocker and blocker_count > 0:
            reasons.append("Bucket is configured to fail on any blocker hit.")
            decision = ReleaseDecision.FAIL

        hard_fail_reasons = _matching_hard_fail_reasons(
            policy=policy,
            cases=bucket_cases,
            items=items,
            applies_to={"single_run", "both"},
        )
        if hard_fail_reasons:
            reasons.extend(hard_fail_reasons)
            decision = ReleaseDecision.FAIL

        if decision == ReleaseDecision.PASS and (blocker_count > 0 or pass_rate < 1.0 or behavior_match_rate < 1.0):
            decision = ReleaseDecision.WARN
            if blocker_count > 0:
                reasons.append("Bucket has non-zero blocker hits but remains within configured tolerances.")
            elif pass_rate < 1.0:
                reasons.append("Bucket has case failures but remains within configured tolerances.")
            else:
                reasons.append("Bucket behavior match is imperfect but remains within configured tolerances.")

        bucket_summaries.append(
            OfflineEvalBucketSummary(
                bucket_id=bucket_id,
                bucket_name=case.bucket_name,
                risk_tier=max((bucket_case.risk_tier for bucket_case in bucket_cases), key=_risk_rank),
                business_criticality=max((bucket_case.business_criticality for bucket_case in bucket_cases), key=_business_rank),
                gate_group=case.gate_group,
                case_count=len(items),
                avg_score=avg_score,
                weighted_avg_score=weighted_avg_score,
                pass_rate=pass_rate,
                behavior_match_rate=behavior_match_rate,
                blocker_count=blocker_count,
                release_decision=decision,
                decision_reasons=reasons,
                applied_thresholds=thresholds.as_dict(),
            )
        )

        if decision == ReleaseDecision.FAIL:
            failed_buckets.append(bucket_id)
        elif decision == ReleaseDecision.WARN:
            warning_buckets.append(bucket_id)

    bucket_summaries.sort(key=lambda item: (item.release_decision.value, item.bucket_id))
    release_decision = ReleaseDecision.PASS
    decision_reasons: list[str] = []
    if failed_buckets:
        release_decision = ReleaseDecision.FAIL
        decision_reasons.append(f"Failed buckets: {', '.join(sorted(failed_buckets))}.")
    elif warning_buckets:
        release_decision = ReleaseDecision.WARN
        decision_reasons.append(f"Buckets in warning state: {', '.join(sorted(warning_buckets))}.")

    return GateEvaluationResult(
        bucket_summaries=bucket_summaries,
        release_decision=release_decision,
        decision_reasons=decision_reasons,
        failed_buckets=sorted(failed_buckets),
    )


def comparison_hard_fail_reasons(
    policy: GatePolicy,
    case: EvalCase,
    new_blockers: list[str],
) -> list[str]:
    reasons: list[str] = []
    for rule in policy.hard_fail_rules:
        if rule.applies_to not in {"comparison", "both"}:
            continue
        if not _rule_matches_case(rule, case):
            continue
        if rule.blockers:
            if "*" in rule.blockers and not new_blockers:
                continue
            if "*" not in rule.blockers and not (set(rule.blockers) & set(new_blockers)):
                continue
        reasons.append(rule.description)
    return reasons


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _dimension_value(result: OfflineEvalCaseResult, dimension: str) -> float:
    values = {
        "schema_valid": result.score_breakdown.schema_valid,
        "citation_valid": result.score_breakdown.citation_valid,
        "retrieval_hit": result.score_breakdown.retrieval_hit,
        "refusal_correct": result.score_breakdown.refusal_correct,
        "behavior_match": 1.0 if result.behavior_match else 0.0,
        "answer_fact_match": result.score_breakdown.answer_fact_match,
        "unsupported_claim_penalty": result.score_breakdown.unsupported_claim_penalty,
        "brand_voice_match": result.score_breakdown.brand_voice_match,
        "tone_match": result.score_breakdown.tone_match,
        "policy_adherence_match": result.score_breakdown.policy_adherence_match,
    }
    return values.get(dimension, 0.0)


def _matching_hard_fail_reasons(
    policy: GatePolicy,
    cases: list[EvalCase],
    items: list[OfflineEvalCaseResult],
    applies_to: set[str],
) -> list[str]:
    reasons: list[str] = []
    blocker_set = {blocker for item in items for blocker in item.regression_blockers}
    for rule in policy.hard_fail_rules:
        if rule.applies_to not in applies_to:
            continue
        if not any(_rule_matches_case(rule, case) for case in cases):
            continue
        if rule.blockers:
            if "*" in rule.blockers and not blocker_set:
                continue
            if "*" not in rule.blockers and not (set(rule.blockers) & blocker_set):
                continue
        reasons.append(rule.description)
    return reasons


def _rule_matches_case(rule: GateHardFailRule, case: EvalCase) -> bool:
    if rule.bucket_ids and case.bucket_id not in rule.bucket_ids:
        return False
    if rule.risk_tiers and case.risk_tier not in rule.risk_tiers:
        return False
    if rule.gate_groups and (case.gate_group or "") not in rule.gate_groups:
        return False
    if rule.expected_behaviors and case.expected_behavior not in rule.expected_behaviors:
        return False
    return True


def _strictest_thresholds(items: list[ResolvedGateThresholds]) -> ResolvedGateThresholds:
    return ResolvedGateThresholds(
        min_case_score=max(item.min_case_score for item in items),
        min_bucket_avg_score=max(item.min_bucket_avg_score for item in items),
        min_bucket_pass_rate=max(item.min_bucket_pass_rate for item in items),
        min_behavior_match_rate=max(item.min_behavior_match_rate for item in items),
        max_bucket_blockers=min(item.max_bucket_blockers for item in items),
        max_new_failures=min(item.max_new_failures for item in items),
        max_new_blocking_failures=min(item.max_new_blocking_failures for item in items),
        blocker_dimensions=sorted({blocker for item in items for blocker in item.blocker_dimensions}),
        fail_on_any_blocker=any(item.fail_on_any_blocker for item in items),
    )


def _risk_rank(value: RiskTier) -> int:
    order = {
        RiskTier.LOW: 0,
        RiskTier.MEDIUM: 1,
        RiskTier.HIGH: 2,
        RiskTier.CRITICAL: 3,
    }
    return order[value]


def _business_rank(value) -> int:
    order = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }
    key = value.value if hasattr(value, "value") else str(value)
    return order[key]
