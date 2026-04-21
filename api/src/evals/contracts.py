from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from model.contracts import (
    Difficulty,
    FailureCategory,
    FailureReason,
    FinalDisposition,
    ModelBackend,
    Outcome,
    ReviewPriority,
    ReviewStatus,
    ScenarioName,
    SuspiciousFlag,
)
from prompts.registry import DEFAULT_PROMPT_VERSION, PROMPT_VERSION_PATTERN


class EvalTag(str, Enum):
    DIRECT_ANSWERABLE = "direct_answerable"
    SHOULD_REFUSE_MISSING_EVIDENCE = "should_refuse_missing_evidence"
    SHOULD_REFUSE_CONFLICTING_EVIDENCE = "should_refuse_conflicting_evidence"
    RETRIEVAL_MISS = "retrieval_miss"
    MALFORMED_OUTPUT = "malformed_output"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    TONE_AND_BRAND_VOICE = "tone_and_brand_voice"
    POLICY_BOUNDARY_CASES = "policy_boundary_cases"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BusinessCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvalBehavior(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    HUMAN_REVIEW = "human_review"


class ReleaseDecision(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class CaseSet(str, Enum):
    PORTABLE = "portable"
    FULL = "full"


SAFETY_SCENARIOS = {
    ScenarioName.CONFLICTING_EVIDENCE.value,
    ScenarioName.UNSUPPORTED_ANSWER.value,
    ScenarioName.WRONG_REFUSAL.value,
}

MOCK_ONLY_SCENARIOS = {
    ScenarioName.MALFORMED_JSON.value,
    ScenarioName.UNSUPPORTED_ANSWER.value,
    ScenarioName.WRONG_REFUSAL.value,
}


def _normalize_text_token(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip().replace("_", "-").replace(" ", "-")
    raw = raw.lower().strip("-")
    return raw or fallback


def _humanize_token(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").title()


def _value_text(value: Any, fallback: str = "") -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if value is None:
        return fallback
    return str(value)


class EvalCase(BaseModel):
    case_id: str
    question: str
    scenario: ScenarioName
    expected_should_refuse: bool = False
    required_citation_ids: list[str] = Field(default_factory=list)
    acceptable_answer_facts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    tags: list[EvalTag] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    bucket_id: str = "general"
    bucket_name: str = "General"
    risk_tier: RiskTier = RiskTier.MEDIUM
    business_criticality: BusinessCriticality = BusinessCriticality.MEDIUM
    expected_behavior: EvalBehavior = EvalBehavior.ANSWER
    refusal_reason_expected: str | None = None
    source_snapshot_id: str | None = None
    retrieval_config_version: str | None = None
    # Reference-only human label text. The offline gates do not score this today.
    reference_response_text: str | None = None
    label_notes: str | None = None
    is_ambiguous: bool = False
    gate_group: str | None = None
    owner: str | None = None
    case_kind: Literal["portable", "stress"] = "portable"
    supported_backends: list[ModelBackend] = Field(
        default_factory=lambda: [ModelBackend.MOCK, ModelBackend.OPENAI]
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        payload.setdefault("required_citation_ids", [])
        payload.setdefault("acceptable_answer_facts", [])
        payload.setdefault("forbidden_claims", [])
        payload.setdefault("tags", [])
        payload.setdefault("difficulty", Difficulty.MEDIUM)

        expected_behavior = payload.get("expected_behavior") or cls._default_expected_behavior(payload)
        payload["expected_behavior"] = expected_behavior
        payload["expected_should_refuse"] = payload.get(
            "expected_should_refuse",
            expected_behavior in {EvalBehavior.ABSTAIN, EvalBehavior.REFUSE},
        )

        bucket_id = payload.get("bucket_id") or cls._default_bucket_id(payload)
        payload["bucket_id"] = _normalize_text_token(bucket_id, "general")
        payload["bucket_name"] = payload.get("bucket_name") or _humanize_token(payload["bucket_id"])
        payload["risk_tier"] = payload.get("risk_tier") or cls._default_risk_tier(payload)
        payload["business_criticality"] = payload.get("business_criticality") or cls._default_business_criticality(payload)
        payload["refusal_reason_expected"] = payload.get("refusal_reason_expected") or cls._default_refusal_reason(payload)
        payload["gate_group"] = payload.get("gate_group") or cls._default_gate_group(payload)
        payload["case_kind"] = payload.get("case_kind") or cls._default_case_kind(payload)
        payload["supported_backends"] = payload.get("supported_backends") or cls._default_supported_backends(payload)
        return payload

    @model_validator(mode="after")
    def sync_legacy_flags(self) -> "EvalCase":
        self.bucket_id = _normalize_text_token(self.bucket_id, "general")
        if not self.bucket_name:
            self.bucket_name = _humanize_token(self.bucket_id)
        self.expected_should_refuse = self.expected_behavior in {EvalBehavior.ABSTAIN, EvalBehavior.REFUSE}
        return self

    @staticmethod
    def _default_bucket_id(payload: dict[str, Any]) -> str:
        tags = [_value_text(tag) for tag in payload.get("tags", [])]
        if tags:
            return tags[0]
        scenario = _value_text(payload.get("scenario"), "general")
        return scenario

    @staticmethod
    def _default_expected_behavior(payload: dict[str, Any]) -> EvalBehavior:
        if "expected_behavior" in payload and payload["expected_behavior"]:
            value = payload["expected_behavior"]
            return value if isinstance(value, EvalBehavior) else EvalBehavior(_value_text(value))
        scenario = _value_text(payload.get("scenario"), "normal")
        tags = {_value_text(tag) for tag in payload.get("tags", [])}
        if payload.get("expected_should_refuse"):
            if scenario == ScenarioName.CONFLICTING_EVIDENCE.value or EvalTag.SHOULD_REFUSE_CONFLICTING_EVIDENCE.value in tags:
                return EvalBehavior.REFUSE
            return EvalBehavior.ABSTAIN
        if scenario == ScenarioName.UNSUPPORTED_ANSWER.value:
            return EvalBehavior.HUMAN_REVIEW
        return EvalBehavior.ANSWER

    @staticmethod
    def _default_risk_tier(payload: dict[str, Any]) -> RiskTier:
        scenario = _value_text(payload.get("scenario"), "normal")
        tags = {_value_text(tag) for tag in payload.get("tags", [])}
        difficulty = _value_text(payload.get("difficulty"), Difficulty.MEDIUM.value)
        if EvalTag.UNSUPPORTED_CLAIM.value in tags:
            return RiskTier.CRITICAL
        if scenario in SAFETY_SCENARIOS:
            return RiskTier.HIGH
        if EvalTag.POLICY_BOUNDARY_CASES.value in tags or EvalTag.SHOULD_REFUSE_CONFLICTING_EVIDENCE.value in tags:
            return RiskTier.HIGH
        if EvalTag.TONE_AND_BRAND_VOICE.value in tags:
            return RiskTier.LOW
        if difficulty == Difficulty.HARD.value:
            return RiskTier.HIGH
        if difficulty == Difficulty.EASY.value:
            return RiskTier.LOW
        return RiskTier.MEDIUM

    @staticmethod
    def _default_business_criticality(payload: dict[str, Any]) -> BusinessCriticality:
        tags = {_value_text(tag) for tag in payload.get("tags", [])}
        scenario = _value_text(payload.get("scenario"), "normal")
        if scenario in {ScenarioName.UNSUPPORTED_ANSWER.value, ScenarioName.CONFLICTING_EVIDENCE.value}:
            return BusinessCriticality.HIGH
        if EvalTag.UNSUPPORTED_CLAIM.value in tags or EvalTag.POLICY_BOUNDARY_CASES.value in tags:
            return BusinessCriticality.HIGH
        if EvalTag.TONE_AND_BRAND_VOICE.value in tags:
            return BusinessCriticality.LOW
        return BusinessCriticality.MEDIUM

    @staticmethod
    def _default_refusal_reason(payload: dict[str, Any]) -> str | None:
        expected_behavior = EvalCase._default_expected_behavior(payload)
        if expected_behavior == EvalBehavior.ABSTAIN:
            return "insufficient_evidence"
        if expected_behavior == EvalBehavior.REFUSE:
            scenario = _value_text(payload.get("scenario"), "normal")
            if scenario == ScenarioName.CONFLICTING_EVIDENCE.value:
                return "conflicting_evidence"
            return "out_of_scope"
        return None

    @staticmethod
    def _default_gate_group(payload: dict[str, Any]) -> str:
        tags = {_value_text(tag) for tag in payload.get("tags", [])}
        scenario = _value_text(payload.get("scenario"), "normal")
        if scenario in SAFETY_SCENARIOS:
            return "safety"
        if EvalTag.UNSUPPORTED_CLAIM.value in tags or EvalTag.POLICY_BOUNDARY_CASES.value in tags:
            return "safety"
        if EvalTag.TONE_AND_BRAND_VOICE.value in tags:
            return "quality"
        return "general"

    @staticmethod
    def _default_case_kind(payload: dict[str, Any]) -> Literal["portable", "stress"]:
        scenario = _value_text(payload.get("scenario"), "normal")
        if scenario in MOCK_ONLY_SCENARIOS:
            return "stress"
        return "portable"

    @staticmethod
    def _default_supported_backends(payload: dict[str, Any]) -> list[ModelBackend]:
        scenario = _value_text(payload.get("scenario"), "normal")
        if scenario in MOCK_ONLY_SCENARIOS:
            return [ModelBackend.MOCK]
        return [ModelBackend.MOCK, ModelBackend.OPENAI]


class OfflineScoreBreakdown(BaseModel):
    schema_valid: float
    citation_valid: float
    retrieval_hit: float
    refusal_correct: float
    answer_fact_match: float
    unsupported_claim_penalty: float
    brand_voice_match: float
    tone_match: float
    policy_adherence_match: float

    @computed_field
    @property
    def total(self) -> float:
        values = [
            self.schema_valid,
            self.citation_valid,
            self.retrieval_hit,
            self.refusal_correct,
            self.answer_fact_match,
            self.unsupported_claim_penalty,
            self.brand_voice_match,
            self.tone_match,
            self.policy_adherence_match,
        ]
        return round(sum(values) / len(values), 4)


class OfflineEvalCaseResult(BaseModel):
    eval_run_id: str
    case_id: str
    scenario: ScenarioName
    prompt_version: str
    model_backend: ModelBackend
    model_name: str
    retrieval_config_version: str | None = None
    source_snapshot_id: str | None = None
    bucket_id: str
    bucket_name: str
    risk_tier: RiskTier
    business_criticality: BusinessCriticality
    gate_group: str | None = None
    score_breakdown: OfflineScoreBreakdown
    weighted_score: float = 0.0
    passed: bool
    expected_should_refuse: bool
    expected_behavior: EvalBehavior
    actual_behavior: EvalBehavior
    behavior_match: bool
    actual_outcome: str
    required_citation_ids: list[str]
    actual_citations: list[str]
    actual_failure_reasons: list[FailureReason] = Field(default_factory=list)
    actual_failure_categories: list[FailureCategory] = Field(default_factory=list)
    suspicious_flags: list[SuspiciousFlag] = Field(default_factory=list)
    regression_blockers: list[str] = Field(default_factory=list)
    answer_preview: str
    run_id: str
    trace_id: str


class SkippedEvalCase(BaseModel):
    case_id: str
    scenario: ScenarioName
    reason: str
    reason_code: str | None = None


class OfflineEvalBucketSummary(BaseModel):
    bucket_id: str
    bucket_name: str
    risk_tier: RiskTier
    business_criticality: BusinessCriticality
    gate_group: str | None = None
    case_count: int
    avg_score: float
    weighted_avg_score: float
    pass_rate: float
    behavior_match_rate: float
    blocker_count: int
    release_decision: ReleaseDecision = ReleaseDecision.PASS
    decision_reasons: list[str] = Field(default_factory=list)
    applied_thresholds: dict[str, Any] = Field(default_factory=dict)


class OfflineEvalSummary(BaseModel):
    eval_run_id: str
    created_at: datetime
    model_backend: ModelBackend
    model_name: str
    prompt_version: str
    retrieval_config_version: str | None = None
    source_snapshot_id: str | None = None
    gate_policy_version: str | None = None
    aggregate_metrics: dict[str, float]
    by_scenario_breakdown: dict[str, dict[str, float]]
    by_bucket_breakdown: list[OfflineEvalBucketSummary] = Field(default_factory=list)
    failure_taxonomy_counts: dict[str, int]
    behavior_taxonomy_counts: dict[str, int] = Field(default_factory=dict)
    worst_case_ids: list[str]
    release_decision: ReleaseDecision = ReleaseDecision.PASS
    decision_reasons: list[str] = Field(default_factory=list)
    failed_buckets: list[str] = Field(default_factory=list)
    new_failures: list[str] = Field(default_factory=list)
    new_blocking_failures: list[str] = Field(default_factory=list)
    fixed_failures: list[str] = Field(default_factory=list)
    skipped_cases: list[SkippedEvalCase] = Field(default_factory=list)


class OfflineEvalRun(BaseModel):
    summary: OfflineEvalSummary
    case_results: list[OfflineEvalCaseResult]


class OfflineEvalConfig(BaseModel):
    label: str | None = None
    model_backend: ModelBackend = ModelBackend.MOCK
    prompt_version: str = Field(default=DEFAULT_PROMPT_VERSION, pattern=PROMPT_VERSION_PATTERN)
    retrieval_config_version: str = "retrieval-config:v1"
    source_snapshot_id: str = "kb-snapshot:current"
    case_set: CaseSet = CaseSet.FULL


class OfflineComparisonRequest(BaseModel):
    baseline: OfflineEvalConfig
    candidate: OfflineEvalConfig

    @model_validator(mode="after")
    def validate_shared_case_set(self) -> "OfflineComparisonRequest":
        if self.baseline.case_set != self.candidate.case_set:
            raise ValueError("baseline and candidate must use the same case_set")
        return self


class OfflineComparisonCaseDelta(BaseModel):
    case_id: str
    bucket_id: str
    bucket_name: str
    risk_tier: RiskTier
    baseline_score: float | None = None
    candidate_score: float | None = None
    score_delta: float | None = None
    baseline_weighted_score: float | None = None
    candidate_weighted_score: float | None = None
    weighted_score_delta: float | None = None
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    baseline_behavior: EvalBehavior | None = None
    candidate_behavior: EvalBehavior | None = None
    behavior_changed: bool = False
    new_failures: list[str] = Field(default_factory=list)
    fixed_failures: list[str] = Field(default_factory=list)
    new_blocking_failures: list[str] = Field(default_factory=list)
    resolved_blockers: list[str] = Field(default_factory=list)


class OfflineComparisonBucketDelta(BaseModel):
    bucket_id: str
    bucket_name: str
    risk_tier: RiskTier
    baseline_avg_score: float | None = None
    candidate_avg_score: float | None = None
    avg_score_delta: float | None = None
    baseline_pass_rate: float | None = None
    candidate_pass_rate: float | None = None
    pass_rate_delta: float | None = None
    new_failures: int = 0
    fixed_failures: int = 0
    new_blocking_failures: int = 0


class OfflineComparisonSummary(BaseModel):
    comparison_id: str
    created_at: datetime
    gate_policy_version: str | None = None
    baseline_config: OfflineEvalConfig
    candidate_config: OfflineEvalConfig
    baseline_summary: OfflineEvalSummary
    candidate_summary: OfflineEvalSummary
    aggregate_deltas: dict[str, float]
    bucket_deltas: list[OfflineComparisonBucketDelta] = Field(default_factory=list)
    release_decision: ReleaseDecision = ReleaseDecision.PASS
    decision_reasons: list[str] = Field(default_factory=list)
    failed_buckets: list[str] = Field(default_factory=list)
    new_failures: list[str] = Field(default_factory=list)
    new_blocking_failures: list[str] = Field(default_factory=list)
    fixed_failures: list[str] = Field(default_factory=list)
    compared_case_ids: list[str] = Field(default_factory=list)
    excluded_case_ids: list[str] = Field(default_factory=list)


class OfflineComparisonRun(BaseModel):
    summary: OfflineComparisonSummary
    case_deltas: list[OfflineComparisonCaseDelta]


class ReviewQueueItem(BaseModel):
    review_queue_item_id: str
    run_id: str
    trace_id: str
    online_score_total: float
    review_priority: ReviewPriority
    suspicious_flags: list[SuspiciousFlag] = Field(default_factory=list)
    review_source: str = "runtime"
    review_reason: str | None = None
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer_label: str | None = None
    reviewer_notes: str | None = None
    final_disposition: FinalDisposition | None = None
    promote_to_offline_eval: bool = False
    should_have_outcome: Outcome | None = None
    should_have_response_text: str | None = None


class ReviewerAnnotation(BaseModel):
    reviewer_label: str
    reviewer_notes: str | None = None
    review_status: ReviewStatus
    final_disposition: FinalDisposition | None = None
    promote_to_offline_eval: bool = False
    should_have_outcome: Outcome | None = None
    should_have_response_text: str | None = None


def parse_eval_cases(payload: Any) -> list[EvalCase]:
    if isinstance(payload, dict):
        items = payload.get("cases", [])
    else:
        items = payload
    return [EvalCase.model_validate(item) for item in items]


def summarize_offline_results(
    eval_run_id: str,
    model_backend: ModelBackend,
    model_name: str,
    prompt_version: str,
    case_results: list[OfflineEvalCaseResult],
    skipped_cases: list[SkippedEvalCase] | None = None,
    retrieval_config_version: str | None = None,
    source_snapshot_id: str | None = None,
    gate_policy_version: str | None = None,
    by_bucket_breakdown: list[OfflineEvalBucketSummary] | None = None,
    release_decision: ReleaseDecision = ReleaseDecision.PASS,
    decision_reasons: list[str] | None = None,
    failed_buckets: list[str] | None = None,
    new_failures: list[str] | None = None,
    new_blocking_failures: list[str] | None = None,
    fixed_failures: list[str] | None = None,
) -> OfflineEvalSummary:
    skipped_cases = skipped_cases or []
    by_bucket_breakdown = by_bucket_breakdown or []
    decision_reasons = decision_reasons or []
    failed_buckets = failed_buckets or []
    new_failures = new_failures or []
    new_blocking_failures = new_blocking_failures or []
    fixed_failures = fixed_failures or []

    failure_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    by_scenario: dict[str, list[OfflineEvalCaseResult]] = defaultdict(list)
    for result in case_results:
        by_scenario[result.scenario.value].append(result)
        behavior_counts[result.actual_behavior.value] += 1
        for reason in result.actual_failure_reasons:
            failure_counts[reason.value] += 1

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    aggregate_metrics = {
        "schema_valid": avg([item.score_breakdown.schema_valid for item in case_results]),
        "citation_valid": avg([item.score_breakdown.citation_valid for item in case_results]),
        "retrieval_hit": avg([item.score_breakdown.retrieval_hit for item in case_results]),
        "refusal_correct": avg([item.score_breakdown.refusal_correct for item in case_results]),
        "answer_fact_match": avg([item.score_breakdown.answer_fact_match for item in case_results]),
        "unsupported_claim_penalty": avg([item.score_breakdown.unsupported_claim_penalty for item in case_results]),
        "policy_adherence_match": avg([item.score_breakdown.policy_adherence_match for item in case_results]),
        "brand_voice_match": avg([item.score_breakdown.brand_voice_match for item in case_results]),
        "tone_match": avg([item.score_breakdown.tone_match for item in case_results]),
        "behavior_match_rate": avg([1.0 if item.behavior_match else 0.0 for item in case_results]),
        "pass_rate": avg([1.0 if item.passed else 0.0 for item in case_results]),
        "weighted_overall": avg([item.weighted_score for item in case_results]),
        "overall": avg([item.score_breakdown.total for item in case_results]),
    }
    aggregate_metrics["valid_grounded_score"] = avg(
        [
            aggregate_metrics["schema_valid"],
            aggregate_metrics["citation_valid"],
            aggregate_metrics["retrieval_hit"],
            aggregate_metrics["answer_fact_match"],
            aggregate_metrics["unsupported_claim_penalty"],
            aggregate_metrics["policy_adherence_match"],
        ]
    )
    aggregate_metrics["behavior_score"] = aggregate_metrics["behavior_match_rate"]
    aggregate_metrics["advisory_quality_score"] = avg(
        [aggregate_metrics["brand_voice_match"], aggregate_metrics["tone_match"]]
    )

    by_scenario_breakdown = {
        scenario: {
            "overall": avg([item.score_breakdown.total for item in items]),
            "weighted_overall": avg([item.weighted_score for item in items]),
            "refusal_correct": avg([item.score_breakdown.refusal_correct for item in items]),
            "citation_valid": avg([item.score_breakdown.citation_valid for item in items]),
            "unsupported_claim_penalty": avg([item.score_breakdown.unsupported_claim_penalty for item in items]),
            "brand_voice_match": avg([item.score_breakdown.brand_voice_match for item in items]),
            "tone_match": avg([item.score_breakdown.tone_match for item in items]),
            "behavior_match_rate": avg([1.0 if item.behavior_match else 0.0 for item in items]),
            "valid_grounded_score": avg(
                [
                    avg([item.score_breakdown.schema_valid for item in items]),
                    avg([item.score_breakdown.citation_valid for item in items]),
                    avg([item.score_breakdown.retrieval_hit for item in items]),
                    avg([item.score_breakdown.answer_fact_match for item in items]),
                    avg([item.score_breakdown.unsupported_claim_penalty for item in items]),
                    avg([item.score_breakdown.policy_adherence_match for item in items]),
                ]
            ),
            "behavior_score": avg([1.0 if item.behavior_match else 0.0 for item in items]),
            "advisory_quality_score": avg(
                [
                    avg([item.score_breakdown.brand_voice_match for item in items]),
                    avg([item.score_breakdown.tone_match for item in items]),
                ]
            ),
        }
        for scenario, items in by_scenario.items()
    }

    worst_case_ids = [
        item.case_id
        for item in sorted(case_results, key=lambda result: (result.weighted_score, result.score_breakdown.total))[:3]
    ]

    return OfflineEvalSummary(
        eval_run_id=eval_run_id,
        created_at=datetime.now(UTC),
        model_backend=model_backend,
        model_name=model_name,
        prompt_version=prompt_version,
        retrieval_config_version=retrieval_config_version,
        source_snapshot_id=source_snapshot_id,
        gate_policy_version=gate_policy_version,
        aggregate_metrics=aggregate_metrics,
        by_scenario_breakdown=by_scenario_breakdown,
        by_bucket_breakdown=by_bucket_breakdown,
        failure_taxonomy_counts=dict(failure_counts),
        behavior_taxonomy_counts=dict(behavior_counts),
        worst_case_ids=worst_case_ids,
        release_decision=release_decision,
        decision_reasons=decision_reasons,
        failed_buckets=failed_buckets,
        new_failures=new_failures,
        new_blocking_failures=new_blocking_failures,
        fixed_failures=fixed_failures,
        skipped_cases=skipped_cases,
    )
