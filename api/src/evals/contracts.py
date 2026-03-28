from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from model.contracts import (
    Difficulty,
    FailureCategory,
    FailureReason,
    FinalDisposition,
    ModelBackend,
    ReviewPriority,
    ReviewStatus,
    ScenarioName,
    SuspiciousFlag,
)


class EvalTag(str, Enum):
    DIRECT_ANSWERABLE = "direct_answerable"
    SHOULD_REFUSE_MISSING_EVIDENCE = "should_refuse_missing_evidence"
    SHOULD_REFUSE_CONFLICTING_EVIDENCE = "should_refuse_conflicting_evidence"
    RETRIEVAL_MISS = "retrieval_miss"
    MALFORMED_OUTPUT = "malformed_output"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    TONE_AND_BRAND_VOICE = "tone_and_brand_voice"
    POLICY_BOUNDARY_CASES = "policy_boundary_cases"


class EvalCase(BaseModel):
    case_id: str
    question: str
    scenario: ScenarioName
    expected_should_refuse: bool
    required_citation_ids: list[str] = Field(default_factory=list)
    acceptable_answer_facts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    tags: list[EvalTag] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM


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
    score_breakdown: OfflineScoreBreakdown
    passed: bool
    expected_should_refuse: bool
    actual_outcome: str
    required_citation_ids: list[str]
    actual_citations: list[str]
    actual_failure_reasons: list[FailureReason] = Field(default_factory=list)
    actual_failure_categories: list[FailureCategory] = Field(default_factory=list)
    suspicious_flags: list[SuspiciousFlag] = Field(default_factory=list)
    answer_preview: str
    run_id: str
    trace_id: str


class SkippedEvalCase(BaseModel):
    case_id: str
    scenario: ScenarioName
    reason: str


class OfflineEvalSummary(BaseModel):
    eval_run_id: str
    created_at: datetime
    model_backend: ModelBackend
    model_name: str
    prompt_version: str
    aggregate_metrics: dict[str, float]
    by_scenario_breakdown: dict[str, dict[str, float]]
    failure_taxonomy_counts: dict[str, int]
    worst_case_ids: list[str]
    skipped_cases: list[SkippedEvalCase] = Field(default_factory=list)


class OfflineEvalRun(BaseModel):
    summary: OfflineEvalSummary
    case_results: list[OfflineEvalCaseResult]


class ReviewQueueItem(BaseModel):
    review_queue_item_id: str
    run_id: str
    trace_id: str
    online_score_total: float
    review_priority: ReviewPriority
    suspicious_flags: list[SuspiciousFlag] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer_label: str | None = None
    reviewer_notes: str | None = None
    final_disposition: FinalDisposition | None = None


class ReviewerAnnotation(BaseModel):
    reviewer_label: str
    reviewer_notes: str | None = None
    review_status: ReviewStatus
    final_disposition: FinalDisposition


def summarize_offline_results(
    eval_run_id: str,
    model_backend: ModelBackend,
    model_name: str,
    prompt_version: str,
    case_results: list[OfflineEvalCaseResult],
    skipped_cases: list[SkippedEvalCase] | None = None,
) -> OfflineEvalSummary:
    skipped_cases = skipped_cases or []
    failure_counts: Counter[str] = Counter()
    by_scenario: dict[str, list[OfflineEvalCaseResult]] = defaultdict(list)
    for result in case_results:
        by_scenario[result.scenario.value].append(result)
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
        "policy_adherence_match": avg([item.score_breakdown.policy_adherence_match for item in case_results]),
        "overall": avg([item.score_breakdown.total for item in case_results]),
    }

    by_scenario_breakdown = {
        scenario: {
            "overall": avg([item.score_breakdown.total for item in items]),
            "refusal_correct": avg([item.score_breakdown.refusal_correct for item in items]),
            "citation_valid": avg([item.score_breakdown.citation_valid for item in items]),
        }
        for scenario, items in by_scenario.items()
    }

    worst_case_ids = [
        item.case_id
        for item in sorted(case_results, key=lambda result: result.score_breakdown.total)[:3]
    ]

    return OfflineEvalSummary(
        eval_run_id=eval_run_id,
        created_at=datetime.now(UTC),
        model_backend=model_backend,
        model_name=model_name,
        prompt_version=prompt_version,
        aggregate_metrics=aggregate_metrics,
        by_scenario_breakdown=by_scenario_breakdown,
        failure_taxonomy_counts=dict(failure_counts),
        worst_case_ids=worst_case_ids,
        skipped_cases=skipped_cases,
    )
