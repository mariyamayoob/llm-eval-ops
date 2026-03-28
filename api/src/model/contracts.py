from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ModelBackend(str, Enum):
    MOCK = "mock"
    OPENAI = "openai"


class ScenarioName(str, Enum):
    NORMAL = "normal"
    RETRIEVAL_MISS = "retrieval_miss"
    MALFORMED_JSON = "malformed_json"
    UNSUPPORTED_ANSWER = "unsupported_answer"
    WRONG_REFUSAL = "wrong_refusal"
    SLOW_RESPONSE = "slow_response"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class Outcome(str, Enum):
    SUPPORTED_ANSWER = "supported_answer"
    REFUSED_MORE_EVIDENCE_NEEDED = "refused_more_evidence_needed"
    HUMAN_REVIEW_RECOMMENDED = "human_review_recommended"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RefusalReason(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    OUT_OF_SCOPE = "out_of_scope"


class ReviewPriority(str, Enum):
    MEDIUM = "medium"
    HIGH = "high"


class SuspiciousFlag(str, Enum):
    ANSWER_WITHOUT_CITATIONS = "answer_without_citations"
    CITATIONS_NOT_IN_RETRIEVAL = "citations_not_in_retrieval"
    HIGH_CONFIDENCE_LOW_SUPPORT = "high_confidence_low_support"
    POSSIBLE_POLICY_MISMATCH = "possible_policy_mismatch"
    TONE_MISMATCH = "tone_mismatch"
    REPAIR_ATTEMPTED = "repair_attempted"
    UNSUPPORTED_CLAIM_SIGNAL = "unsupported_claim_signal"
    REFUSAL_DESPITE_STRONG_RETRIEVAL = "refusal_despite_strong_retrieval"


class FailureCategory(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"
    VALIDATION_FAILURE = "validation_failure"
    POLICY_FAILURE = "policy_failure"
    SYSTEM_FAILURE = "system_failure"


class FailureReason(str, Enum):
    NO_RETRIEVAL_HIT = "no_retrieval_hit"
    WEAK_RETRIEVAL_HIT = "weak_retrieval_hit"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_INVALID = "schema_invalid"
    FABRICATED_CITATION = "fabricated_citation"
    UNSUPPORTED_ANSWER = "unsupported_answer"
    WRONG_REFUSAL = "wrong_refusal"
    MISSING_REFUSAL = "missing_refusal"
    TIMEOUT = "timeout"
    REPAIR_EXHAUSTED = "repair_exhausted"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"


class FinalDisposition(str, Enum):
    APPROVED = "approved"
    CORRECTED = "corrected"
    CONFIRMED_REFUSAL = "confirmed_refusal"
    ESCALATED = "escalated"
    REJECTED_RESPONSE = "rejected_response"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PolicyDeskAssistantRequest(BaseModel):
    question: str = Field(..., min_length=1)
    scenario: ScenarioName = ScenarioName.NORMAL
    model_backend: ModelBackend = ModelBackend.MOCK
    prompt_version: str = "qa-prompt:v1"


class RetrievedChunk(BaseModel):
    id: str
    title: str
    score: float
    tags: list[str]


class RetrievalStats(BaseModel):
    top_k: int
    candidate_count: int
    similarity_min: float
    similarity_max: float
    similarity_mean: float
    retrieval_empty: bool


class EvidenceSummaryItem(BaseModel):
    chunk_id: str
    title: str
    support_snippet: str
    relevance_score: float


class StructuredPolicyOutput(BaseModel):
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    evidence_summary: list[EvidenceSummaryItem] = Field(default_factory=list)
    refusal: bool = False
    refusal_reason: RefusalReason | None = None
    missing_or_conflicting_evidence_summary: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_invariants(self) -> "StructuredPolicyOutput":
        if self.refusal:
            if self.answer:
                raise ValueError("refusal output must not contain answer")
            if self.citations:
                raise ValueError("refusal output must not contain citations")
            if not self.refusal_reason:
                raise ValueError("refusal output must contain refusal_reason")
            if not self.missing_or_conflicting_evidence_summary:
                raise ValueError("refusal output must explain the evidence gap or conflict")
        else:
            if not self.answer:
                raise ValueError("supported output must contain answer")
            if not self.citations:
                raise ValueError("supported output must contain citations")
            if not self.evidence_summary:
                raise ValueError("supported output must contain evidence_summary")
        return self


class ValidationResult(BaseModel):
    structure_valid: bool
    citation_valid: bool
    repair_attempted: bool
    failure_reasons: list[FailureReason] = Field(default_factory=list)
    failure_categories: list[FailureCategory] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    groundedness_proxy_score: float = Field(ge=0.0, le=1.0)
    citation_validity_score: float = Field(ge=0.0, le=1.0)
    policy_adherence_score: float = Field(ge=0.0, le=1.0)
    brand_voice_score: float = Field(ge=0.0, le=1.0)
    tone_appropriateness_score: float = Field(ge=0.0, le=1.0)
    format_validity_score: float = Field(ge=0.0, le=1.0)
    retrieval_support_score: float = Field(ge=0.0, le=1.0)
    total: float = Field(ge=0.0, le=1.0)


class ReviewDecision(BaseModel):
    review_required: bool
    review_priority: ReviewPriority | None = None
    review_queue_item_id: str | None = None
    human_review_reason: str | None = None


class PolicyDeskAssistantResponse(BaseModel):
    run_id: str
    trace_id: str
    model_backend: ModelBackend
    model_name: str
    prompt_version: str
    outcome: Outcome
    online_score_total: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    suspicious_flags: list[SuspiciousFlag] = Field(default_factory=list)
    review_required: bool
    score_breakdown: ScoreBreakdown
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    evidence_summary: list[EvidenceSummaryItem] = Field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    missing_or_conflicting_evidence_summary: str | None = None
    provisional_answer: str | None = None
    review_priority: ReviewPriority | None = None
    review_queue_item_id: str | None = None
    human_review_reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_outcome_contract(self) -> "PolicyDeskAssistantResponse":
        if self.outcome == Outcome.SUPPORTED_ANSWER:
            if not self.answer or not self.citations or not self.evidence_summary:
                raise ValueError("supported_answer requires answer, citations, and evidence_summary")
        elif self.outcome == Outcome.REFUSED_MORE_EVIDENCE_NEEDED:
            if not self.refusal_reason or not self.missing_or_conflicting_evidence_summary:
                raise ValueError("refused_more_evidence_needed requires refusal_reason and evidence summary")
        elif self.outcome == Outcome.HUMAN_REVIEW_RECOMMENDED:
            if not self.review_priority or not self.review_queue_item_id or not self.human_review_reason:
                raise ValueError("human_review_recommended requires review routing fields")
        return self


class RunRecord(BaseModel):
    run_id: str
    trace_id: str
    created_at: datetime
    model_backend: ModelBackend
    model_name: str
    prompt_version: str
    question_hash: str
    question_len: int
    scenario: ScenarioName
    retrieved_ids: list[str]
    retrieval_stats: RetrievalStats
    outcome: Outcome
    validation_result: ValidationResult
    score_breakdown: ScoreBreakdown
    online_score_total: float
    risk_band: RiskBand
    suspicious_flags: list[SuspiciousFlag]
    review_required: bool
    review_queue_item_id: str | None = None
    failure_categories: list[FailureCategory] = Field(default_factory=list)
    failure_reasons: list[FailureReason] = Field(default_factory=list)
    step_timings_ms: dict[str, float] = Field(default_factory=dict)
    response_payload: PolicyDeskAssistantResponse


class RunSummary(BaseModel):
    run_id: str
    trace_id: str
    created_at: datetime
    model_backend: ModelBackend
    model_name: str
    prompt_version: str
    scenario: ScenarioName
    outcome: Outcome
    online_score_total: float
    risk_band: RiskBand
    review_required: bool
    review_queue_item_id: str | None = None
    suspicious_flags: list[SuspiciousFlag] = Field(default_factory=list)


class TraceSummary(BaseModel):
    trace_id: str
    run_id: str
    scenario: ScenarioName
    prompt_version: str
    total_latency_ms: float
    suspicious_flags: list[SuspiciousFlag] = Field(default_factory=list)


class ReviewQueueSummary(BaseModel):
    review_queue_item_id: str
    run_id: str
    trace_id: str
    online_score_total: float
    review_priority: ReviewPriority
    suspicious_flags: list[SuspiciousFlag]
    review_status: ReviewStatus
    final_disposition: FinalDisposition | None = None
