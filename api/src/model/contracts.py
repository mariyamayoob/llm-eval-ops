from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator
from prompts.registry import DEFAULT_PROMPT_VERSION, PROMPT_VERSION_PATTERN


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


class FeedbackEventType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    REGENERATE = "regenerate"
    CLARIFICATION_FOLLOWUP = "clarification_followup"
    ABANDON = "abandon"
    REVIEWER_CONFIRMED_ISSUE = "reviewer_confirmed_issue"


class UserFeedbackEventType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


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


class AlertStatus(str, Enum):
    OK = "ok"
    WATCH = "watch"
    ACTION_REQUIRED = "action_required"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class PolicyDeskAssistantRequest(BaseModel):
    question: str = Field(..., min_length=1)
    scenario: ScenarioName = ScenarioName.NORMAL
    model_backend: ModelBackend = ModelBackend.MOCK
    prompt_version: str = Field(default=DEFAULT_PROMPT_VERSION, pattern=PROMPT_VERSION_PATTERN)
    retrieval_config_version: str = "retrieval-config:v1"
    source_snapshot_id: str = "kb-snapshot:current"


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
    retrieval_config_version: str = "retrieval-config:v1"
    source_snapshot_id: str = "kb-snapshot:current"
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
    question: str | None = None
    model_backend: ModelBackend
    model_name: str
    prompt_version: str
    retrieval_config_version: str = "retrieval-config:v1"
    source_snapshot_id: str = "kb-snapshot:current"
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
    retrieval_config_version: str = "retrieval-config:v1"
    source_snapshot_id: str = "kb-snapshot:current"
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


class RuntimeFeedbackRequest(BaseModel):
    run_id: str = Field(..., min_length=1)
    event_type: UserFeedbackEventType
    session_id: str | None = None
    event_value: str | None = None


class RuntimeFeedbackEvent(BaseModel):
    event_id: str
    run_id: str
    session_id: str | None = None
    event_type: FeedbackEventType
    event_value: str | None = None
    created_at: datetime
    model_backend: ModelBackend
    prompt_version: str
    response_outcome: Outcome
    risk_band: RiskBand


class LLMJudgeStatus(str, Enum):
    QUEUED = "queued"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LLMJudgeAssessment(BaseModel):
    supportedness_score: float = Field(ge=0.0, le=1.0)
    policy_alignment_score: float = Field(ge=0.0, le=1.0)
    response_mode_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    human_review_recommended: bool
    human_review_reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class LLMJudgeRecord(BaseModel):
    judge_id: str
    run_id: str
    created_at: datetime
    completed_at: datetime | None = None
    status: LLMJudgeStatus
    judge_model: str
    sampling_rate: float = Field(ge=0.0, le=1.0)
    model_backend: ModelBackend
    prompt_version: str
    response_outcome: Outcome
    risk_band: RiskBand
    openai_response_id: str | None = None
    review_queue_item_id: str | None = None
    assessment: LLMJudgeAssessment | None = None
    error_message: str | None = None


class OnlineMetricGroup(BaseModel):
    group_key: str
    total_runs: int
    total_feedback_events: int
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    risk_band_counts: dict[str, int] = Field(default_factory=dict)
    avg_groundedness_proxy_score: float = Field(ge=0.0, le=1.0)
    avg_citation_validity_score: float = Field(ge=0.0, le=1.0)
    avg_policy_adherence_score: float = Field(ge=0.0, le=1.0)
    avg_format_validity_score: float = Field(ge=0.0, le=1.0)
    avg_retrieval_support_score: float = Field(ge=0.0, le=1.0)
    avg_brand_voice_score: float = Field(ge=0.0, le=1.0)
    avg_tone_appropriateness_score: float = Field(ge=0.0, le=1.0)
    avg_online_score_total: float = Field(ge=0.0, le=1.0)
    review_recommended_rate: float = Field(ge=0.0, le=1.0)
    supported_answer_rate: float = Field(ge=0.0, le=1.0)
    invalid_output_rate: float = Field(ge=0.0, le=1.0)
    thumbs_up_rate: float = Field(ge=0.0, le=1.0)
    thumbs_down_rate: float = Field(ge=0.0, le=1.0)
    high_risk_run_count: int


class OnlineMetricsSummary(BaseModel):
    window_start: datetime
    window_end: datetime
    total_runs: int
    total_feedback_events: int
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    risk_band_counts: dict[str, int] = Field(default_factory=dict)
    avg_groundedness_proxy_score: float = Field(ge=0.0, le=1.0)
    avg_citation_validity_score: float = Field(ge=0.0, le=1.0)
    avg_policy_adherence_score: float = Field(ge=0.0, le=1.0)
    avg_format_validity_score: float = Field(ge=0.0, le=1.0)
    avg_retrieval_support_score: float = Field(ge=0.0, le=1.0)
    avg_brand_voice_score: float = Field(ge=0.0, le=1.0)
    avg_tone_appropriateness_score: float = Field(ge=0.0, le=1.0)
    avg_online_score_total: float = Field(ge=0.0, le=1.0)
    review_recommended_rate: float = Field(ge=0.0, le=1.0)
    supported_answer_rate: float = Field(ge=0.0, le=1.0)
    invalid_output_rate: float = Field(ge=0.0, le=1.0)
    thumbs_up_rate: float = Field(ge=0.0, le=1.0)
    thumbs_down_rate: float = Field(ge=0.0, le=1.0)
    high_risk_run_count: int
    by_prompt_version: list[OnlineMetricGroup] = Field(default_factory=list)
    by_model_backend: list[OnlineMetricGroup] = Field(default_factory=list)
    by_response_outcome: list[OnlineMetricGroup] = Field(default_factory=list)
    by_risk_band: list[OnlineMetricGroup] = Field(default_factory=list)


class TriggeredAlert(BaseModel):
    metric_name: str
    severity: AlertStatus
    observed_value: float
    threshold_value: float
    comparator: str
    message: str


class OnlineAlertPolicy(BaseModel):
    policy_version: str = "online-alert-policy:v1"
    max_invalid_output_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    max_review_recommended_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    max_thumbs_down_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    min_supported_answer_rate: float = Field(default=0.45, ge=0.0, le=1.0)
    max_high_risk_run_rate: float = Field(default=0.25, ge=0.0, le=1.0)


class OnlineAlertEvaluation(BaseModel):
    status: AlertStatus
    triggered_alerts: list[TriggeredAlert] = Field(default_factory=list)
    rollback_recommended: bool = False
    rollback_reasons: list[str] = Field(default_factory=list)


class OnlineSummarySnapshot(BaseModel):
    summary: OnlineMetricsSummary
    alert_evaluation: OnlineAlertEvaluation
    worst_run_ids: list[str] = Field(default_factory=list)
    review_queue_item_ids: list[str] = Field(default_factory=list)


class ReviewQueueSummary(BaseModel):
    review_queue_item_id: str
    run_id: str
    trace_id: str
    online_score_total: float
    review_priority: ReviewPriority
    suspicious_flags: list[SuspiciousFlag]
    review_status: ReviewStatus
    final_disposition: FinalDisposition | None = None
