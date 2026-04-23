from __future__ import annotations

from dataclasses import dataclass

from model.contracts import (
    FailureReason,
    RefusalReason,
    ReviewDecision,
    ReviewPriority,
    RiskBand,
    ScoreBreakdown,
    StructuredPolicyOutput,
    SuspiciousFlag,
    ValidationResult,
)


CORE_TOTAL_WEIGHTS = {
    "groundedness": 0.30,
    "citation_validity": 0.20,
    "policy_adherence": 0.20,
    "format_validity": 0.15,
    "retrieval_support": 0.15,
}

REVIEW_BLOCKING_FLAGS = {
    SuspiciousFlag.ANSWER_WITHOUT_CITATIONS,
    SuspiciousFlag.CITATIONS_NOT_IN_RETRIEVAL,
    SuspiciousFlag.POSSIBLE_POLICY_MISMATCH,
    SuspiciousFlag.REPAIR_ATTEMPTED,
    SuspiciousFlag.UNSUPPORTED_CLAIM_SIGNAL,
}


@dataclass(frozen=True)
class RuntimeCheckResult:
    score_breakdown: ScoreBreakdown
    suspicious_flags: list[SuspiciousFlag]
    risk_band: RiskBand
    review_decision: ReviewDecision
    review_reason_codes: list[str]


class OnlineScoringService:
    """Backwards-compatible name for runtime deterministic checks.

    The repo's "online scoring" layer is intentionally deterministic and cheap.
    It produces (a) a risk band and (b) a routing recommendation.
    """

    def score(
        self,
        *,
        structured_output: StructuredPolicyOutput,
        validation: ValidationResult,
        retrieval_stats,
    ) -> tuple[ScoreBreakdown, list[SuspiciousFlag], RiskBand, ReviewDecision]:
        result = self.evaluate(
            structured_output=structured_output,
            validation=validation,
            retrieval_stats=retrieval_stats,
        )
        return (
            result.score_breakdown,
            result.suspicious_flags,
            result.risk_band,
            result.review_decision,
        )

    def evaluate(
        self,
        *,
        structured_output: StructuredPolicyOutput,
        validation: ValidationResult,
        retrieval_stats,
    ) -> RuntimeCheckResult:
        # Cheap, deterministic runtime triage:
        # - assign a risk band (low/medium/high)
        # - immediately escalate only on hard blockers
        groundedness = 1.0 if FailureReason.UNSUPPORTED_ANSWER not in validation.failure_reasons and not retrieval_stats.retrieval_empty else 0.35
        citation_validity = 1.0 if validation.citation_valid else 0.0
        policy_adherence = 1.0
        if FailureReason.CONFLICTING_EVIDENCE in validation.failure_reasons or FailureReason.UNSUPPORTED_ANSWER in validation.failure_reasons:
            policy_adherence = 0.2
        brand_voice = 0.95 if self._brand_voice_ok(structured_output) else 0.45
        tone = 0.95 if self._tone_ok(structured_output) else 0.45
        format_validity = 1.0 if validation.structure_valid else 0.0
        retrieval_support = self._retrieval_support(retrieval_stats)
        total = round(
            CORE_TOTAL_WEIGHTS["groundedness"] * groundedness
            + CORE_TOTAL_WEIGHTS["citation_validity"] * citation_validity
            + CORE_TOTAL_WEIGHTS["policy_adherence"] * policy_adherence
            + CORE_TOTAL_WEIGHTS["format_validity"] * format_validity
            + CORE_TOTAL_WEIGHTS["retrieval_support"] * retrieval_support,
            4,
        )
        scores = ScoreBreakdown(
            groundedness_proxy_score=groundedness,
            citation_validity_score=citation_validity,
            policy_adherence_score=policy_adherence,
            brand_voice_score=brand_voice,
            tone_appropriateness_score=tone,
            format_validity_score=format_validity,
            retrieval_support_score=retrieval_support,
            total=total,
        )

        flags: list[SuspiciousFlag] = []
        if not structured_output.refusal and not structured_output.citations:
            flags.append(SuspiciousFlag.ANSWER_WITHOUT_CITATIONS)
        if structured_output.citations and not validation.citation_valid:
            flags.append(SuspiciousFlag.CITATIONS_NOT_IN_RETRIEVAL)
        if not structured_output.refusal and structured_output.confidence >= 0.9 and retrieval_support < 0.4:
            flags.append(SuspiciousFlag.HIGH_CONFIDENCE_LOW_SUPPORT)
        if FailureReason.UNSUPPORTED_ANSWER in validation.failure_reasons or FailureReason.CONFLICTING_EVIDENCE in validation.failure_reasons:
            flags.append(SuspiciousFlag.POSSIBLE_POLICY_MISMATCH)
        if not self._tone_ok(structured_output):
            flags.append(SuspiciousFlag.TONE_MISMATCH)
        if validation.repair_attempted:
            flags.append(SuspiciousFlag.REPAIR_ATTEMPTED)
        if FailureReason.UNSUPPORTED_ANSWER in validation.failure_reasons:
            flags.append(SuspiciousFlag.UNSUPPORTED_CLAIM_SIGNAL)
        if (
            structured_output.refusal
            and retrieval_support >= 0.75
            and structured_output.refusal_reason != RefusalReason.CONFLICTING_EVIDENCE
        ):
            flags.append(SuspiciousFlag.REFUSAL_DESPITE_STRONG_RETRIEVAL)
        flags = list(dict.fromkeys(flags))

        blocking_flags = [flag for flag in flags if flag in REVIEW_BLOCKING_FLAGS]
        advisory_flags = [flag for flag in flags if flag not in REVIEW_BLOCKING_FLAGS]

        if blocking_flags:
            risk_band = RiskBand.HIGH
            # Hard blockers are still flagged deterministically, but human review is now
            # only created downstream (LLM judge / side-channel signals).
            review = ReviewDecision(review_required=False)
        elif advisory_flags or total < 0.60:
            risk_band = RiskBand.HIGH if total < 0.60 else RiskBand.MEDIUM
            review = ReviewDecision(review_required=False)
        elif total < 0.85:
            risk_band = RiskBand.MEDIUM
            review = ReviewDecision(review_required=False)
        else:
            risk_band = RiskBand.LOW
            review = ReviewDecision(review_required=False)

        return RuntimeCheckResult(
            score_breakdown=scores,
            suspicious_flags=flags,
            risk_band=risk_band,
            review_decision=review,
            review_reason_codes=[flag.value for flag in blocking_flags],
        )

    def _brand_voice_ok(self, structured_output: StructuredPolicyOutput) -> bool:
        content = structured_output.answer or structured_output.missing_or_conflicting_evidence_summary or ""
        return "!" not in content and "amazing" not in content.lower()

    def _tone_ok(self, structured_output: StructuredPolicyOutput) -> bool:
        content = structured_output.answer or structured_output.missing_or_conflicting_evidence_summary or ""
        return len(content) <= 220 and "please note" not in content.lower()

    def _retrieval_support(self, retrieval_stats) -> float:
        if retrieval_stats.retrieval_empty:
            return 0.0
        if retrieval_stats.similarity_max >= 0.25:
            return 0.95
        if retrieval_stats.similarity_max >= 0.15:
            return 0.8
        return 0.45


# Clearer architecture alias (used in docs/articles).
RuntimeDeterministicChecks = OnlineScoringService
