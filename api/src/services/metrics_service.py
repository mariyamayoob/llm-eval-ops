from __future__ import annotations

from model.contracts import (
    FailureReason,
    ReviewDecision,
    ReviewPriority,
    RiskBand,
    ScoreBreakdown,
    StructuredPolicyOutput,
    SuspiciousFlag,
    ValidationResult,
)


class OnlineScoringService:
    def score(
        self,
        *,
        structured_output: StructuredPolicyOutput,
        validation: ValidationResult,
        retrieval_stats,
        prompt_version: str,
    ) -> tuple[ScoreBreakdown, list[SuspiciousFlag], RiskBand, ReviewDecision]:
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
            0.25 * groundedness
            + 0.15 * citation_validity
            + 0.20 * policy_adherence
            + 0.10 * brand_voice
            + 0.10 * tone
            + 0.10 * format_validity
            + 0.10 * retrieval_support,
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
        if not validation.citation_valid:
            flags.append(SuspiciousFlag.CITATIONS_NOT_IN_RETRIEVAL)
        if structured_output.confidence >= 0.9 and retrieval_support < 0.4:
            flags.append(SuspiciousFlag.HIGH_CONFIDENCE_LOW_SUPPORT)
        if FailureReason.UNSUPPORTED_ANSWER in validation.failure_reasons or FailureReason.CONFLICTING_EVIDENCE in validation.failure_reasons:
            flags.append(SuspiciousFlag.POSSIBLE_POLICY_MISMATCH)
        if not self._tone_ok(structured_output):
            flags.append(SuspiciousFlag.TONE_MISMATCH)
        if validation.repair_attempted:
            flags.append(SuspiciousFlag.REPAIR_ATTEMPTED)
        if FailureReason.UNSUPPORTED_ANSWER in validation.failure_reasons:
            flags.append(SuspiciousFlag.UNSUPPORTED_CLAIM_SIGNAL)
        if structured_output.refusal and retrieval_support >= 0.75:
            flags.append(SuspiciousFlag.REFUSAL_DESPITE_STRONG_RETRIEVAL)
        flags = list(dict.fromkeys(flags))

        critical_flags = set(flags)
        if critical_flags:
            risk_band = RiskBand.HIGH
            review = ReviewDecision(review_required=True, review_priority=ReviewPriority.HIGH, human_review_reason="Critical review flags were raised.")
        elif total < 0.70:
            risk_band = RiskBand.HIGH
            review = ReviewDecision(review_required=True, review_priority=ReviewPriority.HIGH, human_review_reason="Online score is below the production review threshold.")
        elif total < 0.85:
            risk_band = RiskBand.MEDIUM
            review = ReviewDecision(review_required=False)
        else:
            risk_band = RiskBand.LOW
            review = ReviewDecision(review_required=False)
        return scores, flags, risk_band, review

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
