from __future__ import annotations

from evals.contracts import EvalBehavior, EvalCase, OfflineEvalCaseResult, OfflineScoreBreakdown
from model.contracts import FailureReason, Outcome


CLARIFY_PREFIXES = (
    "can you clarify",
    "could you clarify",
    "please clarify",
    "which ",
    "what type of ",
    "what kind of ",
    "do you mean",
)


def score_offline_case(
    *,
    eval_run_id: str,
    raw_case: dict | EvalCase,
    response,
    run_record,
    retrieval_config_version: str | None = None,
    source_snapshot_id: str | None = None,
) -> OfflineEvalCaseResult:
    case = raw_case if isinstance(raw_case, EvalCase) else EvalCase.model_validate(raw_case)
    actual_behavior = classify_actual_behavior(response)
    behavior_match = _canonical_behavior(actual_behavior) == _canonical_behavior(case.expected_behavior)

    schema_valid = 1.0 if run_record.validation_result.structure_valid else 0.0
    citation_valid = 1.0 if run_record.validation_result.citation_valid else 0.0
    retrieval_hit = 1.0 if set(case.required_citation_ids).issubset(set(run_record.retrieved_ids)) else 0.0
    refusal_correct = 1.0 if behavior_match else 0.0

    answer_text = response.answer or response.provisional_answer or ""
    preview_text = answer_text or response.missing_or_conflicting_evidence_summary or ""
    answer_fact_match = 1.0
    if case.expected_behavior == EvalBehavior.ANSWER:
        answer_fact_match = 1.0 if _contains_any(answer_text, case.acceptable_answer_facts) else 0.0

    unsupported_claim_penalty = 0.0 if _contains_any(answer_text, case.forbidden_claims) else 1.0
    if FailureReason.UNSUPPORTED_ANSWER in run_record.failure_reasons:
        unsupported_claim_penalty = 0.0

    brand_voice_match = run_record.score_breakdown.brand_voice_score
    tone_match = run_record.score_breakdown.tone_appropriateness_score
    policy_adherence_match = run_record.score_breakdown.policy_adherence_score
    score_breakdown = OfflineScoreBreakdown(
        schema_valid=schema_valid,
        citation_valid=citation_valid,
        retrieval_hit=retrieval_hit,
        refusal_correct=refusal_correct,
        answer_fact_match=answer_fact_match,
        unsupported_claim_penalty=unsupported_claim_penalty,
        brand_voice_match=brand_voice_match,
        tone_match=tone_match,
        policy_adherence_match=policy_adherence_match,
    )

    regression_blockers = detect_regression_blockers(
        case=case,
        actual_behavior=actual_behavior,
        behavior_match=behavior_match,
        answer_text=answer_text,
        score_breakdown=score_breakdown,
        failure_reasons=run_record.failure_reasons,
    )

    return OfflineEvalCaseResult(
        eval_run_id=eval_run_id,
        case_id=case.case_id,
        scenario=case.scenario,
        prompt_version=run_record.prompt_version,
        model_backend=run_record.model_backend,
        model_name=run_record.model_name,
        retrieval_config_version=retrieval_config_version,
        source_snapshot_id=source_snapshot_id,
        bucket_id=case.bucket_id,
        bucket_name=case.bucket_name,
        risk_tier=case.risk_tier,
        business_criticality=case.business_criticality,
        gate_group=case.gate_group,
        score_breakdown=score_breakdown,
        passed=False,
        expected_should_refuse=case.expected_should_refuse,
        expected_behavior=case.expected_behavior,
        actual_behavior=actual_behavior,
        behavior_match=behavior_match,
        actual_outcome=run_record.outcome.value,
        required_citation_ids=case.required_citation_ids,
        actual_citations=response.citations,
        actual_failure_reasons=run_record.failure_reasons,
        actual_failure_categories=run_record.failure_categories,
        suspicious_flags=run_record.suspicious_flags,
        regression_blockers=regression_blockers,
        answer_preview=preview_text[:140],
        run_id=run_record.run_id,
        trace_id=run_record.trace_id,
    )


def classify_actual_behavior(response) -> EvalBehavior:
    text = _normalize_text(response.answer or response.provisional_answer or response.missing_or_conflicting_evidence_summary)
    if _looks_like_clarification(text):
        return EvalBehavior.CLARIFY
    if response.outcome == Outcome.HUMAN_REVIEW_RECOMMENDED:
        return EvalBehavior.HUMAN_REVIEW
    if response.outcome == Outcome.REFUSED_MORE_EVIDENCE_NEEDED:
        return EvalBehavior.ABSTAIN
    return EvalBehavior.ANSWER


def detect_regression_blockers(
    *,
    case: EvalCase,
    actual_behavior: EvalBehavior,
    behavior_match: bool,
    answer_text: str,
    score_breakdown: OfflineScoreBreakdown,
    failure_reasons: list[FailureReason],
) -> list[str]:
    blockers: list[str] = []

    if not behavior_match:
        blockers.append("behavior_mismatch")
        expected_behavior = _canonical_behavior(case.expected_behavior)
        actual_gate_behavior = _canonical_behavior(actual_behavior)
        if expected_behavior == EvalBehavior.ANSWER and actual_gate_behavior == EvalBehavior.ABSTAIN:
            blockers.append("over_refusal")
        elif expected_behavior in {EvalBehavior.ABSTAIN, EvalBehavior.HUMAN_REVIEW} and actual_gate_behavior == EvalBehavior.ANSWER:
            blockers.append("unsafe_compliance")
        elif actual_behavior == EvalBehavior.CLARIFY and case.expected_behavior != EvalBehavior.CLARIFY:
            blockers.append("false_clarify")
        elif actual_behavior == EvalBehavior.HUMAN_REVIEW and case.expected_behavior != EvalBehavior.HUMAN_REVIEW:
            blockers.append("unnecessary_escalation")
        elif case.expected_behavior == EvalBehavior.HUMAN_REVIEW and actual_behavior != EvalBehavior.HUMAN_REVIEW:
            blockers.append("missed_human_review")
        elif case.expected_behavior == EvalBehavior.CLARIFY and actual_behavior != EvalBehavior.CLARIFY:
            blockers.append("missed_clarify")

    if score_breakdown.answer_fact_match == 0.0 and case.expected_behavior == EvalBehavior.ANSWER:
        blockers.append("answer_fact_mismatch")
    if score_breakdown.unsupported_claim_penalty == 0.0:
        blockers.append("unsupported_claim")
    if score_breakdown.citation_valid == 0.0:
        blockers.append("invalid_citation")
    if score_breakdown.schema_valid == 0.0:
        blockers.append("schema_invalid")
    if score_breakdown.retrieval_hit == 0.0 and case.required_citation_ids:
        blockers.append("retrieval_miss")
    if FailureReason.UNSUPPORTED_ANSWER in failure_reasons:
        blockers.append("unsupported_claim")
    if FailureReason.FABRICATED_CITATION in failure_reasons:
        blockers.append("invalid_citation")
    if FailureReason.MALFORMED_JSON in failure_reasons or FailureReason.SCHEMA_INVALID in failure_reasons:
        blockers.append("schema_invalid")
    if answer_text and _contains_any(answer_text, case.forbidden_claims):
        blockers.append("unsupported_claim")

    return sorted(dict.fromkeys(blockers))


def _contains_any(text: str, terms: list[str]) -> bool:
    haystack = _normalize_text(text)
    return any(term.lower() in haystack for term in terms)


def _looks_like_clarification(text: str) -> bool:
    return bool(text) and (text.endswith("?") or any(text.startswith(prefix) for prefix in CLARIFY_PREFIXES))


def _normalize_text(text: str | None) -> str:
    return (text or "").strip().lower()


def _canonical_behavior(value: EvalBehavior) -> EvalBehavior:
    return EvalBehavior.ABSTAIN if value == EvalBehavior.REFUSE else value
