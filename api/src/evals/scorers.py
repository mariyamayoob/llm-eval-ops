from __future__ import annotations

from evals.contracts import EvalCase, OfflineEvalCaseResult, OfflineScoreBreakdown
from model.contracts import FailureReason, Outcome


def score_offline_case(*, eval_run_id: str, raw_case: dict, response, run_record) -> OfflineEvalCaseResult:
    case = EvalCase.model_validate(raw_case)
    schema_valid = 1.0 if run_record.validation_result.structure_valid else 0.0
    citation_valid = 1.0 if run_record.validation_result.citation_valid else 0.0
    retrieval_hit = 1.0 if set(case.required_citation_ids).issubset(set(run_record.retrieved_ids)) else 0.0
    model_refused_incorrectly = FailureReason.WRONG_REFUSAL in run_record.failure_reasons
    if case.expected_should_refuse:
        refusal_correct = 1.0 if response.outcome == Outcome.REFUSED_MORE_EVIDENCE_NEEDED else 0.0
    else:
        refusal_correct = 0.0 if model_refused_incorrectly else 1.0
    answer_text = response.answer or response.provisional_answer or ""
    answer_fact_match = 1.0 if case.expected_should_refuse or any(fact.lower() in answer_text.lower() for fact in case.acceptable_answer_facts) else 0.0
    unsupported_claim_penalty = 0.0 if any(claim.lower() in answer_text.lower() for claim in case.forbidden_claims) or FailureReason.UNSUPPORTED_ANSWER in run_record.failure_reasons else 1.0
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
    return OfflineEvalCaseResult(
        eval_run_id=eval_run_id,
        case_id=case.case_id,
        scenario=case.scenario,
        prompt_version=run_record.prompt_version,
        model_backend=run_record.model_backend,
        model_name=run_record.model_name,
        score_breakdown=score_breakdown,
        passed=score_breakdown.total >= 0.8,
        expected_should_refuse=case.expected_should_refuse,
        actual_outcome=run_record.outcome.value,
        required_citation_ids=case.required_citation_ids,
        actual_citations=response.citations,
        actual_failure_reasons=run_record.failure_reasons,
        actual_failure_categories=run_record.failure_categories,
        suspicious_flags=run_record.suspicious_flags,
        answer_preview=answer_text[:140],
        run_id=run_record.run_id,
        trace_id=run_record.trace_id,
    )
