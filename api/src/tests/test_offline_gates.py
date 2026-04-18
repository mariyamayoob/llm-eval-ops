from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from evals.contracts import (
    BusinessCriticality,
    CaseSet,
    EvalBehavior,
    EvalCase,
    OfflineComparisonRequest,
    OfflineEvalConfig,
    OfflineEvalCaseResult,
    OfflineEvalRun,
    OfflineScoreBreakdown,
    ReleaseDecision,
    RiskTier,
    parse_eval_cases,
    summarize_offline_results,
)
from evals.gates import apply_gate_policy, load_gate_policy
from evals.runner import OfflineEvalRunner, load_eval_cases
from evals.scorers import score_offline_case
from model.contracts import ModelBackend, Outcome, RefusalReason, ScenarioName


ROOT = Path(__file__).resolve().parents[3]


def build_case_result(
    *,
    case: EvalCase,
    case_id_suffix: str = "",
    score_breakdown: OfflineScoreBreakdown | None = None,
    actual_behavior: EvalBehavior | None = None,
    behavior_match: bool = True,
    blockers: list[str] | None = None,
) -> OfflineEvalCaseResult:
    score_breakdown = score_breakdown or OfflineScoreBreakdown(
        schema_valid=1.0,
        citation_valid=1.0,
        retrieval_hit=1.0,
        refusal_correct=1.0 if behavior_match else 0.0,
        answer_fact_match=1.0,
        unsupported_claim_penalty=1.0,
        brand_voice_match=1.0,
        tone_match=1.0,
        policy_adherence_match=1.0,
    )
    actual_behavior = actual_behavior or case.expected_behavior
    return OfflineEvalCaseResult(
        eval_run_id="eval-run",
        case_id=f"{case.case_id}{case_id_suffix}",
        scenario=case.scenario,
        prompt_version="qa-prompt:v1",
        model_backend=ModelBackend.MOCK,
        model_name="mock-policy-model",
        retrieval_config_version="retrieval-config:v1",
        source_snapshot_id="kb-snapshot:test",
        bucket_id=case.bucket_id,
        bucket_name=case.bucket_name,
        risk_tier=case.risk_tier,
        business_criticality=case.business_criticality,
        gate_group=case.gate_group,
        score_breakdown=score_breakdown,
        weighted_score=0.0,
        passed=False,
        expected_should_refuse=case.expected_should_refuse,
        expected_behavior=case.expected_behavior,
        actual_behavior=actual_behavior,
        behavior_match=behavior_match,
        actual_outcome=Outcome.SUPPORTED_ANSWER.value,
        required_citation_ids=case.required_citation_ids,
        actual_citations=case.required_citation_ids,
        actual_failure_reasons=[],
        actual_failure_categories=[],
        suspicious_flags=[],
        regression_blockers=blockers or [],
        answer_preview="preview",
        run_id=f"run-{case.case_id}{case_id_suffix}",
        trace_id=f"trace-{case.case_id}{case_id_suffix}",
    )


def test_backward_compatible_dataset_loading_backfills_new_fields():
    legacy_payload = [
        {
            "case_id": "legacy-retrieval-miss",
            "question": "What is the policy for pet insurance reimbursement?",
            "scenario": "retrieval_miss",
            "expected_should_refuse": True,
            "required_citation_ids": [],
            "acceptable_answer_facts": [],
            "forbidden_claims": ["30 days"],
            "tags": ["should_refuse_missing_evidence", "retrieval_miss"],
            "difficulty": "medium",
        }
    ]

    case = parse_eval_cases(legacy_payload)[0]

    assert case.expected_behavior == EvalBehavior.ABSTAIN
    assert case.bucket_id == "should-refuse-missing-evidence"
    assert case.bucket_name == "Should Refuse Missing Evidence"
    assert case.refusal_reason_expected == "insufficient_evidence"
    assert case.gate_group == "general"
    assert case.case_kind == "portable"
    assert case.supported_backends == [ModelBackend.MOCK, ModelBackend.OPENAI]


def test_portable_case_set_filters_out_stress_cases():
    portable_cases = load_eval_cases("data/eval_dataset.json", case_set=CaseSet.PORTABLE)

    assert portable_cases
    assert all(case.case_kind == "portable" for case in portable_cases)
    assert all(ModelBackend.OPENAI in case.supported_backends for case in portable_cases)


def test_no_answer_behavior_scoring_distinguishes_safe_abstain_and_over_refusal():
    refuse_case = EvalCase(
        case_id="conflict-case",
        question="How long does a customer have to request a refund?",
        scenario=ScenarioName.CONFLICTING_EVIDENCE,
        expected_behavior=EvalBehavior.REFUSE,
        refusal_reason_expected="conflicting_evidence",
        bucket_id="conflicting-evidence-refuse",
        bucket_name="Conflicting Evidence Should Refuse",
        risk_tier=RiskTier.HIGH,
        business_criticality=BusinessCriticality.HIGH,
        gate_group="safety",
    )
    over_refusal_case = EvalCase(
        case_id="over-refusal-case",
        question="How long does a customer have to request a refund?",
        scenario=ScenarioName.NORMAL,
        expected_behavior=EvalBehavior.ANSWER,
        required_citation_ids=["policy-refund-30-standard"],
        acceptable_answer_facts=["30 days"],
        bucket_id="direct-answerable",
        bucket_name="Direct Answerable",
        risk_tier=RiskTier.HIGH,
        business_criticality=BusinessCriticality.HIGH,
        gate_group="safety",
    )

    response = SimpleNamespace(
        outcome=Outcome.REFUSED_MORE_EVIDENCE_NEEDED,
        answer=None,
        provisional_answer=None,
        missing_or_conflicting_evidence_summary="The retrieved evidence conflicts.",
        refusal_reason=RefusalReason.CONFLICTING_EVIDENCE,
        citations=[],
    )
    run_record = SimpleNamespace(
        validation_result=SimpleNamespace(structure_valid=True, citation_valid=True),
        retrieved_ids=["policy-refund-30-standard"],
        failure_reasons=[],
        failure_categories=[],
        suspicious_flags=[],
        score_breakdown=SimpleNamespace(
            brand_voice_score=1.0,
            tone_appropriateness_score=1.0,
            policy_adherence_score=1.0,
        ),
        prompt_version="qa-prompt:v1",
        model_backend=ModelBackend.MOCK,
        model_name="mock-policy-model",
        outcome=Outcome.REFUSED_MORE_EVIDENCE_NEEDED,
        run_id="run-1",
        trace_id="trace-1",
    )

    correct_refuse = score_offline_case(eval_run_id="eval-1", raw_case=refuse_case, response=response, run_record=run_record)
    incorrect_refuse = score_offline_case(eval_run_id="eval-1", raw_case=over_refusal_case, response=response, run_record=run_record)

    assert correct_refuse.actual_behavior == EvalBehavior.ABSTAIN
    assert correct_refuse.behavior_match is True
    assert correct_refuse.regression_blockers == []

    assert incorrect_refuse.actual_behavior == EvalBehavior.ABSTAIN
    assert incorrect_refuse.behavior_match is False
    assert "behavior_mismatch" in incorrect_refuse.regression_blockers
    assert "over_refusal" in incorrect_refuse.regression_blockers


def test_hard_fail_blocker_logic_marks_critical_bucket_failed():
    policy = load_gate_policy("data/offline_gate_policy.json")
    case = EvalCase(
        case_id="unsupported-claim-case",
        question="How long does a customer have to request a refund?",
        scenario=ScenarioName.UNSUPPORTED_ANSWER,
        expected_behavior=EvalBehavior.HUMAN_REVIEW,
        bucket_id="unsupported-claim-trap",
        bucket_name="Unsupported Claim Trap",
        risk_tier=RiskTier.CRITICAL,
        business_criticality=BusinessCriticality.HIGH,
        gate_group="safety",
    )
    result = build_case_result(
        case=case,
        score_breakdown=OfflineScoreBreakdown(
            schema_valid=1.0,
            citation_valid=1.0,
            retrieval_hit=1.0,
            refusal_correct=1.0,
            answer_fact_match=1.0,
            unsupported_claim_penalty=0.0,
            brand_voice_match=1.0,
            tone_match=1.0,
            policy_adherence_match=1.0,
        ),
        blockers=["unsupported_claim"],
    )

    gate_result = apply_gate_policy([result], [case], policy)

    assert gate_result.release_decision.value == "fail"
    assert gate_result.failed_buckets == ["unsupported-claim-trap"]
    assert any("hard fails" in reason.lower() for reason in gate_result.bucket_summaries[0].decision_reasons)


def test_bucket_threshold_logic_can_warn_without_full_release_failure():
    policy = load_gate_policy("data/offline_gate_policy.json")
    case_a = EvalCase(
        case_id="direct-a",
        question="How long does a customer have to request a refund?",
        scenario=ScenarioName.NORMAL,
        expected_behavior=EvalBehavior.ANSWER,
        bucket_id="direct-answerable",
        bucket_name="Direct Answerable",
        risk_tier=RiskTier.MEDIUM,
        business_criticality=BusinessCriticality.MEDIUM,
        gate_group="general",
    )
    case_b = case_a.model_copy(update={"case_id": "direct-b"})

    passing = build_case_result(case=case_a)
    low_score = build_case_result(
        case=case_b,
        score_breakdown=OfflineScoreBreakdown(
            schema_valid=0.0,
            citation_valid=0.0,
            retrieval_hit=0.0,
            refusal_correct=1.0,
            answer_fact_match=1.0,
            unsupported_claim_penalty=1.0,
            brand_voice_match=1.0,
            tone_match=1.0,
            policy_adherence_match=0.0,
        ),
    )

    gate_result = apply_gate_policy([passing, low_score], [case_a, case_b], policy)

    assert gate_result.release_decision.value == "warn"
    assert gate_result.bucket_summaries[0].release_decision.value == "warn"
    assert any("within configured tolerances" in reason for reason in gate_result.bucket_summaries[0].decision_reasons)


def test_offline_comparison_endpoint_persists_portable_case_set(client):
    payload = {
        "baseline": {
            "label": "baseline",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
            "retrieval_config_version": "retrieval-config:v1",
            "source_snapshot_id": "kb-snapshot:2026-04-01",
            "case_set": "portable",
        },
        "candidate": {
            "label": "candidate",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
            "retrieval_config_version": "retrieval-config:v1",
            "source_snapshot_id": "kb-snapshot:2026-04-01",
            "case_set": "portable",
        },
    }

    response = client.post("/policy-desk-assistant/evals/offline/compare", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["candidate_config"]["case_set"] == "portable"
    assert body["summary"]["compared_case_ids"]
    assert body["summary"]["excluded_case_ids"] == []
    assert body["summary"]["new_failures"] == []

    comparison_id = body["summary"]["comparison_id"]
    detail = client.get(f"/policy-desk-assistant/evals/offline/comparisons/{comparison_id}")
    assert detail.status_code == 200
    assert detail.json()["summary"]["comparison_id"] == comparison_id


def test_release_decision_fields_are_present_on_existing_offline_endpoint(client):
    response = client.get("/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["release_decision"] in {"pass", "warn", "fail"}
    assert isinstance(payload["summary"]["decision_reasons"], list)
    assert isinstance(payload["summary"]["by_bucket_breakdown"], list)
    assert "case_results" in payload


def test_comparison_requires_shared_case_set(client):
    response = client.post(
        "/policy-desk-assistant/evals/offline/compare",
        json={
            "baseline": {
                "label": "baseline",
                "model_backend": "mock",
                "prompt_version": "qa-prompt:v1",
                "retrieval_config_version": "retrieval-config:v1",
                "source_snapshot_id": "kb-snapshot:2026-04-01",
                "case_set": "portable",
            },
            "candidate": {
                "label": "candidate",
                "model_backend": "mock",
                "prompt_version": "qa-prompt:v2",
                "retrieval_config_version": "retrieval-config:v1",
                "source_snapshot_id": "kb-snapshot:2026-04-01",
                "case_set": "full",
            },
        },
    )

    assert response.status_code == 422


def test_comparison_ignores_new_advisory_regressions_for_failure_counts():
    policy = load_gate_policy("data/offline_gate_policy.json")
    case = EvalCase(
        case_id="brand-voice-case",
        question="How should we describe our annual plan?",
        scenario=ScenarioName.NORMAL,
        expected_behavior=EvalBehavior.ANSWER,
        required_citation_ids=["policy-brand-annual-plan"],
        acceptable_answer_facts=["annual plan"],
        bucket_id="tone-brand",
        bucket_name="Tone And Brand",
        risk_tier=RiskTier.LOW,
        business_criticality=BusinessCriticality.LOW,
        gate_group="quality",
    )

    workspace_tmp = ROOT / "test_artifacts" / str(uuid4())
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    dataset_path = workspace_tmp / "comparison_cases.json"
    dataset_path.write_text(json.dumps({"cases": [case.model_dump(mode="json")]}), encoding="utf-8")

    baseline_result = build_case_result(case=case)
    candidate_result = build_case_result(
        case=case,
        score_breakdown=OfflineScoreBreakdown(
            schema_valid=1.0,
            citation_valid=1.0,
            retrieval_hit=1.0,
            refusal_correct=1.0,
            answer_fact_match=0.0,
            unsupported_claim_penalty=1.0,
            brand_voice_match=0.85,
            tone_match=1.0,
            policy_adherence_match=1.0,
        ),
        blockers=["answer_fact_mismatch"],
    )

    def build_run(result: OfflineEvalCaseResult, prompt_version: str) -> OfflineEvalRun:
        result.prompt_version = prompt_version
        result.model_backend = ModelBackend.OPENAI
        result.model_name = "openai-demo-model"
        gate_result = apply_gate_policy([result], [case], policy)
        summary = summarize_offline_results(
            eval_run_id=f"eval-{prompt_version}",
            model_backend=ModelBackend.OPENAI,
            model_name="openai-demo-model",
            prompt_version=prompt_version,
            case_results=[result],
            retrieval_config_version="retrieval-config:v1",
            source_snapshot_id="kb-snapshot:test",
            gate_policy_version=policy.policy_version,
            by_bucket_breakdown=gate_result.bucket_summaries,
            release_decision=gate_result.release_decision,
            decision_reasons=gate_result.decision_reasons,
            failed_buckets=gate_result.failed_buckets,
        )
        return OfflineEvalRun(summary=summary, case_results=[result])

    baseline_run = build_run(baseline_result, "qa-prompt:v1")
    candidate_run = build_run(candidate_result, "qa-prompt:v2")

    class StorageStub:
        def __init__(self) -> None:
            self.saved_summary = None
            self.saved_case_deltas = None

        def write_offline_comparison(self, summary, case_deltas) -> None:
            self.saved_summary = summary
            self.saved_case_deltas = case_deltas

    storage = StorageStub()
    runner = OfflineEvalRunner(
        policy_service=SimpleNamespace(),
        storage=storage,
        dataset_path=str(dataset_path),
        gate_policy_path="data/offline_gate_policy.json",
    )

    runs_by_prompt = {
        "qa-prompt:v1": baseline_run,
        "qa-prompt:v2": candidate_run,
    }
    runner._run_config = lambda config, cases=None: runs_by_prompt[config.prompt_version]  # type: ignore[method-assign]

    try:
        comparison = runner.compare(
            OfflineComparisonRequest(
                baseline=OfflineEvalConfig(
                    label="baseline",
                    model_backend=ModelBackend.OPENAI,
                    prompt_version="qa-prompt:v1",
                    case_set=CaseSet.PORTABLE,
                ),
                candidate=OfflineEvalConfig(
                    label="candidate",
                    model_backend=ModelBackend.OPENAI,
                    prompt_version="qa-prompt:v2",
                    case_set=CaseSet.PORTABLE,
                ),
            )
        )
        assert baseline_run.case_results[0].passed is True
        assert candidate_run.case_results[0].passed is True
        assert comparison.case_deltas[0].new_failures == []
        assert comparison.case_deltas[0].new_blocking_failures == []
        assert comparison.summary.new_failures == []
        assert comparison.summary.new_blocking_failures == []
        assert comparison.summary.release_decision == ReleaseDecision.PASS
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)
