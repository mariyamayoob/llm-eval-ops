from __future__ import annotations

from types import SimpleNamespace

from evals.runner import OPENAI_SUPPORTED_SCENARIOS
from model.contracts import ScenarioName, StructuredPolicyOutput, SuspiciousFlag, ValidationResult
from services.metrics_service import OnlineScoringService


def test_online_scoring_thresholds_and_flags(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "How long does a customer have to request a refund?",
            "scenario": "wrong_refusal",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v2",
        },
    )
    payload = response.json()
    assert payload["risk_band"] == "medium"
    assert "refusal_despite_strong_retrieval" in payload["suspicious_flags"]


def test_review_queue_annotation(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "How long does a customer have to request a refund?",
            "scenario": "unsupported_answer",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
        },
    )
    run_id = response.json()["run_id"]

    feedback = client.post(
        "/policy-desk-assistant/feedback",
        json={"run_id": run_id, "event_type": "thumbs_down"},
    )
    assert feedback.status_code == 200

    queue = client.get("/policy-desk-assistant/review-queue")
    assert queue.status_code == 200
    item = next((candidate for candidate in queue.json() if candidate["run_id"] == run_id), None)
    assert item is not None
    item_id = item["review_queue_item_id"]
    annotation = client.post(
        f"/policy-desk-assistant/review-queue/{item_id}/annotate",
        json={
            "reviewer_label": "reviewer-1",
            "reviewer_notes": "Unsupported claim confirmed.",
            "review_status": "resolved",
            "final_disposition": "rejected_response",
        },
    )
    assert annotation.status_code == 200
    assert annotation.json()["final_disposition"] == "rejected_response"


def test_offline_eval_runner_and_summary(client):
    response = client.get("/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["aggregate_metrics"]["overall"] >= 0.0
    assert payload["summary"]["failure_taxonomy_counts"] is not None
    eval_id = payload["summary"]["eval_run_id"]

    detail = client.get(f"/policy-desk-assistant/evals/offline/{eval_id}")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert len(detail_payload["case_results"]) >= 1


def test_offline_scorer_catches_wrong_refusal_case(client):
    response = client.get("/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v2")
    payload = response.json()
    assert any(
        case["case_id"] == "wrong_refusal_case" and case["score_breakdown"]["refusal_correct"] == 0.0
        for case in payload["case_results"]
    )


def test_offline_eval_payload_includes_score_breakdown_total(client):
    response = client.get("/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v1")
    payload = response.json()
    assert all("total" in case["score_breakdown"] for case in payload["case_results"])


def test_offline_eval_summary_surfaces_story_metrics(client):
    response = client.get("/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v1&case_set=portable")
    metrics = response.json()["summary"]["aggregate_metrics"]

    assert "valid_grounded_score" in metrics
    assert "behavior_score" in metrics
    assert "advisory_quality_score" in metrics
    assert "unsupported_claim_penalty" in metrics


def test_offline_eval_v1_can_pass_on_portable_cases(client):
    response = client.get("/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v1&case_set=portable")
    payload = response.json()
    assert payload["summary"]["release_decision"] == "pass"
    assert payload["summary"]["failed_buckets"] == []
    assert all(case["passed"] for case in payload["case_results"])


def test_offline_eval_v2_fails_on_portable_cases(client):
    response = client.get("/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v2&case_set=portable")
    payload = response.json()
    assert payload["summary"]["prompt_version"] == "qa-prompt:v2"
    assert payload["summary"]["release_decision"] == "fail"
    assert "missing-evidence-abstain" in payload["summary"]["failed_buckets"]
    assert any(case["case_id"] == "retrieval_miss_case" and not case["passed"] for case in payload["case_results"])


def test_offline_compare_can_detect_v2_regression_on_portable_cases(client):
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
                "case_set": "portable",
            },
        },
    )
    payload = response.json()
    assert payload["summary"]["release_decision"] == "fail"
    assert payload["summary"]["excluded_case_ids"] == []
    assert "retrieval_miss_case" in payload["summary"]["new_failures"]
    assert any(item.startswith("retrieval_miss_case:") for item in payload["summary"]["new_blocking_failures"])
    assert payload["summary"]["compared_case_ids"]


def test_openai_supported_scenarios_subset_is_explicit():
    assert ScenarioName.NORMAL in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.RETRIEVAL_MISS in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.CONFLICTING_EVIDENCE in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.SLOW_RESPONSE in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.MALFORMED_JSON not in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.UNSUPPORTED_ANSWER not in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.WRONG_REFUSAL not in OPENAI_SUPPORTED_SCENARIOS


def test_tone_mismatch_is_advisory_for_online_review_routing():
    scoring = OnlineScoringService()
    structured_output = StructuredPolicyOutput.model_validate(
        {
            "answer": "A" * 221,
            "citations": ["policy-refund-30-standard"],
            "evidence_summary": [
                {
                    "chunk_id": "policy-refund-30-standard",
                    "title": "Standard refund window",
                    "support_snippet": "Customers may request a refund within 30 days.",
                    "relevance_score": 0.95,
                }
            ],
            "refusal": False,
            "refusal_reason": None,
            "missing_or_conflicting_evidence_summary": None,
            "confidence": 0.8,
        }
    )
    validation = ValidationResult(
        structure_valid=True,
        citation_valid=True,
        repair_attempted=False,
        failure_reasons=[],
        failure_categories=[],
    )
    retrieval_stats = SimpleNamespace(retrieval_empty=False, similarity_max=0.3)

    _, flags, risk_band, review = scoring.score(
        structured_output=structured_output,
        validation=validation,
        retrieval_stats=retrieval_stats,
    )

    assert SuspiciousFlag.TONE_MISMATCH in flags
    assert review.review_required is False
    assert risk_band.value == "medium"


def test_high_confidence_low_support_refusal_is_advisory():
    scoring = OnlineScoringService()
    structured_output = StructuredPolicyOutput.model_validate(
        {
            "answer": "",
            "citations": [],
            "evidence_summary": [],
            "refusal": True,
            "refusal_reason": "insufficient_evidence",
            "missing_or_conflicting_evidence_summary": "There is not enough retrieved support to answer.",
            "confidence": 0.98,
        }
    )
    validation = ValidationResult(
        structure_valid=True,
        citation_valid=True,
        repair_attempted=False,
        failure_reasons=[],
        failure_categories=[],
    )
    retrieval_stats = SimpleNamespace(retrieval_empty=True, similarity_max=0.0)

    _, flags, risk_band, review = scoring.score(
        structured_output=structured_output,
        validation=validation,
        retrieval_stats=retrieval_stats,
    )

    assert SuspiciousFlag.HIGH_CONFIDENCE_LOW_SUPPORT not in flags
    assert review.review_required is False
    assert risk_band.value == "medium"
