from __future__ import annotations

from evals.runner import OPENAI_SUPPORTED_SCENARIOS
from model.contracts import ScenarioName


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
    assert payload["risk_band"] == "high"
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
    item_id = response.json()["review_queue_item_id"]
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


def test_openai_supported_scenarios_subset_is_explicit():
    assert ScenarioName.NORMAL in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.RETRIEVAL_MISS in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.CONFLICTING_EVIDENCE in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.SLOW_RESPONSE in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.MALFORMED_JSON not in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.UNSUPPORTED_ANSWER not in OPENAI_SUPPORTED_SCENARIOS
    assert ScenarioName.WRONG_REFUSAL not in OPENAI_SUPPORTED_SCENARIOS
