from __future__ import annotations


def test_supported_answer_path(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "How long does a customer have to request a refund?",
            "scenario": "normal",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "supported_answer"
    assert payload["review_required"] is False
    assert payload["citations"] == ["policy-refund-30-standard"]


def test_refused_more_evidence_needed_path(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "What is the policy for pet insurance reimbursement?",
            "scenario": "retrieval_miss",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
        },
    )
    payload = response.json()
    assert payload["outcome"] == "refused_more_evidence_needed"
    assert payload["refusal_reason"] == "insufficient_evidence"


def test_sloppy_prompt_routes_missing_evidence_to_review(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "What is the policy for pet insurance reimbursement?",
            "scenario": "retrieval_miss",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v2",
        },
    )
    payload = response.json()
    assert payload["outcome"] == "human_review_recommended"
    assert payload["review_required"] is True


def test_human_review_recommended_path(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "How long does a customer have to request a refund?",
            "scenario": "unsupported_answer",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
        },
    )
    payload = response.json()
    assert payload["outcome"] == "human_review_recommended"
    assert payload["review_required"] is True
    assert payload["review_queue_item_id"]


def test_malformed_json_repair_creates_review(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "When do unused service credits expire?",
            "scenario": "malformed_json",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
        },
    )
    payload = response.json()
    assert payload["outcome"] == "human_review_recommended"
    assert "repair_attempted" in payload["suspicious_flags"]


def test_conflicting_evidence_handling(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "How long does a customer have to request a refund?",
            "scenario": "conflicting_evidence",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
        },
    )
    payload = response.json()
    assert payload["outcome"] in {"refused_more_evidence_needed", "human_review_recommended"}


def test_review_queue_item_creation(client):
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
    queue = client.get("/policy-desk-assistant/review-queue")
    assert queue.status_code == 200
    assert any(item["review_queue_item_id"] == item_id for item in queue.json())


def test_trace_id_and_run_persistence(client):
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "What should support do when VPN access fails after a password reset?",
            "scenario": "normal",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v1",
        },
    )
    payload = response.json()
    assert payload["trace_id"]
    run = client.get(f"/policy-desk-assistant/runs/{payload['run_id']}")
    assert run.status_code == 200
    body = run.json()
    assert body["run"]["retrieved_ids"]
    assert body["run"]["step_timings_ms"]


def test_unsupported_prompt_versions_are_rejected(client):
    respond = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": "How long does a customer have to request a refund?",
            "scenario": "normal",
            "model_backend": "mock",
            "prompt_version": "qa-prompt:v6",
        },
    )
    assert respond.status_code == 422

    offline = client.get("/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v6")
    assert offline.status_code == 422
