from __future__ import annotations

from datetime import UTC, datetime

from model.contracts import OnlineMetricsSummary


def create_run(client, *, question: str, scenario: str = "normal", prompt_version: str = "qa-prompt:v1") -> dict:
    response = client.post(
        "/policy-desk-assistant/respond",
        json={
            "question": question,
            "scenario": scenario,
            "model_backend": "mock",
            "prompt_version": prompt_version,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_feedback_event_validation_and_persistence(client):
    run = create_run(client, question="How long does a customer have to request a refund?")

    invalid = client.post(
        "/policy-desk-assistant/feedback",
        json={
            "run_id": run["run_id"],
            "event_type": "not_a_valid_event",
        },
    )
    assert invalid.status_code == 422

    created = client.post(
        "/policy-desk-assistant/feedback",
        json={
            "run_id": run["run_id"],
            "event_type": "thumbs_down",
            "session_id": "session-1",
            "event_value": "customer disagreed",
        },
    )
    assert created.status_code == 200
    event = created.json()
    assert event["run_id"] == run["run_id"]
    assert event["event_type"] == "thumbs_down"
    assert event["response_outcome"] == run["outcome"]
    assert event["risk_band"] == run["risk_band"]

    events = client.get(f"/policy-desk-assistant/feedback?run_id={run['run_id']}")
    assert events.status_code == 200
    body = events.json()
    assert len(body) == 1
    assert body[0]["session_id"] == "session-1"


def test_feedback_summary_aggregation(client):
    supported = create_run(client, question="How long does a customer have to request a refund?")
    refused = create_run(
        client,
        question="What is the policy for pet insurance reimbursement?",
        scenario="retrieval_miss",
    )
    review = create_run(
        client,
        question="How long does a customer have to request a refund?",
        scenario="unsupported_answer",
    )
    repaired = create_run(
        client,
        question="When do unused service credits expire?",
        scenario="malformed_json",
    )

    for run_id, event_type in [
        (supported["run_id"], "thumbs_up"),
        (repaired["run_id"], "thumbs_down"),
        (refused["run_id"], "thumbs_down"),
    ]:
        response = client.post(
            "/policy-desk-assistant/feedback",
            json={"run_id": run_id, "event_type": event_type},
        )
        assert response.status_code == 200

    response = client.get("/policy-desk-assistant/feedback/summary?limit=10")
    assert response.status_code == 200
    payload = response.json()
    summary = payload["summary"]

    assert summary["total_runs"] == 4
    assert summary["total_feedback_events"] == 3
    assert summary["outcome_counts"]["supported_answer"] == 1
    assert summary["outcome_counts"]["refused_more_evidence_needed"] == 1
    assert summary["outcome_counts"]["human_review_recommended"] == 2
    assert summary["review_recommended_rate"] == 0.5
    assert summary["supported_answer_rate"] == 0.25
    assert summary["invalid_output_rate"] == 0.25
    assert summary["thumbs_up_rate"] == 0.25
    assert summary["thumbs_down_rate"] == 0.5
    assert summary["high_risk_run_count"] == 2
    assert "avg_groundedness_proxy_score" in summary
    assert "avg_online_score_total" in summary
    assert summary["by_prompt_version"]
    assert summary["by_model_backend"][0]["group_key"] == "mock"


def test_alert_policy_evaluation_marks_action_required(client):
    summary = OnlineMetricsSummary(
        window_start=datetime.now(UTC),
        window_end=datetime.now(UTC),
        total_runs=10,
        total_feedback_events=4,
        outcome_counts={"supported_answer": 2, "human_review_recommended": 8},
        risk_band_counts={"high": 7, "medium": 3},
        avg_groundedness_proxy_score=0.4,
        avg_citation_validity_score=0.5,
        avg_policy_adherence_score=0.3,
        avg_format_validity_score=0.6,
        avg_retrieval_support_score=0.45,
        avg_brand_voice_score=0.8,
        avg_tone_appropriateness_score=0.8,
        avg_online_score_total=0.42,
        review_recommended_rate=0.8,
        supported_answer_rate=0.2,
        invalid_output_rate=0.4,
        thumbs_up_rate=0.1,
        thumbs_down_rate=0.3,
        high_risk_run_count=7,
        by_prompt_version=[],
        by_model_backend=[],
        by_response_outcome=[],
        by_risk_band=[],
    )

    evaluation = client.app.state.online_control_service.evaluate_summary(summary)

    assert evaluation.status.value == "action_required"
    assert evaluation.rollback_recommended is True
    assert any(alert.metric_name == "invalid_output_rate" for alert in evaluation.triggered_alerts)
    assert any("Supported answer rate" in reason for reason in evaluation.rollback_reasons)


def test_rollback_recommended_stays_false_for_watch_only_alert(client):
    summary = OnlineMetricsSummary(
        window_start=datetime.now(UTC),
        window_end=datetime.now(UTC),
        total_runs=20,
        total_feedback_events=1,
        outcome_counts={"supported_answer": 18, "human_review_recommended": 2},
        risk_band_counts={"medium": 20},
        avg_groundedness_proxy_score=0.9,
        avg_citation_validity_score=0.9,
        avg_policy_adherence_score=0.9,
        avg_format_validity_score=0.9,
        avg_retrieval_support_score=0.9,
        avg_brand_voice_score=0.9,
        avg_tone_appropriateness_score=0.9,
        avg_online_score_total=0.9,
        review_recommended_rate=0.1,
        supported_answer_rate=0.9,
        invalid_output_rate=0.0,
        thumbs_up_rate=0.15,
        thumbs_down_rate=0.22,
        high_risk_run_count=0,
        by_prompt_version=[],
        by_model_backend=[],
        by_response_outcome=[],
        by_risk_band=[],
    )

    evaluation = client.app.state.online_control_service.evaluate_summary(summary)

    assert evaluation.status.value == "watch"
    assert evaluation.rollback_recommended is False
    assert evaluation.rollback_reasons == []


def test_action_required_summary_auto_enqueues_review_items_without_duplicate_spam(client):
    created_runs = [
        create_run(client, question=f"How long does a customer have to request a refund? {index}")
        for index in range(3)
    ]

    queue_before = client.get("/policy-desk-assistant/review-queue")
    assert queue_before.status_code == 200
    assert queue_before.json() == []

    for run in created_runs:
        response = client.post(
            "/policy-desk-assistant/feedback",
            json={"run_id": run["run_id"], "event_type": "thumbs_down"},
        )
        assert response.status_code == 200

    first_summary = client.get("/policy-desk-assistant/feedback/summary?limit=3")
    assert first_summary.status_code == 200
    first_payload = first_summary.json()
    assert first_payload["alert_evaluation"]["status"] == "action_required"
    assert len(first_payload["review_queue_item_ids"]) == 3

    queue_after_first = client.get("/policy-desk-assistant/review-queue")
    assert queue_after_first.status_code == 200
    assert len(queue_after_first.json()) == 3
    assert {item["run_id"] for item in queue_after_first.json()} == {run["run_id"] for run in created_runs}

    second_summary = client.get("/policy-desk-assistant/feedback/summary?limit=3")
    assert second_summary.status_code == 200
    queue_after_second = client.get("/policy-desk-assistant/review-queue")
    assert len(queue_after_second.json()) == 3
