from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from main import create_app


ROOT = Path(__file__).resolve().parents[3]


def test_offline_eval_export_redacts_and_emits_case(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_ENABLED", "false")
    workspace_tmp = ROOT / "test_artifacts" / str(uuid.uuid4())
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    db_path = workspace_tmp / "policy_desk.db"
    app = create_app(db_path=str(db_path))

    try:
        with TestClient(app) as client:
            response = client.post(
                "/policy-desk-assistant/respond",
                json={
                    "question": "Hi, my email is alice@example.com and my phone is 212-555-0100. How long do I have to request a refund?",
                    "scenario": "unsupported_answer",
                    "model_backend": "mock",
                    "prompt_version": "qa-prompt:v1",
                },
            )
            assert response.status_code == 200
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

            annotate = client.post(
                f"/policy-desk-assistant/review-queue/{item_id}/annotate",
                json={
                    "reviewer_label": "demo",
                    "reviewer_notes": "promote to offline",
                    "review_status": "resolved",
                    "final_disposition": "corrected",
                    "promote_to_offline_eval": True,
                    "should_have_response_text": "The refund window is 30 days. Contact alice@example.com or call 212-555-0100 for help.",
                },
            )
            assert annotate.status_code == 200

            exported = client.get("/policy-desk-assistant/offline-eval/export")
            assert exported.status_code == 200
            payload = exported.json()
            assert "cases" in payload
            assert len(payload["cases"]) == 1
            case = payload["cases"][0]
            assert "alice@example.com" not in case["question"]
            assert "212-555-0100" not in case["question"]
            assert "[REDACTED_EMAIL]" in case["question"]
            assert "[REDACTED_PHONE]" in case["question"]
            assert case["expected_behavior"] == "human_review"
            assert "alice@example.com" not in (case.get("reference_response_text") or "")
            assert "212-555-0100" not in (case.get("reference_response_text") or "")
            assert "[REDACTED_EMAIL]" in (case.get("reference_response_text") or "")
            assert "[REDACTED_PHONE]" in (case.get("reference_response_text") or "")
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)
