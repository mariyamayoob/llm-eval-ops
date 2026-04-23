from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from main import create_app


ROOT = Path(__file__).resolve().parents[3]


class _FakeJudgeResponse:
    def __init__(self, *, output_text: str, response_id: str = "resp_judge_1") -> None:
        self.output_text = output_text
        self.id = response_id


class _FakeJudgeClient:
    def __init__(self, *, output_text: str) -> None:
        self.responses = self
        self._output_text = output_text

    def create(self, **kwargs):
        return _FakeJudgeResponse(output_text=self._output_text)


def _judge_payload(*, recommend_human_review: bool) -> str:
    reason = "Evidence support is too weak for a fully trusted answer." if recommend_human_review else None
    return json.dumps(
        {
            "supportedness_score": 0.42,
            "policy_alignment_score": 0.55,
            "response_mode_score": 0.48,
            "overall_score": 0.48,
            "human_review_recommended": recommend_human_review,
            "human_review_reason": reason,
            "confidence": 0.71,
            "rationale": "The response is readable but not fully supported by the retrieved evidence.",
        }
    )


def _judge_enabled_client(monkeypatch):
    monkeypatch.setenv("LLM_JUDGE_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    workspace_tmp = ROOT / "test_artifacts" / str(uuid.uuid4())
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    db_path = workspace_tmp / "policy_desk.db"
    app = create_app(db_path=str(db_path))
    return workspace_tmp, app


def test_llm_judge_persists_completed_record(monkeypatch):
    workspace_tmp, app = _judge_enabled_client(monkeypatch)
    service = app.state.llm_judge_service
    monkeypatch.setattr(service, "_sample_bucket", lambda run_id: 0.0)
    monkeypatch.setattr(service, "_client", lambda: _FakeJudgeClient(output_text=_judge_payload(recommend_human_review=False)))

    try:
        with TestClient(app) as client:
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

            judge_records = client.get("/policy-desk-assistant/llm-judge?limit=10")
            assert judge_records.status_code == 200
            body = judge_records.json()
            assert len(body) == 1
            assert body[0]["status"] == "completed"
            assert body[0]["assessment"]["human_review_recommended"] is False

            queue = client.get("/policy-desk-assistant/review-queue")
            assert queue.status_code == 200
            assert queue.json() == []
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)


def test_llm_judge_can_route_to_human_review(monkeypatch):
    workspace_tmp, app = _judge_enabled_client(monkeypatch)
    service = app.state.llm_judge_service
    monkeypatch.setattr(service, "_sample_bucket", lambda run_id: 0.0)
    monkeypatch.setattr(service, "_client", lambda: _FakeJudgeClient(output_text=_judge_payload(recommend_human_review=True)))

    try:
        with TestClient(app) as client:
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
            run_id = response.json()["run_id"]

            judge_records = client.get("/policy-desk-assistant/llm-judge?limit=10")
            assert judge_records.status_code == 200
            body = judge_records.json()
            assert len(body) == 1
            assert body[0]["assessment"]["human_review_recommended"] is True
            assert body[0]["review_queue_item_id"]

            queue = client.get("/policy-desk-assistant/review-queue")
            assert queue.status_code == 200
            assert len(queue.json()) == 1
            assert queue.json()[0]["run_id"] == run_id
            assert queue.json()[0]["review_source"] == "llm_judge"

            run = client.get(f"/policy-desk-assistant/runs/{run_id}")
            assert run.status_code == 200
            assert run.json()["run"]["review_required"] is True
            assert run.json()["run"]["review_queue_item_id"] == body[0]["review_queue_item_id"]
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)


def test_force_llm_judge_endpoint_bypasses_sampling(monkeypatch):
    workspace_tmp, app = _judge_enabled_client(monkeypatch)
    service = app.state.llm_judge_service
    monkeypatch.setattr(service, "_sample_bucket", lambda run_id: 0.999)
    monkeypatch.setattr(service, "_client", lambda: _FakeJudgeClient(output_text=_judge_payload(recommend_human_review=True)))

    try:
        with TestClient(app) as client:
            response = client.post(
                "/policy-desk-assistant/respond",
                json={
                    "question": "How long does a customer have to request a refund?",
                    "scenario": "normal",
                    "model_backend": "mock",
                    "prompt_version": "qa-prompt:v1",
                },
            )
            run_id = response.json()["run_id"]

            forced = client.post(f"/policy-desk-assistant/llm-judge/run/{run_id}")
            assert forced.status_code == 200

            judge_records = client.get("/policy-desk-assistant/llm-judge?limit=10")
            assert judge_records.status_code == 200
            assert len(judge_records.json()) == 1
            assert judge_records.json()[0]["assessment"]["human_review_recommended"] is True
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)


def test_low_risk_run_skips_llm_judge_sampling(monkeypatch):
    workspace_tmp, app = _judge_enabled_client(monkeypatch)
    service = app.state.llm_judge_service
    monkeypatch.setattr(service, "_sample_bucket", lambda run_id: 0.2)

    try:
        with TestClient(app) as client:
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

            judge_records = client.get("/policy-desk-assistant/llm-judge?limit=10")
            assert judge_records.status_code == 200
            assert judge_records.json() == []
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)


def test_medium_risk_run_is_selected_for_llm_judge(monkeypatch):
    workspace_tmp, app = _judge_enabled_client(monkeypatch)
    service = app.state.llm_judge_service
    monkeypatch.setattr(service, "_sample_bucket", lambda run_id: 0.0)
    monkeypatch.setattr(service, "_client", lambda: _FakeJudgeClient(output_text=_judge_payload(recommend_human_review=False)))

    try:
        with TestClient(app) as client:
            response = client.post(
                "/policy-desk-assistant/respond",
                json={
                    "question": "How long does a customer have to request a refund?",
                    "scenario": "wrong_refusal",
                    "model_backend": "mock",
                    "prompt_version": "qa-prompt:v1",
                },
            )
            assert response.status_code == 200

            judge_records = client.get("/policy-desk-assistant/llm-judge?limit=10")
            assert judge_records.status_code == 200
            body = judge_records.json()
            assert len(body) == 1
            assert body[0]["status"] == "completed"
            assert body[0]["risk_band"] == "medium"
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)
