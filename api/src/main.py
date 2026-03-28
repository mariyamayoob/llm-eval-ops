from __future__ import annotations

from evals.contracts import ReviewerAnnotation
from evals.runner import OfflineEvalRunner
from model.contracts import ModelBackend, PolicyDeskAssistantRequest, PolicyDeskAssistantResponse
from observability.tracing import TraceRecorder
from services.kb_service import KBService
from services.metrics_service import OnlineScoringService
from services.model_service import ModelService
from services.qa_service import PolicyDeskService
from services.retrieval_service import RetrievalService
from services.run_service import RunService
from services.storage_service import StorageService
from services.ui_service import UIService
from services.validation_service import ValidationService
from fastapi import FastAPI, HTTPException, Query


BASE_PATH = "/policy-desk-assistant"


def create_app(
    kb_path: str = "data/kb.json",
    db_path: str = "data/policy_desk.db",
    eval_dataset_path: str = "data/eval_dataset.json",
) -> FastAPI:
    app = FastAPI(title="Policy Desk Assistant", version="1.0.0")

    kb_service = KBService(kb_path=kb_path)
    retrieval_service = RetrievalService(kb_service=kb_service)
    model_service = ModelService()
    validation_service = ValidationService(model_service=model_service)
    scoring_service = OnlineScoringService()
    storage_service = StorageService(db_path=db_path)
    run_service = RunService(storage=storage_service)
    tracer = TraceRecorder()
    policy_service = PolicyDeskService(
        kb_service=kb_service,
        retrieval_service=retrieval_service,
        model_service=model_service,
        validation_service=validation_service,
        scoring_service=scoring_service,
        run_service=run_service,
        tracer=tracer,
    )
    eval_runner = OfflineEvalRunner(policy_service=policy_service, storage=storage_service, dataset_path=eval_dataset_path)
    ui_service = UIService()

    app.state.policy_service = policy_service
    app.state.run_service = run_service
    app.state.storage_service = storage_service
    app.state.tracer = tracer
    app.state.eval_runner = eval_runner
    app.state.ui_service = ui_service

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post(f"{BASE_PATH}/respond", response_model=PolicyDeskAssistantResponse)
    def respond(request: PolicyDeskAssistantRequest):
        return policy_service.respond(request)

    @app.get(f"{BASE_PATH}/runs")
    def list_runs(limit: int = 50):
        return run_service.list_runs(limit=limit)

    @app.get(f"{BASE_PATH}/runs/{{run_id}}")
    def get_run(run_id: str):
        record = run_service.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"run_id": run_id, "message": "Run not found"})
        trace = tracer.get_trace(record.trace_id)
        return ui_service.run_explorer(record, trace)

    @app.get(f"{BASE_PATH}/evals/offline")
    def run_offline_evals(
        model_backend: ModelBackend = Query(default=ModelBackend.MOCK),
        prompt_version: str = Query(default="qa-prompt:v1"),
    ):
        return eval_runner.run(model_backend=model_backend, prompt_version=prompt_version)

    @app.get(f"{BASE_PATH}/evals/offline/{{eval_run_id}}")
    def get_offline_eval(eval_run_id: str):
        payload = storage_service.get_offline_eval(eval_run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail={"eval_run_id": eval_run_id, "message": "Offline eval run not found"})
        return ui_service.offline_eval_summary(payload["summary"], payload["case_results"])

    @app.get(f"{BASE_PATH}/review-queue")
    def review_queue():
        return ui_service.review_queue(run_service.list_review_queue())

    @app.post(f"{BASE_PATH}/review-queue/{{item_id}}/annotate")
    def annotate_review_queue(item_id: str, annotation: ReviewerAnnotation):
        item = run_service.annotate_review(item_id, annotation)
        if item is None:
            raise HTTPException(status_code=404, detail={"item_id": item_id, "message": "Review queue item not found"})
        return item

    return app


app = create_app()
