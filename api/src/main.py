from __future__ import annotations

from evals.contracts import CaseSet, OfflineComparisonRequest, ReviewerAnnotation
from evals.runner import OfflineEvalRunner
from model.contracts import (
    ModelBackend,
    PolicyDeskAssistantRequest,
    PolicyDeskAssistantResponse,
    RuntimeFeedbackRequest,
)
from observability.tracing import TraceRecorder
from prompts.registry import DEFAULT_PROMPT_VERSION, PROMPT_VERSION_PATTERN
from services.kb_service import KBService
from services.human_review_router import HumanReviewRouter
from services.metrics_service import OnlineScoringService
from services.llm_judge_service import OpenAIJudgeService
from services.model_service import ModelService
from services.online_control_service import OnlineControlPlaneService
from services.offline_eval_export_service import OfflineEvalExportService
from services.qa_service import PolicyDeskService
from services.retrieval_service import RetrievalService
from services.run_service import RunService
from services.storage_service import StorageService
from services.ui_service import UIService
from services.validation_service import ValidationService
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel


BASE_PATH = "/policy-desk-assistant"


class DevResetRequest(BaseModel):
    confirm: str


def create_app(
    kb_path: str = "data/kb.json",
    db_path: str = "data/policy_desk.db",
    eval_dataset_path: str = "data/eval_dataset.json",
    offline_gate_policy_path: str = "data/offline_gate_policy.json",
    online_alert_policy_path: str = "data/online_alert_policy.json",
) -> FastAPI:
    app = FastAPI(title="Policy Desk Assistant", version="1.0.0")

    kb_service = KBService(kb_path=kb_path)
    retrieval_service = RetrievalService(kb_service=kb_service)
    model_service = ModelService()
    validation_service = ValidationService(model_service=model_service)
    scoring_service = OnlineScoringService()
    storage_service = StorageService(db_path=db_path)
    run_service = RunService(storage=storage_service)
    review_router = HumanReviewRouter(run_service=run_service)
    tracer = TraceRecorder()
    policy_service = PolicyDeskService(
        kb_service=kb_service,
        retrieval_service=retrieval_service,
        model_service=model_service,
        validation_service=validation_service,
        scoring_service=scoring_service,
        run_service=run_service,
        review_router=review_router,
        tracer=tracer,
    )
    eval_runner = OfflineEvalRunner(
        policy_service=policy_service,
        storage=storage_service,
        dataset_path=eval_dataset_path,
        gate_policy_path=offline_gate_policy_path,
    )
    online_control_service = OnlineControlPlaneService(
        run_service=run_service,
        storage=storage_service,
        review_router=review_router,
        alert_policy_path=online_alert_policy_path,
    )
    llm_judge_service = OpenAIJudgeService(
        kb_service=kb_service,
        run_service=run_service,
        storage=storage_service,
        review_router=review_router,
    )
    offline_export_service = OfflineEvalExportService(run_service=run_service)
    ui_service = UIService()

    app.state.policy_service = policy_service
    app.state.run_service = run_service
    app.state.storage_service = storage_service
    app.state.tracer = tracer
    app.state.eval_runner = eval_runner
    app.state.online_control_service = online_control_service
    app.state.llm_judge_service = llm_judge_service
    app.state.offline_export_service = offline_export_service
    app.state.ui_service = ui_service

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post(f"{BASE_PATH}/respond", response_model=PolicyDeskAssistantResponse)
    def respond(request: PolicyDeskAssistantRequest, background_tasks: BackgroundTasks):
        response = policy_service.respond(request)
        if llm_judge_service.should_schedule_judge(
            run_id=response.run_id,
            risk_band=response.risk_band,
            review_required=response.review_required,
        ):
            background_tasks.add_task(llm_judge_service.maybe_judge_run, response.run_id)
        return response

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

    @app.post(f"{BASE_PATH}/feedback")
    def create_feedback(request: RuntimeFeedbackRequest):
        try:
            event = online_control_service.record_feedback(request)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail={"run_id": request.run_id, "message": str(exc)}) from exc
        return ui_service.feedback_event(event)

    @app.get(f"{BASE_PATH}/feedback")
    def list_feedback(limit: int = 50, run_id: str | None = None):
        return ui_service.feedback_events(online_control_service.list_feedback(limit=limit, run_id=run_id))

    @app.get(f"{BASE_PATH}/feedback/summary")
    def feedback_summary(limit: int = 50):
        return ui_service.online_control_summary(online_control_service.build_live_summary(limit=limit))

    @app.get(f"{BASE_PATH}/llm-judge")
    def list_llm_judge(limit: int = 50, run_id: str | None = None):
        return ui_service.llm_judge_records(llm_judge_service.list_judge_records(limit=limit, run_id=run_id))

    @app.get(f"{BASE_PATH}/llm-judge/{{judge_id}}")
    def get_llm_judge(judge_id: str):
        record = llm_judge_service.get_judge_record(judge_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"judge_id": judge_id, "message": "LLM judge record not found"})
        return ui_service.llm_judge_record(record)

    @app.post(f"{BASE_PATH}/llm-judge/run/{{run_id}}")
    def force_llm_judge(run_id: str, background_tasks: BackgroundTasks):
        background_tasks.add_task(llm_judge_service.force_judge_run, run_id)
        return {"status": "queued", "run_id": run_id}

    @app.get(f"{BASE_PATH}/offline-eval/export")
    def export_offline_eval_cases():
        return {"cases": offline_export_service.export_promoted_cases()}

    @app.get(f"{BASE_PATH}/evals/offline")
    def run_offline_evals(
        model_backend: ModelBackend = Query(default=ModelBackend.MOCK),
        prompt_version: str = Query(default=DEFAULT_PROMPT_VERSION, pattern=PROMPT_VERSION_PATTERN),
        retrieval_config_version: str = Query(default="retrieval-config:v1"),
        source_snapshot_id: str = Query(default="kb-snapshot:current"),
        case_set: CaseSet = Query(default=CaseSet.FULL),
    ):
        return eval_runner.run(
            model_backend=model_backend,
            prompt_version=prompt_version,
            retrieval_config_version=retrieval_config_version,
            source_snapshot_id=source_snapshot_id,
            case_set=case_set,
        )

    @app.post(f"{BASE_PATH}/evals/offline/compare")
    def compare_offline_evals(request: OfflineComparisonRequest):
        return eval_runner.compare(request)

    @app.get(f"{BASE_PATH}/evals/offline/{{eval_run_id}}")
    def get_offline_eval(eval_run_id: str):
        payload = storage_service.get_offline_eval(eval_run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail={"eval_run_id": eval_run_id, "message": "Offline eval run not found"})
        return ui_service.offline_eval_summary(payload["summary"], payload["case_results"])

    @app.get(f"{BASE_PATH}/evals/offline/comparisons/{{comparison_id}}")
    def get_offline_comparison(comparison_id: str):
        payload = storage_service.get_offline_comparison(comparison_id)
        if payload is None:
            raise HTTPException(status_code=404, detail={"comparison_id": comparison_id, "message": "Offline comparison run not found"})
        return ui_service.offline_comparison_summary(payload["summary"], payload["case_deltas"])

    @app.get(f"{BASE_PATH}/review-queue")
    def review_queue():
        return ui_service.review_queue(run_service.list_review_queue())

    @app.post(f"{BASE_PATH}/review-queue/{{item_id}}/annotate")
    def annotate_review_queue(item_id: str, annotation: ReviewerAnnotation):
        item = run_service.annotate_review(item_id, annotation)
        if item is None:
            raise HTTPException(status_code=404, detail={"item_id": item_id, "message": "Review queue item not found"})
        return item

    @app.post(f"{BASE_PATH}/review-queue/prune")
    def prune_review_queue(*, max_open_runtime_items: int = 20):
        deleted_count = run_service.prune_review_queue_open_runtime(max_open_runtime_items=max_open_runtime_items)
        return {"deleted_count": deleted_count, "max_open_runtime_items": max_open_runtime_items}

    @app.post(f"{BASE_PATH}/dev/reset")
    def dev_reset(request: DevResetRequest):
        if request.confirm.strip().upper() != "RESET":
            raise HTTPException(status_code=400, detail={"message": "Confirmation required. Send {'confirm':'RESET'}."})
        deleted = storage_service.clear_all_tables()
        return {"status": "cleared", "deleted": deleted}

    @app.post(f"{BASE_PATH}/dev/reset/review-and-judge")
    def dev_reset_review_and_judge(request: DevResetRequest):
        if request.confirm.strip().upper() != "RESET":
            raise HTTPException(status_code=400, detail={"message": "Confirmation required. Send {'confirm':'RESET'}."})
        deleted = storage_service.clear_review_and_judge_tables()
        return {"status": "cleared", "deleted": deleted}

    return app


app = create_app()
