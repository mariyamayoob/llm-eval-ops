from __future__ import annotations

from evals.contracts import OfflineEvalSummary, ReviewQueueItem
from model.contracts import RunRecord, RunSummary
from observability.tracing import TraceRecord


class UIService:
    def inference_card(self, response) -> dict:
        return response.model_dump()

    def run_explorer(self, run_record: RunRecord, trace_record: TraceRecord | None) -> dict:
        return {
            "run": run_record.model_dump(),
            "trace": trace_record.model_dump() if trace_record else None,
        }

    def offline_eval_summary(self, summary: OfflineEvalSummary, case_results) -> dict:
        return {
            "summary": summary.model_dump(),
            "case_results": [item.model_dump() for item in case_results],
        }

    def review_queue(self, items: list[ReviewQueueItem]) -> list[dict]:
        return [item.model_dump() for item in items]
