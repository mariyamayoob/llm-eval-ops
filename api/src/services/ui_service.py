from __future__ import annotations

from evals.contracts import OfflineComparisonSummary, OfflineEvalSummary, ReviewQueueItem
from model.contracts import RunRecord
from observability.tracing import TraceRecord


class UIService:
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

    def offline_comparison_summary(self, summary: OfflineComparisonSummary, case_deltas) -> dict:
        return {
            "summary": summary.model_dump(),
            "case_deltas": [item.model_dump() for item in case_deltas],
        }

    def review_queue(self, items: list[ReviewQueueItem]) -> list[dict]:
        return [item.model_dump() for item in items]
