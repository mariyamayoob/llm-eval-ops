from __future__ import annotations

from evals.contracts import ReviewQueueItem, ReviewerAnnotation
from model.contracts import RunRecord, RunSummary
from services.storage_service import StorageService


class RunService:
    def __init__(self, storage: StorageService) -> None:
        self.storage = storage

    def write_run(self, record: RunRecord) -> None:
        self.storage.write_run(record)

    def list_runs(self, limit: int = 50) -> list[RunSummary]:
        return self.storage.list_runs(limit=limit)

    def list_run_records(self, limit: int = 50) -> list[RunRecord]:
        return self.storage.list_run_records(limit=limit)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.storage.get_run(run_id)

    def enqueue_review(self, item: ReviewQueueItem) -> None:
        self.storage.create_review_item(item)

    def find_review_item_by_run(self, run_id: str) -> ReviewQueueItem | None:
        return self.storage.find_review_item_by_run(run_id)

    def list_review_queue(self) -> list[ReviewQueueItem]:
        return self.storage.list_review_queue()

    def prune_review_queue_open_runtime(self, *, max_open_runtime_items: int = 20) -> int:
        return self.storage.prune_review_queue_open_runtime(max_open_runtime_items=max_open_runtime_items)

    def annotate_review(self, item_id: str, annotation: ReviewerAnnotation) -> ReviewQueueItem | None:
        return self.storage.annotate_review_item(item_id, annotation)
