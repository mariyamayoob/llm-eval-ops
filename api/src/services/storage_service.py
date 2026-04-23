from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from evals.contracts import (
    OfflineComparisonCaseDelta,
    OfflineComparisonSummary,
    OfflineEvalCaseResult,
    OfflineEvalSummary,
    ReviewQueueItem,
    ReviewerAnnotation,
)
from model.contracts import LLMJudgeRecord, RunRecord, RunSummary, RuntimeFeedbackEvent


class StorageService:
    def __init__(self, db_path: str = "data/policy_desk.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS run_records (run_id TEXT PRIMARY KEY, created_at TEXT, trace_id TEXT, scenario TEXT, outcome TEXT, review_required INTEGER, online_score_total REAL, record_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS offline_eval_runs (eval_run_id TEXT PRIMARY KEY, created_at TEXT, summary_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS offline_eval_cases (eval_run_id TEXT, case_id TEXT, result_json TEXT, PRIMARY KEY(eval_run_id, case_id))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS review_queue (review_queue_item_id TEXT PRIMARY KEY, run_id TEXT, trace_id TEXT, review_status TEXT, priority TEXT, item_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS reviewer_annotations (review_queue_item_id TEXT, reviewer_label TEXT, annotation_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runtime_feedback_events (event_id TEXT PRIMARY KEY, run_id TEXT, created_at TEXT, event_type TEXT, event_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_judge_records (judge_id TEXT PRIMARY KEY, run_id TEXT, created_at TEXT, status TEXT, record_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS offline_comparison_runs (comparison_id TEXT PRIMARY KEY, created_at TEXT, summary_json TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS offline_comparison_cases (comparison_id TEXT, case_id TEXT, delta_json TEXT, PRIMARY KEY(comparison_id, case_id))"
            )

    def write_run(self, record: RunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO run_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.run_id,
                    record.created_at.isoformat(),
                    record.trace_id,
                    record.scenario.value,
                    record.outcome.value,
                    int(record.review_required),
                    record.online_score_total,
                    record.model_dump_json(),
                ),
            )

    def list_runs(self, limit: int = 50) -> list[RunSummary]:
        with self._connect() as conn:
            rows = conn.execute("SELECT record_json FROM run_records ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._summary_from_record_json(row[0]) for row in rows]

    def list_run_records(self, limit: int = 50) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT record_json FROM run_records ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [RunRecord.model_validate_json(row[0]) for row in rows]

    def _summary_from_record_json(self, raw: str) -> RunSummary:
        record = RunRecord.model_validate_json(raw)
        return RunSummary(
            run_id=record.run_id,
            trace_id=record.trace_id,
            created_at=record.created_at,
            model_backend=record.model_backend,
            model_name=record.model_name,
            prompt_version=record.prompt_version,
            retrieval_config_version=record.retrieval_config_version,
            source_snapshot_id=record.source_snapshot_id,
            scenario=record.scenario,
            outcome=record.outcome,
            online_score_total=record.online_score_total,
            risk_band=record.risk_band,
            review_required=record.review_required,
            review_queue_item_id=record.review_queue_item_id,
            suspicious_flags=record.suspicious_flags,
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT record_json FROM run_records WHERE run_id = ?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def write_offline_eval(self, summary: OfflineEvalSummary, case_results: list[OfflineEvalCaseResult]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO offline_eval_runs VALUES (?, ?, ?)",
                (summary.eval_run_id, summary.created_at.isoformat(), summary.model_dump_json()),
            )
            for case_result in case_results:
                conn.execute(
                    "INSERT OR REPLACE INTO offline_eval_cases VALUES (?, ?, ?)",
                    (summary.eval_run_id, case_result.case_id, case_result.model_dump_json()),
                )

    def latest_offline_eval_summary(self) -> OfflineEvalSummary | None:
        with self._connect() as conn:
            row = conn.execute("SELECT summary_json FROM offline_eval_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        return OfflineEvalSummary.model_validate_json(row[0]) if row else None

    def get_offline_eval(self, eval_run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            summary_row = conn.execute("SELECT summary_json FROM offline_eval_runs WHERE eval_run_id = ?", (eval_run_id,)).fetchone()
            case_rows = conn.execute("SELECT result_json FROM offline_eval_cases WHERE eval_run_id = ? ORDER BY case_id", (eval_run_id,)).fetchall()
        if not summary_row:
            return None
        return {
            "summary": OfflineEvalSummary.model_validate_json(summary_row[0]),
            "case_results": [OfflineEvalCaseResult.model_validate_json(row[0]) for row in case_rows],
        }

    def write_offline_comparison(
        self,
        summary: OfflineComparisonSummary,
        case_deltas: list[OfflineComparisonCaseDelta],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO offline_comparison_runs VALUES (?, ?, ?)",
                (summary.comparison_id, summary.created_at.isoformat(), summary.model_dump_json()),
            )
            for delta in case_deltas:
                conn.execute(
                    "INSERT OR REPLACE INTO offline_comparison_cases VALUES (?, ?, ?)",
                    (summary.comparison_id, delta.case_id, delta.model_dump_json()),
                )

    def get_offline_comparison(self, comparison_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            summary_row = conn.execute(
                "SELECT summary_json FROM offline_comparison_runs WHERE comparison_id = ?",
                (comparison_id,),
            ).fetchone()
            delta_rows = conn.execute(
                "SELECT delta_json FROM offline_comparison_cases WHERE comparison_id = ? ORDER BY case_id",
                (comparison_id,),
            ).fetchall()
        if not summary_row:
            return None
        return {
            "summary": OfflineComparisonSummary.model_validate_json(summary_row[0]),
            "case_deltas": [OfflineComparisonCaseDelta.model_validate_json(row[0]) for row in delta_rows],
        }

    def create_review_item(self, item: ReviewQueueItem) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO review_queue VALUES (?, ?, ?, ?, ?, ?)",
                (
                    item.review_queue_item_id,
                    item.run_id,
                    item.trace_id,
                    item.review_status.value,
                    item.review_priority.value,
                    item.model_dump_json(),
                ),
            )

    def find_review_item_by_run(self, run_id: str) -> ReviewQueueItem | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT item_json FROM review_queue WHERE run_id = ? ORDER BY rowid DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        return ReviewQueueItem.model_validate_json(row[0]) if row else None

    def list_review_queue(self) -> list[ReviewQueueItem]:
        with self._connect() as conn:
            rows = conn.execute("SELECT item_json FROM review_queue ORDER BY priority DESC, review_status ASC").fetchall()
        return [ReviewQueueItem.model_validate_json(row[0]) for row in rows]

    def prune_review_queue_open_runtime(self, *, max_open_runtime_items: int = 20) -> int:
        """Demo-only cleanup to keep the Review Queue usable in screenshots.

        Deletes older *pending* runtime review items (not llm_judge items) that have not
        been touched by a reviewer. This is not a production retention policy.
        """
        if max_open_runtime_items < 0:
            return 0

        with self._connect() as conn:
            rows = conn.execute("SELECT rowid, item_json FROM review_queue ORDER BY rowid DESC").fetchall()

            candidates: list[int] = []
            for rowid, raw in rows:
                item = ReviewQueueItem.model_validate_json(raw)
                if item.review_source == "llm_judge":
                    continue
                if item.review_status.value != "pending":
                    continue
                if item.promote_to_offline_eval:
                    continue
                if item.reviewer_label or item.reviewer_notes:
                    continue
                if item.final_disposition is not None:
                    continue
                if item.should_have_outcome is not None:
                    continue
                if getattr(item, "should_have_response_text", None):
                    continue
                candidates.append(int(rowid))

            to_delete = candidates[max_open_runtime_items:]
            if not to_delete:
                return 0

            placeholders = ", ".join("?" for _ in to_delete)
            conn.execute(f"DELETE FROM review_queue WHERE rowid IN ({placeholders})", tuple(to_delete))
            return len(to_delete)

    def annotate_review_item(self, item_id: str, annotation: ReviewerAnnotation) -> ReviewQueueItem | None:
        with self._connect() as conn:
            row = conn.execute("SELECT item_json FROM review_queue WHERE review_queue_item_id = ?", (item_id,)).fetchone()
            if not row:
                return None
            item = ReviewQueueItem.model_validate_json(row[0])
            item.review_status = annotation.review_status
            item.reviewer_label = annotation.reviewer_label
            item.reviewer_notes = annotation.reviewer_notes
            if annotation.final_disposition is not None:
                item.final_disposition = annotation.final_disposition
            item.promote_to_offline_eval = annotation.promote_to_offline_eval
            item.should_have_outcome = annotation.should_have_outcome
            if annotation.should_have_response_text is not None:
                item.should_have_response_text = annotation.should_have_response_text
            conn.execute(
                "UPDATE review_queue SET review_status = ?, item_json = ? WHERE review_queue_item_id = ?",
                (item.review_status.value, item.model_dump_json(), item_id),
            )
            conn.execute(
                "INSERT INTO reviewer_annotations VALUES (?, ?, ?)",
                (item_id, annotation.reviewer_label, annotation.model_dump_json()),
            )
            return item

    def clear_all_tables(self) -> dict[str, int]:
        """Delete all rows from the demo SQLite tables (keeps schema + db file)."""
        tables = [
            "reviewer_annotations",
            "review_queue",
            "runtime_feedback_events",
            "llm_judge_records",
            "run_records",
            "offline_eval_cases",
            "offline_eval_runs",
            "offline_comparison_cases",
            "offline_comparison_runs",
        ]
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            for table in tables:
                deleted[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
        return deleted

    def clear_review_and_judge_tables(self) -> dict[str, int]:
        """Delete all rows from human review + judge tables (keeps runs/metrics)."""
        tables = [
            "reviewer_annotations",
            "review_queue",
            "llm_judge_records",
        ]
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            for table in tables:
                deleted[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
        return deleted

    def write_feedback_event(self, event: RuntimeFeedbackEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runtime_feedback_events VALUES (?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.run_id,
                    event.created_at.isoformat(),
                    event.event_type.value,
                    event.model_dump_json(),
                ),
            )

    def list_feedback_events(
        self,
        *,
        limit: int = 50,
        run_id: str | None = None,
        run_ids: list[str] | None = None,
    ) -> list[RuntimeFeedbackEvent]:
        if run_ids is not None and not run_ids:
            return []
        query = "SELECT event_json FROM runtime_feedback_events"
        params: list[object] = []
        clauses: list[str] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if run_ids:
            placeholders = ", ".join("?" for _ in run_ids)
            clauses.append(f"run_id IN ({placeholders})")
            params.extend(run_ids)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [RuntimeFeedbackEvent.model_validate_json(row[0]) for row in rows]

    def write_llm_judge_record(self, record: LLMJudgeRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_judge_records VALUES (?, ?, ?, ?, ?)",
                (
                    record.judge_id,
                    record.run_id,
                    record.created_at.isoformat(),
                    record.status.value,
                    record.model_dump_json(),
                ),
            )

    def get_llm_judge_record(self, judge_id: str) -> LLMJudgeRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT record_json FROM llm_judge_records WHERE judge_id = ?", (judge_id,)).fetchone()
        return LLMJudgeRecord.model_validate_json(row[0]) if row else None

    def list_llm_judge_records(self, *, limit: int = 50, run_id: str | None = None) -> list[LLMJudgeRecord]:
        query = "SELECT record_json FROM llm_judge_records"
        params: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY created_at DESC"
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [LLMJudgeRecord.model_validate_json(row[0]) for row in rows]
