from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from uuid import uuid4

from evals.contracts import ReviewQueueItem
from model.contracts import (
    FeedbackEventType,
    LLMJudgeAssessment,
    LLMJudgeRecord,
    LLMJudgeStatus,
    Outcome,
    ReviewPriority,
    RiskBand,
)
from services.kb_service import KBService
from services.run_service import RunService
from services.storage_service import StorageService


SAMPLE_RATES = {
    RiskBand.LOW: 0.10,
    RiskBand.MEDIUM: 0.30,
    RiskBand.HIGH: 1.00,
}


class OpenAIJudgeService:
    def __init__(
        self,
        *,
        kb_service: KBService,
        run_service: RunService,
        storage: StorageService,
    ) -> None:
        self.kb_service = kb_service
        self.run_service = run_service
        self.storage = storage
        self.enabled = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("LLM_JUDGE_ENABLED", "true").lower() != "false"
        self.model_name = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-mini")

    def maybe_judge_run(self, run_id: str) -> LLMJudgeRecord | None:
        if not self.enabled:
            return None

        run = self.run_service.get_run(run_id)
        if run is None or run.question is None:
            return None
        if run.review_required:
            return None
        existing = self.storage.list_llm_judge_records(limit=1, run_id=run_id)
        if existing:
            return existing[0]

        sample_rate = SAMPLE_RATES[run.risk_band]
        if self._sample_bucket(run.run_id) >= sample_rate:
            return None

        record = LLMJudgeRecord(
            judge_id=str(uuid4()),
            run_id=run.run_id,
            created_at=datetime.now(UTC),
            status=LLMJudgeStatus.QUEUED,
            judge_model=self.model_name,
            sampling_rate=sample_rate,
            model_backend=run.model_backend,
            prompt_version=run.prompt_version,
            response_outcome=run.outcome,
            risk_band=run.risk_band,
        )
        self.storage.write_llm_judge_record(record)
        return self._execute_judge(run=run, record=record)

    def list_judge_records(self, *, limit: int = 50, run_id: str | None = None) -> list[LLMJudgeRecord]:
        return self.storage.list_llm_judge_records(limit=limit, run_id=run_id)

    def get_judge_record(self, judge_id: str) -> LLMJudgeRecord | None:
        return self.storage.get_llm_judge_record(judge_id)

    def force_judge_run(self, run_id: str) -> LLMJudgeRecord | None:
        """Bypass sampling for demo/operator use. Still skips runs already in human review."""
        if not self.enabled:
            return None
        run = self.run_service.get_run(run_id)
        if run is None or run.question is None:
            return None
        if run.review_required:
            return None
        existing = self.storage.list_llm_judge_records(limit=1, run_id=run_id)
        if existing:
            return existing[0]

        record = LLMJudgeRecord(
            judge_id=str(uuid4()),
            run_id=run.run_id,
            created_at=datetime.now(UTC),
            status=LLMJudgeStatus.QUEUED,
            judge_model=self.model_name,
            sampling_rate=1.0,
            model_backend=run.model_backend,
            prompt_version=run.prompt_version,
            response_outcome=run.outcome,
            risk_band=run.risk_band,
        )
        self.storage.write_llm_judge_record(record)
        return self._execute_judge(run=run, record=record)

    def _execute_judge(self, *, run, record: LLMJudgeRecord) -> LLMJudgeRecord:
        try:
            response = self._client().responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": self._judge_system_prompt()},
                    {"role": "user", "content": json.dumps(self._judge_payload(run))},
                ],
                temperature=0,
                max_output_tokens=700,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "llm_judge_assessment",
                        "strict": True,
                        "schema": self._strict_json_schema(LLMJudgeAssessment.model_json_schema()),
                    }
                },
            )
            output_text = getattr(response, "output_text", None)
            if not output_text:
                raise ValueError("OpenAI judge response did not contain output_text")
            assessment = LLMJudgeAssessment.model_validate_json(output_text)
            record.status = LLMJudgeStatus.COMPLETED
            record.completed_at = datetime.now(UTC)
            record.openai_response_id = getattr(response, "id", None)
            record.assessment = assessment
            self.storage.write_llm_judge_record(record)

            if assessment.human_review_recommended:
                review_item = self._ensure_review_for_run(
                    run_id=run.run_id,
                    reason=assessment.human_review_reason or "LLM judge recommended human review.",
                )
                record.review_queue_item_id = review_item.review_queue_item_id
                self.storage.write_llm_judge_record(record)
            return record
        except Exception as exc:
            record.status = LLMJudgeStatus.FAILED
            record.completed_at = datetime.now(UTC)
            record.error_message = str(exc)
            self.storage.write_llm_judge_record(record)
            return record

    def _ensure_review_for_run(self, *, run_id: str, reason: str) -> ReviewQueueItem:
        existing = self.run_service.find_review_item_by_run(run_id)
        if existing is not None:
            return existing

        run = self.run_service.get_run(run_id)
        if run is None:
            raise LookupError(f"Run {run_id} was not found.")

        item = ReviewQueueItem(
            review_queue_item_id=str(uuid4()),
            run_id=run.run_id,
            trace_id=run.trace_id,
            online_score_total=run.online_score_total,
            review_priority=ReviewPriority.HIGH if run.risk_band == RiskBand.HIGH else ReviewPriority.MEDIUM,
            suspicious_flags=run.suspicious_flags,
            review_source="llm_judge",
            review_reason=reason,
        )
        self.run_service.enqueue_review(item)

        run.review_required = True
        run.review_queue_item_id = item.review_queue_item_id
        self.run_service.write_run(run)
        return item

    def _judge_payload(self, run) -> dict[str, object]:
        response_payload = run.response_payload
        response_text = (
            response_payload.answer
            or response_payload.provisional_answer
            or response_payload.missing_or_conflicting_evidence_summary
            or ""
        )
        retrieved_chunks = [
            {"id": chunk["id"], "title": chunk["title"], "body": str(chunk["body"])[:400]}
            for chunk in self.kb_service.get_chunks(run.retrieved_ids)
        ]
        return {
            "question": run.question,
            "response_outcome": run.outcome.value,
            "response_text": response_text,
            "citations": response_payload.citations,
            "evidence_summary": [item.model_dump() for item in response_payload.evidence_summary],
            "retrieved_chunks": retrieved_chunks,
            "runtime": {
                "risk_band": run.risk_band.value,
                "online_score_total": run.online_score_total,
                "suspicious_flags": [flag.value for flag in run.suspicious_flags],
                "score_breakdown": run.score_breakdown.model_dump(),
                "thumbs_down_already_recorded": any(
                    event.event_type == FeedbackEventType.THUMBS_DOWN
                    for event in self.storage.list_feedback_events(run_id=run.run_id, limit=10)
                ),
            },
        }

    def _judge_system_prompt(self) -> str:
        return (
            "You are an LLM judge for a policy assistant. "
            "Grade a single response using only the question, retrieved policy evidence, and the returned answer or refusal. "
            "Use a clear rubric and be conservative about unsupported claims. "
            "Return JSON only. "
            "Scoring rubric: "
            "supportedness_score is how well the response is backed by retrieved evidence; "
            "policy_alignment_score is how well the response follows policy and avoids invented details; "
            "response_mode_score is whether answer vs refusal vs later human review was the right choice. "
            "overall_score should reflect the average of those dimensions. "
            "Recommend human review when evidence is weak or conflicting, the response mode seems wrong, or a policy-risk answer should be inspected by a person. "
            "Keep rationale brief and concrete."
        )

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError("The openai package is required for the LLM judge flow.") from exc
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _sample_bucket(self, run_id: str) -> float:
        digest = hashlib.sha256(f"llm-judge:{run_id}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _strict_json_schema(self, schema: dict[str, object]) -> dict[str, object]:
        def walk(node: object) -> object:
            if isinstance(node, dict):
                result = {key: walk(value) for key, value in node.items()}
                if result.get("type") == "object":
                    result.setdefault("additionalProperties", False)
                    if "properties" in result:
                        result["required"] = sorted(result["properties"].keys())
                return result
            if isinstance(node, list):
                return [walk(value) for value in node]
            return node

        return walk(schema)
