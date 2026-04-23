from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from uuid import uuid4

from model.contracts import (
    FeedbackEventType,
    LLMJudgeAssessment,
    LLMJudgeRecord,
    LLMJudgeStatus,
    ReviewPriority,
    RiskBand,
)
from services.kb_service import KBService
from services.human_review_router import HumanReviewRouter, ReviewRouteRequest
from services.run_service import RunService
from services.storage_service import StorageService


SAMPLE_RATES = {
    RiskBand.LOW: 0.20,
    RiskBand.MEDIUM: 1.00,
    RiskBand.HIGH: 1.00,
}


class OpenAIJudgeService:
    def __init__(
        self,
        *,
        kb_service: KBService,
        run_service: RunService,
        storage: StorageService,
        review_router: HumanReviewRouter,
    ) -> None:
        self.kb_service = kb_service
        self.run_service = run_service
        self.storage = storage
        self.review_router = review_router
        self.enabled = bool(os.getenv("OPENAI_API_KEY")) and os.getenv("LLM_JUDGE_ENABLED", "true").lower() != "false"
        self.model_name = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-mini")

    def should_schedule_judge(self, *, run_id: str, risk_band: RiskBand, review_required: bool) -> bool:
        if not self.enabled:
            return False
        sample_rate = SAMPLE_RATES[risk_band]
        return self._sample_bucket(run_id) < sample_rate

    def maybe_judge_run(self, run_id: str) -> LLMJudgeRecord | None:
        if not self.enabled:
            return None

        run = self.run_service.get_run(run_id)
        if run is None or run.question is None:
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
        """Bypass sampling for demo/operator use."""
        if not self.enabled:
            return None
        run = self.run_service.get_run(run_id)
        if run is None or run.question is None:
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
                base_reason = assessment.human_review_reason or "LLM judge recommended human review."
                if run.risk_band == RiskBand.LOW:
                    reason = f"Runtime risk band was LOW, but semantic review recommends human review: {base_reason}"
                    reason_codes = ["semantic_review_recommended", "runtime_low_risk_judge_override"]
                else:
                    reason = base_reason
                    reason_codes = ["semantic_review_recommended"]
                review_item = self.review_router.ensure_review_item(
                    ReviewRouteRequest(
                        run_id=run.run_id,
                        source="llm_judge",
                        reason=reason,
                        reason_codes=reason_codes,
                        priority=ReviewPriority.HIGH if run.risk_band == RiskBand.HIGH else ReviewPriority.MEDIUM,
                        metadata={
                            "judge_id": record.judge_id,
                            "judge_model": record.judge_model,
                            "overall_score": assessment.overall_score,
                            "runtime_risk_band": run.risk_band.value,
                        },
                    )
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
        return int(digest[:8], 16) / 0x100000000

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


# Clearer architecture alias (used in docs/articles).
SemanticReviewOrchestrator = OpenAIJudgeService
