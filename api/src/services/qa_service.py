from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

from evals.contracts import ReviewQueueItem
from model.contracts import (
    FailureCategory,
    FailureReason,
    Outcome,
    PolicyDeskAssistantRequest,
    PolicyDeskAssistantResponse,
    RefusalReason,
    ReviewDecision,
    ReviewPriority,
    RiskBand,
    RunRecord,
    ScenarioName,
    StructuredPolicyOutput,
    SuspiciousFlag,
)
from observability.tracing import TraceRecorder
from services.kb_service import KBService
from services.metrics_service import OnlineScoringService
from services.model_service import ModelCallResult, ModelService
from services.retrieval_service import RetrievalService
from services.run_service import RunService
from services.validation_service import ValidationService


class PolicyDeskService:
    def __init__(
        self,
        kb_service: KBService,
        retrieval_service: RetrievalService,
        model_service: ModelService,
        validation_service: ValidationService,
        scoring_service: OnlineScoringService,
        run_service: RunService,
        tracer: TraceRecorder,
    ) -> None:
        self.kb_service = kb_service
        self.retrieval_service = retrieval_service
        self.model_service = model_service
        self.validation_service = validation_service
        self.scoring_service = scoring_service
        self.run_service = run_service
        self.tracer = tracer

    def respond(self, request: PolicyDeskAssistantRequest) -> PolicyDeskAssistantResponse:
        run_id = str(uuid4())
        question_hash = hashlib.sha256(request.question.encode("utf-8")).hexdigest()[:16]
        trace_id = self.tracer.start_trace(
            run_id=run_id,
            scenario=request.scenario,
            prompt_version=request.prompt_version,
            question_hash=question_hash,
            question_len=len(request.question),
        )

        timings: dict[str, float] = {}
        retrieved_chunks_payload: list[dict[str, object]] = []

        retrieve_span = self.tracer.start_span(trace_id, "retrieve", scenario=request.scenario.value)
        retrieved_chunks, retrieval_stats = self.retrieval_service.retrieve(request.question, request.scenario.value)
        retrieved_ids = [item.id for item in retrieved_chunks]
        retrieved_chunks_payload = self.kb_service.get_chunks(retrieved_ids)
        timings["retrieve"] = retrieve_span.finish(
            retrieved_ids=retrieved_ids,
            candidate_count=retrieval_stats.candidate_count,
            retrieval_empty=retrieval_stats.retrieval_empty,
            similarity_max=retrieval_stats.similarity_max,
        )

        prompt_span = self.tracer.start_span(trace_id, "prompt_build", prompt_version=request.prompt_version)
        timings["prompt_build"] = prompt_span.finish(retrieved_count=len(retrieved_ids))

        llm_span = self.tracer.start_span(trace_id, "llm_call", model_backend=request.model_backend.value)
        model_result = self.model_service.call(
            model_backend=request.model_backend,
            question=request.question,
            retrieved_chunks=retrieved_chunks_payload,
            prompt_version=request.prompt_version,
            scenario=request.scenario,
        )
        timings["llm_call"] = llm_span.finish(model_name=model_result.model_name)

        parse_span = self.tracer.start_span(trace_id, "output_parse")
        cited_bodies = {chunk["id"]: str(chunk["body"]) for chunk in retrieved_chunks_payload}
        structured_output, validation = self.validation_service.parse_and_validate(
            raw_output=model_result.raw_text,
            model_backend=request.model_backend,
            prompt_version=request.prompt_version,
            retrieved_ids=retrieved_ids,
            cited_bodies=cited_bodies,
        )
        timings["output_parse"] = parse_span.finish(structure_valid=validation.structure_valid)

        validate_span = self.tracer.start_span(trace_id, "output_validate")
        timings["output_validate"] = validate_span.finish(
            citation_valid=validation.citation_valid,
            repair_attempted=validation.repair_attempted,
            failure_reasons=[reason.value for reason in validation.failure_reasons],
        )

        if validation.repair_attempted:
            repair_span = self.tracer.start_span(trace_id, "output_repair")
            timings["output_repair"] = repair_span.finish(repair_attempted=True)

        score_span = self.tracer.start_span(trace_id, "online_score")
        score_breakdown, suspicious_flags, risk_band, review_decision = self.scoring_service.score(
            structured_output=structured_output,
            validation=validation,
            retrieval_stats=retrieval_stats,
            prompt_version=request.prompt_version,
        )
        timings["online_score"] = score_span.finish(
            online_score_total=score_breakdown.total,
            suspicious_flags=[flag.value for flag in suspicious_flags],
        )

        outcome = self._choose_outcome(structured_output, review_decision)
        review_queue_item_id = None

        review_span = self.tracer.start_span(trace_id, "review_route")
        if review_decision.review_required:
            review_queue_item_id = str(uuid4())
            queue_item = ReviewQueueItem(
                review_queue_item_id=review_queue_item_id,
                run_id=run_id,
                trace_id=trace_id,
                online_score_total=score_breakdown.total,
                review_priority=review_decision.review_priority or ReviewPriority.HIGH,
                suspicious_flags=suspicious_flags,
            )
            self.run_service.enqueue_review(queue_item)
        timings["review_route"] = review_span.finish(
            review_required=review_decision.review_required,
            review_queue_item_id=review_queue_item_id,
        )

        response = self._build_response(
            run_id=run_id,
            trace_id=trace_id,
            request=request,
            model_result=model_result,
            structured_output=structured_output,
            outcome=outcome,
            score_breakdown=score_breakdown,
            risk_band=risk_band,
            suspicious_flags=suspicious_flags,
            review_decision=review_decision,
            review_queue_item_id=review_queue_item_id,
        )

        persist_span = self.tracer.start_span(trace_id, "persist_run")
        failure_reasons = validation.failure_reasons.copy()
        failure_categories = validation.failure_categories.copy()
        failure_reasons, failure_categories = self._augment_failures(
            scenario=request.scenario,
            structured_output=structured_output,
            retrieval_stats=retrieval_stats,
            suspicious_flags=suspicious_flags,
            failure_reasons=failure_reasons,
            failure_categories=failure_categories,
        )
        run_record = RunRecord(
            run_id=run_id,
            trace_id=trace_id,
            created_at=datetime.now(UTC),
            model_backend=model_result.backend,
            model_name=model_result.model_name,
            prompt_version=request.prompt_version,
            question_hash=question_hash,
            question_len=len(request.question),
            scenario=request.scenario,
            retrieved_ids=retrieved_ids,
            retrieval_stats=retrieval_stats,
            outcome=outcome,
            validation_result=validation,
            score_breakdown=score_breakdown,
            online_score_total=score_breakdown.total,
            risk_band=risk_band,
            suspicious_flags=suspicious_flags,
            review_required=review_decision.review_required,
            review_queue_item_id=review_queue_item_id,
            failure_categories=failure_categories,
            failure_reasons=failure_reasons,
            step_timings_ms=timings,
            response_payload=response,
        )
        self.run_service.write_run(run_record)
        timings["persist_run"] = persist_span.finish(outcome=outcome.value)
        run_record.step_timings_ms = timings
        self.run_service.write_run(run_record)

        self.tracer.finalize_trace(
            trace_id,
            suspicious_flags=suspicious_flags,
            outcome=outcome.value,
            risk_band=risk_band.value,
            review_required=review_decision.review_required,
            review_queue_item_id=review_queue_item_id,
            failure_reasons=[reason.value for reason in failure_reasons],
        )
        return response

    def _choose_outcome(self, structured_output: StructuredPolicyOutput, review_decision: ReviewDecision) -> Outcome:
        if review_decision.review_required:
            return Outcome.HUMAN_REVIEW_RECOMMENDED
        if structured_output.refusal:
            return Outcome.REFUSED_MORE_EVIDENCE_NEEDED
        return Outcome.SUPPORTED_ANSWER

    def _build_response(
        self,
        *,
        run_id: str,
        trace_id: str,
        request: PolicyDeskAssistantRequest,
        model_result: ModelCallResult,
        structured_output: StructuredPolicyOutput,
        outcome: Outcome,
        score_breakdown,
        risk_band: RiskBand,
        suspicious_flags: list[SuspiciousFlag],
        review_decision: ReviewDecision,
        review_queue_item_id: str | None,
    ) -> PolicyDeskAssistantResponse:
        common = {
            "run_id": run_id,
            "trace_id": trace_id,
            "model_backend": model_result.backend,
            "model_name": model_result.model_name,
            "prompt_version": request.prompt_version,
            "outcome": outcome,
            "online_score_total": score_breakdown.total,
            "risk_band": risk_band,
            "suspicious_flags": suspicious_flags,
            "review_required": review_decision.review_required,
            "score_breakdown": score_breakdown,
            "confidence": structured_output.confidence,
        }
        if outcome == Outcome.SUPPORTED_ANSWER:
            return PolicyDeskAssistantResponse(
                **common,
                answer=structured_output.answer,
                citations=structured_output.citations,
                evidence_summary=structured_output.evidence_summary,
            )
        if outcome == Outcome.REFUSED_MORE_EVIDENCE_NEEDED:
            return PolicyDeskAssistantResponse(
                **common,
                refusal_reason=structured_output.refusal_reason,
                missing_or_conflicting_evidence_summary=structured_output.missing_or_conflicting_evidence_summary,
            )
        return PolicyDeskAssistantResponse(
            **common,
            provisional_answer=structured_output.answer,
            citations=structured_output.citations,
            evidence_summary=structured_output.evidence_summary,
            review_priority=review_decision.review_priority,
            review_queue_item_id=review_queue_item_id,
            human_review_reason=review_decision.human_review_reason,
        )

    def _augment_failures(
        self,
        *,
        scenario: ScenarioName,
        structured_output: StructuredPolicyOutput,
        retrieval_stats,
        suspicious_flags: list[SuspiciousFlag],
        failure_reasons: list[FailureReason],
        failure_categories: list[FailureCategory],
    ) -> tuple[list[FailureReason], list[FailureCategory]]:
        if retrieval_stats.retrieval_empty:
            failure_reasons.append(FailureReason.NO_RETRIEVAL_HIT)
            failure_categories.append(FailureCategory.RETRIEVAL_FAILURE)
        elif retrieval_stats.similarity_max < 0.2:
            failure_reasons.append(FailureReason.WEAK_RETRIEVAL_HIT)
            failure_categories.append(FailureCategory.RETRIEVAL_FAILURE)
        if scenario == ScenarioName.WRONG_REFUSAL or (structured_output.refusal and retrieval_stats.similarity_max >= 0.25):
            failure_reasons.append(FailureReason.WRONG_REFUSAL)
            failure_categories.append(FailureCategory.POLICY_FAILURE)
        if structured_output.refusal and structured_output.refusal_reason == RefusalReason.CONFLICTING_EVIDENCE:
            failure_reasons.append(FailureReason.CONFLICTING_EVIDENCE)
            failure_categories.append(FailureCategory.POLICY_FAILURE)
        if any(flag == SuspiciousFlag.POSSIBLE_POLICY_MISMATCH for flag in suspicious_flags):
            failure_categories.append(FailureCategory.POLICY_FAILURE)
        return self._dedupe(failure_reasons), self._dedupe(failure_categories)

    def _dedupe(self, values):
        return list(dict.fromkeys(values))
