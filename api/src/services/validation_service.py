from __future__ import annotations

import json
import re

from pydantic import ValidationError

from model.contracts import (
    FailureCategory,
    FailureReason,
    ModelBackend,
    RefusalReason,
    StructuredPolicyOutput,
    ValidationResult,
)
from services.model_service import ModelService


DURATION_RE = re.compile(r"(\d+)\s+(day|days|hour|hours)", re.IGNORECASE)


class ValidationService:
    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service

    def parse_and_validate(
        self,
        *,
        raw_output: str,
        model_backend: ModelBackend,
        retrieved_ids: list[str],
        cited_bodies: dict[str, str],
    ) -> tuple[StructuredPolicyOutput, ValidationResult]:
        repair_attempted = False
        failure_reasons: list[FailureReason] = []
        failure_categories: list[FailureCategory] = []

        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            failure_reasons.append(FailureReason.MALFORMED_JSON)
            failure_categories.append(FailureCategory.GENERATION_FAILURE)
            repair_attempted = True
            try:
                repaired = self.model_service.repair(model_backend=model_backend, raw_text=raw_output)
                payload = json.loads(repaired)
            except Exception:
                failure_reasons.append(FailureReason.REPAIR_EXHAUSTED)
                failure_categories.append(FailureCategory.VALIDATION_FAILURE)
                return self._fail_closed(RefusalReason.INSUFFICIENT_EVIDENCE), ValidationResult(
                    structure_valid=False,
                    citation_valid=False,
                    repair_attempted=repair_attempted,
                    failure_reasons=self._dedupe(failure_reasons),
                    failure_categories=self._dedupe(failure_categories),
                )

        try:
            parsed = StructuredPolicyOutput.model_validate(payload)
            structure_valid = True
        except ValidationError:
            failure_reasons.append(FailureReason.SCHEMA_INVALID)
            failure_categories.append(FailureCategory.VALIDATION_FAILURE)
            return self._fail_closed(RefusalReason.INSUFFICIENT_EVIDENCE), ValidationResult(
                structure_valid=False,
                citation_valid=False,
                repair_attempted=repair_attempted,
                failure_reasons=self._dedupe(failure_reasons),
                failure_categories=self._dedupe(failure_categories),
            )

        citation_valid = set(parsed.citations).issubset(set(retrieved_ids))
        if not citation_valid:
            failure_reasons.append(FailureReason.FABRICATED_CITATION)
            failure_categories.append(FailureCategory.VALIDATION_FAILURE)

        if not parsed.refusal and self._unsupported_answer(parsed.answer, parsed.citations, cited_bodies):
            failure_reasons.append(FailureReason.UNSUPPORTED_ANSWER)
            failure_categories.append(FailureCategory.POLICY_FAILURE)

        if parsed.refusal and parsed.refusal_reason is None:
            failure_reasons.append(FailureReason.MISSING_REFUSAL)
            failure_categories.append(FailureCategory.VALIDATION_FAILURE)

        validation = ValidationResult(
            structure_valid=structure_valid,
            citation_valid=citation_valid,
            repair_attempted=repair_attempted,
            failure_reasons=self._dedupe(failure_reasons),
            failure_categories=self._dedupe(failure_categories),
        )
        return parsed, validation

    def _unsupported_answer(self, answer: str, citations: list[str], cited_bodies: dict[str, str]) -> bool:
        answer_durations = {match.group(0).lower() for match in DURATION_RE.finditer(answer)}
        if not answer_durations:
            return False
        supported: set[str] = set()
        for citation in citations:
            supported.update(match.group(0).lower() for match in DURATION_RE.finditer(cited_bodies.get(citation, "")))
        return bool(answer_durations - supported)

    def _fail_closed(self, reason: RefusalReason) -> StructuredPolicyOutput:
        return StructuredPolicyOutput(
            answer="",
            citations=[],
            evidence_summary=[],
            refusal=True,
            refusal_reason=reason,
            missing_or_conflicting_evidence_summary="The assistant could not produce a trustworthy policy-backed answer.",
            confidence=0.0,
        )

    def _dedupe(self, items):
        return list(dict.fromkeys(items))
