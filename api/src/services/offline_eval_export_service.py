from __future__ import annotations

import re
from datetime import UTC, datetime

from evals.contracts import EvalBehavior
from model.contracts import Outcome, ScenarioName
from services.run_service import RunService


class OfflineEvalExportService:
    def __init__(self, *, run_service: RunService) -> None:
        self.run_service = run_service

    def export_promoted_cases(self) -> list[dict]:
        items = self.run_service.list_review_queue()
        promoted = [
            item
            for item in items
            if item.review_status.value == "resolved" and item.promote_to_offline_eval
        ]

        cases: list[dict] = []
        for item in promoted:
            run = self.run_service.get_run(item.run_id)
            if run is None or not run.question:
                continue

            # Minimal, honest export: it creates a portable case skeleton that a developer
            # can refine (citations/facts/forbidden claims) after redaction.
            redacted_question = _redact_pii(run.question)
            redacted_reference_response_text = (
                _redact_pii(item.should_have_response_text) if item.should_have_response_text else None
            )
            case_id = f"review_{item.review_queue_item_id.replace('-', '')[:12]}"
            expected_behavior = EvalBehavior.HUMAN_REVIEW
            expected_should_refuse = False
            if item.should_have_outcome == Outcome.SUPPORTED_ANSWER:
                expected_behavior = EvalBehavior.ANSWER
                expected_should_refuse = False
            elif item.should_have_outcome == Outcome.REFUSED_MORE_EVIDENCE_NEEDED:
                expected_behavior = EvalBehavior.ABSTAIN
                expected_should_refuse = True
            elif item.should_have_outcome == Outcome.HUMAN_REVIEW_RECOMMENDED:
                expected_behavior = EvalBehavior.HUMAN_REVIEW
                expected_should_refuse = False

            cases.append(
                {
                    "case_id": case_id,
                    "question": redacted_question,
                    "scenario": run.scenario.value if isinstance(run.scenario, ScenarioName) else str(run.scenario),
                    "expected_should_refuse": expected_should_refuse,
                    "expected_behavior": expected_behavior.value,
                    "reference_response_text": redacted_reference_response_text,
                    "required_citation_ids": [],
                    "acceptable_answer_facts": [],
                    "forbidden_claims": [],
                    "tags": ["policy_boundary_cases", "from_human_review"],
                    "difficulty": "medium",
                    "bucket_id": "policy-boundary-cases",
                    "bucket_name": "Policy Boundary Cases",
                    "risk_tier": "medium",
                    "business_criticality": "medium",
                    "gate_group": "human_review",
                    "owner": "policy-ops",
                    "case_kind": "portable",
                    "supported_backends": ["mock", "openai"],
                    "label_notes": f"Exported {datetime.now(UTC).date().isoformat()} from review item {item.review_queue_item_id}. Review source: {item.review_source}. Final disposition: {item.final_disposition.value if item.final_disposition else 'unknown'}.",
                    "source_snapshot_id": run.source_snapshot_id,
                    "retrieval_config_version": run.retrieval_config_version,
                }
            )
        return cases


def _redact_pii(text: str) -> str:
    # Demo-grade redaction: useful for the reference loop, not a production DLP system.
    text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[REDACTED_EMAIL]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b\d{4,}\b", "[REDACTED_NUMBER]", text)
    return text
