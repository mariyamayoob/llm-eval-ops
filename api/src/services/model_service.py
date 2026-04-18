from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol

from model.contracts import (
    ModelBackend,
    RefusalReason,
    ScenarioName,
    StructuredPolicyOutput,
)
from prompts.registry import PromptProfile, REPAIR_PROMPT, get_prompt_profile


@dataclass
class ModelCallResult:
    backend: ModelBackend
    model_name: str
    raw_text: str


class ModelAdapter(Protocol):
    backend: ModelBackend
    model_name: str

    def generate(self, *, question: str, retrieved_chunks: list[dict[str, object]], prompt_version: str, scenario: ScenarioName) -> str:
        ...

    def repair(self, raw_text: str) -> str:
        ...


class MockModelAdapter:
    backend = ModelBackend.MOCK
    model_name = "mock-policy-model"

    def generate(self, *, question: str, retrieved_chunks: list[dict[str, object]], prompt_version: str, scenario: ScenarioName) -> str:
        prompt = get_prompt_profile(prompt_version)
        if scenario == ScenarioName.SLOW_RESPONSE:
            time.sleep(0.05)
        if scenario == ScenarioName.MALFORMED_JSON:
            return "{'answer': 'Unused service credits expire 45 days after they are issued.', 'citations': ['policy-credit-expiry-45'], 'evidence_summary': [{'chunk_id': 'policy-credit-expiry-45', 'title': 'Credit expiry', 'support_snippet': 'Credits expire 45 days after issue.', 'relevance_score': 0.91}], 'refusal': false, 'refusal_reason': null, 'missing_or_conflicting_evidence_summary': null, 'confidence': 0.87}"
        if scenario == ScenarioName.RETRIEVAL_MISS:
            if prompt.missing_evidence_mode == "best_effort_answer":
                return self._best_effort_missing_evidence_response(prompt)
            return json.dumps(
                {
                    "answer": "",
                    "citations": [],
                    "evidence_summary": [],
                    "refusal": True,
                    "refusal_reason": RefusalReason.INSUFFICIENT_EVIDENCE.value,
                    "missing_or_conflicting_evidence_summary": "Retrieved evidence did not contain policy support for the request.",
                    "confidence": 0.28,
                }
            )
        if scenario == ScenarioName.CONFLICTING_EVIDENCE:
            if prompt.conflict_mode == "best_effort_answer":
                return self._best_effort_conflict_response(prompt, retrieved_chunks)
            return json.dumps(
                {
                    "answer": "",
                    "citations": [],
                    "evidence_summary": [],
                    "refusal": True,
                    "refusal_reason": RefusalReason.CONFLICTING_EVIDENCE.value,
                    "missing_or_conflicting_evidence_summary": "Retrieved policy snippets indicate both a standard refund window and a documented exception path.",
                    "confidence": 0.46,
                }
            )
        if scenario == ScenarioName.WRONG_REFUSAL:
            return json.dumps(
                {
                    "answer": "",
                    "citations": [],
                    "evidence_summary": [],
                    "refusal": True,
                    "refusal_reason": RefusalReason.INSUFFICIENT_EVIDENCE.value,
                    "missing_or_conflicting_evidence_summary": "The assistant claims evidence is insufficient.",
                    "confidence": 0.82,
                }
            )
        citation_id = retrieved_chunks[0]["id"] if retrieved_chunks else ""
        evidence = self._evidence_summary(retrieved_chunks)
        answer = self._answer_for(question, citation_id)
        if scenario == ScenarioName.UNSUPPORTED_ANSWER:
            answer = "Customers can request a refund within 90 days of purchase."
            citation_id = "policy-refund-30-standard"
            evidence = self._evidence_summary([chunk for chunk in retrieved_chunks if chunk["id"] == citation_id])
        payload = {
            "answer": answer,
            "citations": [citation_id] if citation_id else [],
            "evidence_summary": evidence,
            "refusal": False,
            "refusal_reason": None,
            "missing_or_conflicting_evidence_summary": None,
            "confidence": 0.92,
        }
        return json.dumps(payload)

    def repair(self, raw_text: str) -> str:
        return raw_text.replace("'", '"').replace(": false", ": false").replace(": null", ": null")

    def _evidence_summary(self, retrieved_chunks: list[dict[str, object]]) -> list[dict[str, object]]:
        items = []
        for chunk in retrieved_chunks[:2]:
            items.append(
                {
                    "chunk_id": chunk["id"],
                    "title": chunk["title"],
                    "support_snippet": str(chunk["body"])[:120],
                    "relevance_score": round(float(chunk.get("score", 0.85)), 2),
                }
            )
        return items

    def _answer_for(self, question: str, citation_id: str) -> str:
        mapping = {
            "policy-refund-30-standard": "The standard refund window is 30 days from the purchase date.",
            "policy-cancellation-immediate": "Cancellation becomes effective immediately after confirmation and stops future billing cycles.",
            "policy-password-reset-24h": "Password reset changes can take up to 24 hours to propagate across all access channels.",
            "policy-document-deadline-10": "Supporting documents must be submitted within 10 calendar days of the request.",
            "policy-travel-reimbursement-60": "Travel reimbursement requests must be submitted within 60 days of trip completion.",
            "policy-overtime-approval-pre": "Overtime must be pre-approved by a manager before hours are worked.",
            "policy-credit-expiry-45": "Unused service credits expire 45 days after they are issued.",
            "policy-escalation-sensitive-2h": "Sensitive account changes should be escalated to the policy desk within 2 hours.",
            "policy-receipts-required-25": "Receipts are required for reimbursement claims of 25 dollars or more.",
            "policy-vpn-resync": "Support should clear cached credentials, confirm multifactor enrollment, and retry from the managed VPN client.",
        }
        return mapping.get(citation_id, "The retrieved policy supports a bounded answer to the question.")

    def _best_effort_missing_evidence_response(self, prompt: PromptProfile) -> str:
        citation_id = prompt.placeholder_citation_id or "best-effort-policy"
        title = prompt.placeholder_evidence_title or "Best effort guidance"
        return json.dumps(
            {
                "answer": "Pet insurance reimbursement is usually handled through a standard claims review path.",
                "citations": [citation_id],
                "evidence_summary": [
                    {
                        "chunk_id": citation_id,
                        "title": title,
                        "support_snippet": "No retrieved snippets were available, so this answer is being provided as best effort.",
                        "relevance_score": 0.21,
                    }
                ],
                "refusal": False,
                "refusal_reason": None,
                "missing_or_conflicting_evidence_summary": None,
                "confidence": 0.84,
            }
        )

    def _best_effort_conflict_response(self, prompt: PromptProfile, retrieved_chunks: list[dict[str, object]]) -> str:
        citation_id = str(retrieved_chunks[0]["id"]) if retrieved_chunks else (prompt.placeholder_citation_id or "best-effort-policy")
        evidence = self._evidence_summary(retrieved_chunks) if retrieved_chunks else [
            {
                "chunk_id": prompt.placeholder_citation_id or "best-effort-policy",
                "title": prompt.placeholder_evidence_title or "Best effort guidance",
                "support_snippet": "Conflicting evidence was present, but the assistant still chose a most likely policy path.",
                "relevance_score": 0.22,
            }
        ]
        return json.dumps(
            {
                "answer": "The most likely answer is the standard 30-day refund window.",
                "citations": [citation_id],
                "evidence_summary": evidence,
                "refusal": False,
                "refusal_reason": None,
                "missing_or_conflicting_evidence_summary": None,
                "confidence": 0.8,
            }
        )


class OpenAIModelAdapter:
    backend = ModelBackend.OPENAI

    def __init__(self) -> None:
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_BACKEND=openai")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError("The openai package is required for the OpenAI backend.") from exc
        self._client = OpenAI(api_key=api_key)

    def generate(self, *, question: str, retrieved_chunks: list[dict[str, object]], prompt_version: str, scenario: ScenarioName) -> str:
        if scenario not in {ScenarioName.NORMAL, ScenarioName.RETRIEVAL_MISS, ScenarioName.CONFLICTING_EVIDENCE, ScenarioName.SLOW_RESPONSE}:
            raise ValueError("Selected scenario is supported only on the OpenAI backend for normal, retrieval_miss, conflicting_evidence, or slow_response.")
        if scenario == ScenarioName.SLOW_RESPONSE:
            time.sleep(0.05)
        schema = self._strict_json_schema(StructuredPolicyOutput.model_json_schema())
        prompt = get_prompt_profile(prompt_version)
        input_payload = {
            "question": question,
            "retrieved_chunks": [
                {"id": chunk["id"], "title": chunk["title"], "body": chunk["body"][:240]}
                for chunk in retrieved_chunks
            ],
            "prompt_profile": {
                "label": prompt.label,
                "description": prompt.description,
                "missing_evidence_mode": prompt.missing_evidence_mode,
                "conflict_mode": prompt.conflict_mode,
                "placeholder_citation_id": prompt.placeholder_citation_id,
                "placeholder_evidence_title": prompt.placeholder_evidence_title,
            },
            "instructions": prompt.instructions,
        }
        response = self._client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": json.dumps(input_payload)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "policy_desk_output",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise ValueError("OpenAI response did not contain output_text")
        return output_text

    def repair(self, raw_text: str) -> str:
        response = self._client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": REPAIR_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "policy_desk_output_repair",
                    "strict": True,
                    "schema": self._strict_json_schema(StructuredPolicyOutput.model_json_schema()),
                }
            },
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise ValueError("OpenAI repair response did not contain output_text")
        return output_text

    def _strict_json_schema(self, schema: dict[str, object]) -> dict[str, object]:
        def walk(node: object) -> object:
            if isinstance(node, dict):
                updated = {key: walk(value) for key, value in node.items()}
                if updated.get("type") == "object":
                    properties = updated.get("properties", {})
                    if isinstance(properties, dict):
                        updated["required"] = list(properties.keys())
                    updated["additionalProperties"] = False
                return updated
            if isinstance(node, list):
                return [walk(item) for item in node]
            return node

        return walk(schema)


class ModelService:
    def __init__(self) -> None:
        self._mock = MockModelAdapter()

    def call(
        self,
        *,
        model_backend: ModelBackend,
        question: str,
        retrieved_chunks: list[dict[str, object]],
        prompt_version: str,
        scenario: ScenarioName,
    ) -> ModelCallResult:
        adapter = self._select_adapter(model_backend)
        raw_text = adapter.generate(
            question=question,
            retrieved_chunks=retrieved_chunks,
            prompt_version=prompt_version,
            scenario=scenario,
        )
        return ModelCallResult(backend=adapter.backend, model_name=adapter.model_name, raw_text=raw_text)

    def repair(self, *, model_backend: ModelBackend, raw_text: str) -> str:
        adapter = self._select_adapter(model_backend)
        return adapter.repair(raw_text)

    def _select_adapter(self, model_backend: ModelBackend) -> ModelAdapter:
        if model_backend == ModelBackend.OPENAI:
            return OpenAIModelAdapter()
        return self._mock
