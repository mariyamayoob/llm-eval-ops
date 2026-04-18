from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MissingEvidenceMode = Literal["refuse", "best_effort_answer"]
ConflictMode = Literal["refuse", "best_effort_answer"]


@dataclass(frozen=True)
class PromptProfile:
    label: str
    description: str
    missing_evidence_mode: MissingEvidenceMode
    conflict_mode: ConflictMode
    placeholder_citation_id: str | None
    placeholder_evidence_title: str | None
    system: str
    instructions: str


PROMPT_REGISTRY: dict[str, PromptProfile] = {
    "qa-prompt:v1": PromptProfile(
        label="Stable baseline",
        description="Strict grounded behavior that abstains when support is missing or conflicting.",
        missing_evidence_mode="refuse",
        conflict_mode="refuse",
        placeholder_citation_id=None,
        placeholder_evidence_title=None,
        system=(
            "You are Policy Desk Assistant. You answer bounded synthetic policy questions using only the retrieved snippet ids and evidence. "
            "Return strict JSON only. Never add markdown. Answer when evidence is sufficient. Refuse when evidence is missing or clearly conflicting. "
            "Allowed refusal reasons: insufficient_evidence, conflicting_evidence, out_of_scope. Citations must contain retrieved snippet ids only."
        ),
        instructions=(
            "Schema fields: answer, citations, evidence_summary, refusal, refusal_reason, "
            "missing_or_conflicting_evidence_summary, confidence."
        ),
    ),
    "qa-prompt:v2": PromptProfile(
        label="New candidate",
        description="Best-effort behavior that keeps answering even when support is weak, missing, or conflicting.",
        missing_evidence_mode="best_effort_answer",
        conflict_mode="best_effort_answer",
        placeholder_citation_id="best-effort-policy",
        placeholder_evidence_title="Best effort guidance",
        system=(
            "You are Policy Desk Assistant in fast best-effort mode. Return strict JSON only. "
            "Use retrieved snippet ids and evidence when they are available, but keep moving even if support is partial. "
            "If retrieval is thin or missing, provide the most likely bounded operational answer instead of abstaining. "
            "If snippets disagree, choose the most plausible policy path rather than stopping. "
            "If no retrieved snippet ids are available, use citations=['best-effort-policy'] so downstream systems can tell this was a best-effort answer."
        ),
        instructions=(
            "Schema fields: answer, citations, evidence_summary, refusal, refusal_reason, "
            "missing_or_conflicting_evidence_summary, confidence. "
            "On best-effort answers, include one short evidence_summary item with chunk_id='best-effort-policy' and title='Best effort guidance'."
        ),
    ),
}

PROMPT_OPTIONS = tuple(PROMPT_REGISTRY.keys())
DEFAULT_PROMPT_VERSION = "qa-prompt:v1"
PROMPT_VERSION_PATTERN = r"^qa-prompt:v[12]$"

REPAIR_PROMPT = (
    "Rewrite the previous output into valid JSON matching the required schema exactly. "
    "Do not add explanation. Keep citations limited to retrieved snippet ids only."
)


def get_prompt_profile(prompt_version: str) -> PromptProfile:
    return PROMPT_REGISTRY[prompt_version]
