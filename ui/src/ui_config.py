from __future__ import annotations

# Streamlit UI constants. Keep these business-readable and demo-friendly.

BASE_PATH = "/policy-desk-assistant"

PROMPT_OPTIONS: dict[str, str] = {
    "Stable baseline (v1)": "qa-prompt:v1",
    "New candidate (v2)": "qa-prompt:v2",
}
PROMPT_LABELS: tuple[str, ...] = tuple(PROMPT_OPTIONS.keys())
PROMPT_LABEL_BY_VERSION: dict[str, str] = {value: label for label, value in PROMPT_OPTIONS.items()}

CASE_SET_OPTIONS: dict[str, str] = {
    "Core release suite": "portable",
    "Stress demo suite": "full",
}

DEFAULT_RETRIEVAL_CONFIG_VERSION = "retrieval-config:v1"
DEFAULT_SOURCE_SNAPSHOT_ID = "kb-snapshot:current"

THUMBS_UP_EVENT_TYPE = "thumbs_up"
THUMBS_DOWN_EVENT_TYPE = "thumbs_down"


def prompt_label_for(prompt_version: str) -> str:
    return PROMPT_LABEL_BY_VERSION.get(prompt_version, prompt_version)

