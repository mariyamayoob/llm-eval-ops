from __future__ import annotations

import streamlit as st

from ui_api import ApiClient
from ui_config import (
    BASE_PATH,
    DEFAULT_RETRIEVAL_CONFIG_VERSION,
    DEFAULT_SOURCE_SNAPSHOT_ID,
    PROMPT_LABELS,
    PROMPT_OPTIONS,
    THUMBS_DOWN_EVENT_TYPE,
    THUMBS_UP_EVENT_TYPE,
)
from ui_utils import inference_display_text


def _render_inference_result(payload: dict):
    st.success(payload["outcome"])
    st.write(inference_display_text(payload))
    if payload.get("citations"):
        st.caption("Citations: " + ", ".join(f"`{citation}`" for citation in payload["citations"]))
    else:
        st.caption("No citations attached to this response.")
    with st.expander("Raw JSON response"):
        st.json(payload)


def _record_inference_feedback(api: ApiClient, run_id: str, event_type: str):
    api.post(
        f"{BASE_PATH}/feedback",
        {"run_id": run_id, "event_type": event_type, "session_id": None, "event_value": None},
    )
    feedback_by_run = st.session_state.setdefault("inference_feedback_by_run", {})
    feedback_by_run[run_id] = event_type
    st.session_state["inference_feedback_message"] = event_type


def _render_inference_feedback_controls(api: ApiClient, payload: dict):
    run_id = payload["run_id"]
    feedback_by_run = st.session_state.setdefault("inference_feedback_by_run", {})
    recorded_feedback = feedback_by_run.get(run_id)

    st.caption(f"Run id: `{run_id}`")
    if recorded_feedback == THUMBS_UP_EVENT_TYPE:
        st.success("Thumbs up recorded for this run.")
        return
    if recorded_feedback == THUMBS_DOWN_EVENT_TYPE:
        st.warning("Thumbs down recorded for this run.")
        return

    st.caption("Quick operator feedback")
    feedback_cols = st.columns(2)
    if feedback_cols[0].button("Thumbs up", key=f"thumbs_up_{run_id}"):
        try:
            _record_inference_feedback(api, run_id, THUMBS_UP_EVENT_TYPE)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if feedback_cols[1].button("Thumbs down", key=f"thumbs_down_{run_id}"):
        try:
            _record_inference_feedback(api, run_id, THUMBS_DOWN_EVENT_TYPE)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render(api: ApiClient):
    st.title("LLM Eval Ops")
    st.caption("Ask one question and inspect whether the response is valid, grounded, and trusted enough to serve.")

    question = st.text_area("Question", value="How long does a customer have to request a refund?")
    model_backend = st.selectbox("Model backend", ["mock", "openai"])
    prompt_label = st.selectbox("Prompt profile", PROMPT_LABELS)
    prompt_version = PROMPT_OPTIONS[prompt_label]
    scenario = st.selectbox(
        "Scenario",
        ["normal", "retrieval_miss", "malformed_json", "unsupported_answer", "wrong_refusal", "slow_response", "conflicting_evidence"],
    )

    payload = st.session_state.get("last_inference_payload")
    if st.button("Submit"):
        try:
            if not question.strip():
                raise ValueError("Question is required.")
            payload = api.post(
                f"{BASE_PATH}/respond",
                {
                    "question": question,
                    "model_backend": model_backend,
                    "prompt_version": prompt_version,
                    "retrieval_config_version": DEFAULT_RETRIEVAL_CONFIG_VERSION,
                    "source_snapshot_id": DEFAULT_SOURCE_SNAPSHOT_ID,
                    "scenario": scenario,
                },
            )
            st.session_state["last_inference_payload"] = payload
            st.session_state["inference_feedback_message"] = None
        except Exception as exc:
            st.error(str(exc))

    if payload:
        _render_inference_result(payload)
        _render_inference_feedback_controls(api, payload)
