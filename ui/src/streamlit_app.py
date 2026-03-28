import json
from urllib import parse, request

import streamlit as st

st.set_page_config(page_title="LLM Eval Ops", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
BASE_PATH = "/policy-desk-assistant"
page = st.sidebar.radio("Page", ["Inference", "Run Explorer", "Offline Eval Summary", "Review Queue"])


def api_get(path: str):
    with request.urlopen(f"{API_BASE}{path}") as response:
        return json.loads(response.read().decode("utf-8"))


def api_post(path: str, payload: dict):
    req = request.Request(
        f"{API_BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


if page == "Inference":
    st.title("LLM Eval Ops")
    st.caption("Runtime inference with online scoring, suspicious flags, and review routing.")
    question = st.text_area("Question", value="How long does a customer have to request a refund?")
    model_backend = st.selectbox("Model backend", ["mock", "openai"])
    prompt_version = st.selectbox("Prompt version", ["qa-prompt:v1", "qa-prompt:v2"])
    scenario = st.selectbox(
        "Scenario",
        ["normal", "retrieval_miss", "malformed_json", "unsupported_answer", "wrong_refusal", "slow_response", "conflicting_evidence"],
    )
    if st.button("Submit"):
        try:
            payload = api_post(f"{BASE_PATH}/respond", {
                "question": question,
                "model_backend": model_backend,
                "prompt_version": prompt_version,
                "scenario": scenario,
            })
            st.success(payload["outcome"])
            st.json(payload)
        except Exception as exc:
            st.error(str(exc))

elif page == "Run Explorer":
    st.title("Run Explorer")
    st.caption("Inspect stored run records, retrieval behavior, scores, and failure signals.")
    runs = api_get(f"{BASE_PATH}/runs")
    st.dataframe(runs)
    run_id = st.text_input("Run ID")
    if run_id:
        st.json(api_get(f"{BASE_PATH}/runs/{run_id}"))

elif page == "Offline Eval Summary":
    st.title("Offline Eval Summary")
    st.caption("Replay labeled synthetic scenarios through the runtime pipeline and inspect aggregate results.")
    model_backend = st.selectbox("Eval backend", ["mock", "openai"], key="eval_backend")
    prompt_version = st.selectbox("Eval prompt", ["qa-prompt:v1", "qa-prompt:v2"], key="eval_prompt")
    if st.button("Run offline eval"):
        query = parse.urlencode({"model_backend": model_backend, "prompt_version": prompt_version})
        st.json(api_get(f"{BASE_PATH}/evals/offline?{query}"))

elif page == "Review Queue":
    st.title("Review Queue")
    st.caption("Inspect flagged runs and capture reviewer dispositions that can later inform new eval cases or prompt updates.")
    items = api_get(f"{BASE_PATH}/review-queue")
    st.dataframe(items)
    item_id = st.text_input("Review queue item id")
    reviewer = st.text_input("Reviewer")
    notes = st.text_area("Notes")
    review_status = st.selectbox("Review status", ["pending", "in_review", "resolved"])
    final_disposition = st.selectbox("Final disposition", ["approved", "corrected", "confirmed_refusal", "escalated", "rejected_response"])
    if st.button("Save annotation") and item_id and reviewer:
        st.json(api_post(f"{BASE_PATH}/review-queue/{item_id}/annotate", {
            "reviewer_label": reviewer,
            "reviewer_notes": notes,
            "review_status": review_status,
            "final_disposition": final_disposition,
        }))
