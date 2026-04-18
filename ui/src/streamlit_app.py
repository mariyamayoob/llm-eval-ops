import json
from urllib import parse, request

import streamlit as st

st.set_page_config(page_title="LLM Eval Ops", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
BASE_PATH = "/policy-desk-assistant"
page = st.sidebar.radio("Page", ["Inference", "Run Explorer", "Offline Gates", "Review Queue"])
PROMPT_OPTIONS = {
    "Stable baseline (v1)": "qa-prompt:v1",
    "New candidate (v2)": "qa-prompt:v2",
}
PROMPT_LABELS = tuple(PROMPT_OPTIONS.keys())
PROMPT_LABEL_BY_VERSION = {value: label for label, value in PROMPT_OPTIONS.items()}
CASE_SET_OPTIONS = {
    "Core release suite": "portable",
    "Stress demo suite": "full",
}
DEFAULT_RETRIEVAL_CONFIG_VERSION = "retrieval-config:v1"
DEFAULT_SOURCE_SNAPSHOT_ID = "kb-snapshot:current"


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


def offline_score_total(score_breakdown: dict) -> float:
    if "total" in score_breakdown:
        return float(score_breakdown["total"])
    numeric_values = [
        float(value)
        for key, value in score_breakdown.items()
        if key != "total" and isinstance(value, int | float)
    ]
    if not numeric_values:
        return 0.0
    return round(sum(numeric_values) / len(numeric_values), 4)


def format_pct(value: float) -> str:
    return f"{round(float(value) * 100)}%"


def summary_case_count(summary: dict) -> int:
    return sum(int(bucket.get("case_count", 0)) for bucket in summary.get("by_bucket_breakdown", []))


def matched_behavior_count(summary: dict) -> int:
    return round(
        sum(
            float(bucket.get("behavior_match_rate", 0.0)) * int(bucket.get("case_count", 0))
            for bucket in summary.get("by_bucket_breakdown", [])
        )
    )


def passed_case_count(summary: dict) -> int:
    return round(
        sum(
            float(bucket.get("pass_rate", 0.0)) * int(bucket.get("case_count", 0))
            for bucket in summary.get("by_bucket_breakdown", [])
        )
    )


def flatten_case_rows(case_rows: list[dict]) -> list[dict]:
    flattened = []
    for row in case_rows:
        flattened.append(
            {
                "case_id": row["case_id"],
                "bucket": row["bucket_id"],
                "risk_tier": row["risk_tier"],
                "passed": row["passed"],
                "expected_behavior": row["expected_behavior"],
                "actual_behavior": row["actual_behavior"],
                "weighted_score": row["weighted_score"],
                "score": offline_score_total(row["score_breakdown"]),
                "blockers": ", ".join(row["regression_blockers"]),
                "preview": row["answer_preview"],
            }
        )
    return flattened


def inference_display_text(payload: dict) -> str:
    return (
        payload.get("answer")
        or payload.get("provisional_answer")
        or payload.get("missing_or_conflicting_evidence_summary")
        or "No response text returned."
    )


def render_inference_result(payload: dict):
    st.success(payload["outcome"])
    st.write(inference_display_text(payload))
    if payload.get("citations"):
        st.caption("Citations: " + ", ".join(f"`{citation}`" for citation in payload["citations"]))
    else:
        st.caption("No citations attached to this response.")
    with st.expander("Raw JSON response"):
        st.json(payload)


def render_release_banner(summary: dict):
    st.metric("Release decision", summary["release_decision"])
    if summary["decision_reasons"]:
        st.write("Reasons")
        for reason in summary["decision_reasons"]:
            st.write(f"- {reason}")


def core_metric_view(summary: dict) -> dict:
    metrics = summary["aggregate_metrics"]
    return {
        "valid_grounded_score": metrics.get("valid_grounded_score", 0.0),
        "behavior_score": metrics.get("behavior_score", metrics.get("behavior_match_rate", 0.0)),
        "pass_rate": metrics.get("pass_rate", 0.0),
        "weighted_overall": metrics.get("weighted_overall", 0.0),
    }


def advisory_metric_view(summary: dict) -> dict:
    metrics = summary["aggregate_metrics"]
    return {
        "advisory_quality_score": metrics.get("advisory_quality_score", 0.0),
        "brand_voice_match": metrics.get("brand_voice_match", 0.0),
        "tone_match": metrics.get("tone_match", 0.0),
    }


def render_story_cards(summary: dict):
    metrics = core_metric_view(summary)
    total_cases = summary_case_count(summary)
    behavior_matches = matched_behavior_count(summary)
    passed_cases = passed_case_count(summary)
    columns = st.columns(3)
    columns[0].metric("Output validity & support", format_pct(metrics["valid_grounded_score"]))
    columns[0].caption("Schema, citations, retrieval support, and answer checks.")
    columns[1].metric("Expected response choice", format_pct(metrics["behavior_score"]))
    columns[1].caption(
        f"{behavior_matches}/{total_cases} cases matched answer vs abstain/refuse/escalate."
        if total_cases
        else "Answer vs abstain/refuse/escalate."
    )
    columns[2].metric("Gate pass rate", format_pct(metrics["pass_rate"]))
    columns[2].caption(
        f"{passed_cases}/{total_cases} cases cleared thresholds with no blocking issues."
        if total_cases
        else "Cleared thresholds with no blocking issues."
    )
    st.caption("Core release checks focus on valid structure, evidence support, and choosing the right response mode. Tone and brand stay advisory.")


def visible_eval_config(config: dict) -> dict:
    visible = {
        key: config[key]
        for key in ["label", "model_backend", "prompt_version", "case_set"]
        if key in config and config[key] is not None
    }
    if "prompt_version" in visible:
        visible["prompt_version"] = f"{prompt_label_for(str(visible['prompt_version']))} [{visible['prompt_version']}]"
    return visible


def prompt_label_for(prompt_version: str) -> str:
    return PROMPT_LABEL_BY_VERSION.get(prompt_version, prompt_version)


if page == "Inference":
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
    if st.button("Submit"):
        try:
            payload = api_post(f"{BASE_PATH}/respond", {
                "question": question,
                "model_backend": model_backend,
                "prompt_version": prompt_version,
                "retrieval_config_version": DEFAULT_RETRIEVAL_CONFIG_VERSION,
                "source_snapshot_id": DEFAULT_SOURCE_SNAPSHOT_ID,
                "scenario": scenario,
            })
            render_inference_result(payload)
        except Exception as exc:
            st.error(str(exc))

elif page == "Run Explorer":
    st.title("Run Explorer")
    st.caption("Inspect stored runs, core trust checks, and any advisory or review signals.")
    runs = api_get(f"{BASE_PATH}/runs")
    st.dataframe(runs)
    run_id = st.text_input("Run ID")
    if run_id:
        st.json(api_get(f"{BASE_PATH}/runs/{run_id}"))

elif page == "Offline Gates":
    st.title("Offline Regression Gates")
    st.caption("Three questions: can the system return a valid supported response, did it choose the right action, and did the candidate regress?")

    st.subheader("Single Run")
    model_backend = st.selectbox("Eval backend", ["mock", "openai"], key="single_backend")
    prompt_label = st.selectbox("Eval prompt", PROMPT_LABELS, key="single_prompt")
    prompt_version = PROMPT_OPTIONS[prompt_label]
    single_case_set_label = st.selectbox("Case set", list(CASE_SET_OPTIONS.keys()), key="single_case_set")
    single_case_set = CASE_SET_OPTIONS[single_case_set_label]

    if st.button("Run offline eval"):
        query = parse.urlencode(
            {
                "model_backend": model_backend,
                "prompt_version": prompt_version,
                "retrieval_config_version": DEFAULT_RETRIEVAL_CONFIG_VERSION,
                "source_snapshot_id": DEFAULT_SOURCE_SNAPSHOT_ID,
                "case_set": single_case_set,
            }
        )
        payload = api_get(f"{BASE_PATH}/evals/offline?{query}")
        summary = payload["summary"]
        st.write("Core story")
        render_story_cards(summary)
        render_release_banner(summary)
        st.write("Core gate metrics")
        st.json(core_metric_view(summary))
        st.write("Advisory quality signals")
        st.json(advisory_metric_view(summary))
        st.write("Bucket gates")
        st.dataframe(summary["by_bucket_breakdown"])
        st.write("Worst cases")
        worst_rows = [row for row in flatten_case_rows(payload["case_results"]) if row["case_id"] in summary["worst_case_ids"]]
        st.dataframe(worst_rows)
        st.write("All cases")
        st.dataframe(flatten_case_rows(payload["case_results"]))

    st.subheader("Baseline Vs Candidate")
    baseline_col, candidate_col = st.columns(2)
    with baseline_col:
        st.caption("Baseline")
        baseline_backend = st.selectbox("Backend", ["mock", "openai"], key="baseline_backend")
        baseline_prompt_label = st.selectbox("Prompt", PROMPT_LABELS, key="baseline_prompt")
        baseline_prompt = PROMPT_OPTIONS[baseline_prompt_label]
    with candidate_col:
        st.caption("Candidate")
        candidate_backend = st.selectbox("Backend", ["mock", "openai"], key="candidate_backend")
        candidate_prompt_label = st.selectbox("Prompt", PROMPT_LABELS, index=1, key="candidate_prompt")
        candidate_prompt = PROMPT_OPTIONS[candidate_prompt_label]
    compare_case_set_label = st.selectbox("Case set", list(CASE_SET_OPTIONS.keys()), key="compare_case_set")
    compare_case_set = CASE_SET_OPTIONS[compare_case_set_label]

    if st.button("Compare offline evals"):
        payload = api_post(
            f"{BASE_PATH}/evals/offline/compare",
            {
                "baseline": {
                    "label": "baseline",
                    "model_backend": baseline_backend,
                    "prompt_version": baseline_prompt,
                    "retrieval_config_version": DEFAULT_RETRIEVAL_CONFIG_VERSION,
                    "source_snapshot_id": DEFAULT_SOURCE_SNAPSHOT_ID,
                    "case_set": compare_case_set,
                },
                "candidate": {
                    "label": "candidate",
                    "model_backend": candidate_backend,
                    "prompt_version": candidate_prompt,
                    "retrieval_config_version": DEFAULT_RETRIEVAL_CONFIG_VERSION,
                    "source_snapshot_id": DEFAULT_SOURCE_SNAPSHOT_ID,
                    "case_set": compare_case_set,
                },
            },
        )
        summary = payload["summary"]
        st.write("Regression story")
        story_columns = st.columns(4)
        story_columns[0].metric("Release decision", summary["release_decision"])
        story_columns[1].metric("New failures", len(summary["new_failures"]))
        story_columns[2].metric("Fixed failures", len(summary["fixed_failures"]))
        story_columns[3].metric("New blockers", len(summary["new_blocking_failures"]))
        render_release_banner(summary)

        metric_left, metric_right = st.columns(2)
        with metric_left:
            st.write("Baseline checks")
            render_story_cards(summary["baseline_summary"])
            st.json(
                {
                    "config": visible_eval_config(summary["baseline_config"]),
                    "core_metrics": core_metric_view(summary["baseline_summary"]),
                    "advisory_signals": advisory_metric_view(summary["baseline_summary"]),
                    "release_decision": summary["baseline_summary"]["release_decision"],
                }
            )
        with metric_right:
            st.write("Candidate checks")
            render_story_cards(summary["candidate_summary"])
            st.json(
                {
                    "config": visible_eval_config(summary["candidate_config"]),
                    "core_metrics": core_metric_view(summary["candidate_summary"]),
                    "advisory_signals": advisory_metric_view(summary["candidate_summary"]),
                    "release_decision": summary["candidate_summary"]["release_decision"],
                }
            )

        st.write("Core deltas")
        st.json(
            {
                key: summary["aggregate_deltas"].get(key, 0.0)
                for key in ["valid_grounded_score", "behavior_score", "pass_rate", "weighted_overall", "skipped_case_delta"]
            }
        )

        st.write("Bucket regressions")
        st.dataframe(summary["bucket_deltas"])

        new_failures = [row for row in payload["case_deltas"] if row["new_failures"] or row["new_blocking_failures"]]
        st.write("New failures")
        st.dataframe(new_failures)

        worst_regressions = sorted(
            payload["case_deltas"],
            key=lambda row: row["weighted_score_delta"] if row["weighted_score_delta"] is not None else 999,
        )[:5]
        st.write("Worst regressions")
        st.dataframe(worst_regressions)

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
