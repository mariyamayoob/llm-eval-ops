from __future__ import annotations

from urllib import parse

import streamlit as st

from ui_api import ApiClient
from ui_config import (
    BASE_PATH,
    CASE_SET_OPTIONS,
    DEFAULT_RETRIEVAL_CONFIG_VERSION,
    DEFAULT_SOURCE_SNAPSHOT_ID,
    PROMPT_LABELS,
    PROMPT_OPTIONS,
    prompt_label_for,
)
from ui_utils import format_pct, offline_score_total


def _summary_case_count(summary: dict) -> int:
    return sum(int(bucket.get("case_count", 0)) for bucket in summary.get("by_bucket_breakdown", []))


def _matched_behavior_count(summary: dict) -> int:
    return round(
        sum(
            float(bucket.get("behavior_match_rate", 0.0)) * int(bucket.get("case_count", 0))
            for bucket in summary.get("by_bucket_breakdown", [])
        )
    )


def _passed_case_count(summary: dict) -> int:
    return round(
        sum(
            float(bucket.get("pass_rate", 0.0)) * int(bucket.get("case_count", 0))
            for bucket in summary.get("by_bucket_breakdown", [])
        )
    )


def _flatten_case_rows(case_rows: list[dict]) -> list[dict]:
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


def _render_release_banner(summary: dict):
    st.metric("Release decision", summary["release_decision"])
    if summary["decision_reasons"]:
        st.write("Reasons")
        for reason in summary["decision_reasons"]:
            st.write(f"- {reason}")


def _core_metric_view(summary: dict) -> dict:
    metrics = summary["aggregate_metrics"]
    return {
        "valid_grounded_score": metrics.get("valid_grounded_score", 0.0),
        "behavior_score": metrics.get("behavior_score", metrics.get("behavior_match_rate", 0.0)),
        "pass_rate": metrics.get("pass_rate", 0.0),
        "weighted_overall": metrics.get("weighted_overall", 0.0),
    }


def _advisory_metric_view(summary: dict) -> dict:
    metrics = summary["aggregate_metrics"]
    return {
        "advisory_quality_score": metrics.get("advisory_quality_score", 0.0),
        "brand_voice_match": metrics.get("brand_voice_match", 0.0),
        "tone_match": metrics.get("tone_match", 0.0),
    }


def _render_story_cards(summary: dict):
    metrics = _core_metric_view(summary)
    total_cases = _summary_case_count(summary)
    behavior_matches = _matched_behavior_count(summary)
    passed_cases = _passed_case_count(summary)
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
    st.caption(
        "Core release checks focus on valid structure, evidence support, and choosing the right response mode. Tone and brand stay advisory."
    )


def _visible_eval_config(config: dict) -> dict:
    visible = {key: config[key] for key in ["label", "model_backend", "prompt_version", "case_set"] if key in config and config[key] is not None}
    if "prompt_version" in visible:
        visible["prompt_version"] = f"{prompt_label_for(str(visible['prompt_version']))} [{visible['prompt_version']}]"
    return visible


def render(api: ApiClient):
    st.title("Offline Regression Gates")
    st.caption("Three questions: can the system return a valid supported response, did it choose the right action, and did the candidate regress?")

    st.subheader("Single Run")
    model_backend = st.selectbox("Eval backend", ["mock", "openai"], key="single_backend")
    prompt_label = st.selectbox("Eval prompt", PROMPT_LABELS, key="single_prompt")
    prompt_version = PROMPT_OPTIONS[prompt_label]
    single_case_set_label = st.selectbox("Case set", list(CASE_SET_OPTIONS.keys()), key="single_case_set")
    single_case_set = CASE_SET_OPTIONS[single_case_set_label]

    if st.button("Run offline eval"):
        try:
            query = parse.urlencode(
                {
                    "model_backend": model_backend,
                    "prompt_version": prompt_version,
                    "retrieval_config_version": DEFAULT_RETRIEVAL_CONFIG_VERSION,
                    "source_snapshot_id": DEFAULT_SOURCE_SNAPSHOT_ID,
                    "case_set": single_case_set,
                }
            )
            payload = api.get(f"{BASE_PATH}/evals/offline?{query}")
            summary = payload["summary"]
            st.write("Core story")
            _render_story_cards(summary)
            _render_release_banner(summary)

            st.subheader("Case results")
            st.dataframe(_flatten_case_rows(payload["case_results"]))
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Compare Two Runs")
    baseline_backend = st.selectbox("Baseline backend", ["mock", "openai"], key="baseline_backend")
    baseline_prompt_label = st.selectbox("Baseline prompt", PROMPT_LABELS, key="baseline_prompt")
    baseline_prompt_version = PROMPT_OPTIONS[baseline_prompt_label]
    candidate_backend = st.selectbox("Candidate backend", ["mock", "openai"], key="candidate_backend")
    candidate_prompt_label = st.selectbox("Candidate prompt", PROMPT_LABELS, key="candidate_prompt")
    candidate_prompt_version = PROMPT_OPTIONS[candidate_prompt_label]
    compare_case_set_label = st.selectbox("Compare case set", list(CASE_SET_OPTIONS.keys()), key="compare_case_set")
    compare_case_set = CASE_SET_OPTIONS[compare_case_set_label]

    if st.button("Run comparison"):
        try:
            payload = api.post(
                f"{BASE_PATH}/evals/offline/compare",
                {
                    "baseline": {
                        "label": "baseline",
                        "model_backend": baseline_backend,
                        "prompt_version": baseline_prompt_version,
                        "case_set": compare_case_set,
                        "retrieval_config_version": DEFAULT_RETRIEVAL_CONFIG_VERSION,
                        "source_snapshot_id": DEFAULT_SOURCE_SNAPSHOT_ID,
                    },
                    "candidate": {
                        "label": "candidate",
                        "model_backend": candidate_backend,
                        "prompt_version": candidate_prompt_version,
                        "case_set": compare_case_set,
                        "retrieval_config_version": DEFAULT_RETRIEVAL_CONFIG_VERSION,
                        "source_snapshot_id": DEFAULT_SOURCE_SNAPSHOT_ID,
                    },
                },
            )
            summary = payload["summary"]

            st.write("Core story")
            story_columns = st.columns(4)
            story_columns[0].metric("Gate pass delta", format_pct(summary["aggregate_deltas"].get("pass_rate", 0.0)))
            story_columns[1].metric("Behavior delta", format_pct(summary["aggregate_deltas"].get("behavior_score", 0.0)))
            story_columns[2].metric("Validity delta", format_pct(summary["aggregate_deltas"].get("valid_grounded_score", 0.0)))
            story_columns[3].metric("New blockers", len(summary["new_blocking_failures"]))
            _render_release_banner(summary)

            metric_left, metric_right = st.columns(2)
            with metric_left:
                st.write("Baseline checks")
                _render_story_cards(summary["baseline_summary"])
                st.json(
                    {
                        "config": _visible_eval_config(summary["baseline_config"]),
                        "core_metrics": _core_metric_view(summary["baseline_summary"]),
                        "advisory_signals": _advisory_metric_view(summary["baseline_summary"]),
                        "release_decision": summary["baseline_summary"]["release_decision"],
                    }
                )
            with metric_right:
                st.write("Candidate checks")
                _render_story_cards(summary["candidate_summary"])
                st.json(
                    {
                        "config": _visible_eval_config(summary["candidate_config"]),
                        "core_metrics": _core_metric_view(summary["candidate_summary"]),
                        "advisory_signals": _advisory_metric_view(summary["candidate_summary"]),
                        "release_decision": summary["candidate_summary"]["release_decision"],
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
        except Exception as exc:
            st.error(str(exc))
