from __future__ import annotations

from collections import Counter

import streamlit as st

from ui_api import ApiClient
from ui_config import BASE_PATH
from ui_utils import avg_value, format_pct, safe_rate


def _load_recent_run_details(api: ApiClient, limit: int) -> list[dict]:
    run_summaries = api.get(f"{BASE_PATH}/runs?limit={limit}")
    runs: list[dict] = []
    for run_summary in run_summaries:
        detail = api.get(f"{BASE_PATH}/runs/{run_summary['run_id']}")
        runs.append(detail["run"])
    return runs


def _summarize_judge_records(records: list[dict]) -> dict:
    completed = [record for record in records if record["status"] == "completed" and record.get("assessment")]
    queued = [record for record in records if record["status"] == "queued"]
    failed = [record for record in records if record["status"] == "failed"]
    escalated = [record for record in completed if record["assessment"]["human_review_recommended"]]
    return {
        "total_records": len(records),
        "completed_count": len(completed),
        "queued_count": len(queued),
        "failed_count": len(failed),
        "escalated_count": len(escalated),
        "avg_overall_score": avg_value([record["assessment"]["overall_score"] for record in completed]),
        "avg_supportedness_score": avg_value([record["assessment"]["supportedness_score"] for record in completed]),
        "avg_policy_alignment_score": avg_value([record["assessment"]["policy_alignment_score"] for record in completed]),
        "avg_response_mode_score": avg_value([record["assessment"]["response_mode_score"] for record in completed]),
    }


def _group_judge_result_rows(records: list[dict], group_key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record[group_key]), []).append(record)

    rows: list[dict] = []
    for key in sorted(grouped):
        items = grouped[key]
        completed = [item for item in items if item["status"] == "completed" and item.get("assessment")]
        escalated = [item for item in completed if item["assessment"]["human_review_recommended"]]
        rows.append(
            {
                "group": key,
                "total_records": len(items),
                "completed_count": len(completed),
                "escalated_count": len(escalated),
                "avg_overall_score": avg_value([item["assessment"]["overall_score"] for item in completed]),
            }
        )
    return rows


def _flatten_llm_judge_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        assessment = record.get("assessment") or {}
        rows.append(
            {
                "judge_id": record["judge_id"],
                "run_id": record["run_id"],
                "created_at": record["created_at"],
                "status": record["status"],
                "judge_model": record["judge_model"],
                "sampling_rate": record["sampling_rate"],
                "model_backend": record["model_backend"],
                "prompt_version": record["prompt_version"],
                "response_outcome": record["response_outcome"],
                "risk_band": record["risk_band"],
                "overall_score": assessment.get("overall_score"),
                "supportedness_score": assessment.get("supportedness_score"),
                "policy_alignment_score": assessment.get("policy_alignment_score"),
                "response_mode_score": assessment.get("response_mode_score"),
                "human_review_recommended": assessment.get("human_review_recommended"),
                "human_review_reason": assessment.get("human_review_reason"),
            }
        )
    return rows


def _flatten_feedback_rows(events: list[dict]) -> list[dict]:
    return [
        {
            "event_id": event["event_id"],
            "created_at": event["created_at"],
            "run_id": event["run_id"],
            "event_type": event["event_type"],
            "event_value": event.get("event_value"),
            "session_id": event.get("session_id"),
            "response_outcome": event["response_outcome"],
            "risk_band": event["risk_band"],
            "model_backend": event["model_backend"],
            "prompt_version": event["prompt_version"],
        }
        for event in events
    ]


def _online_metric_rows(summary: dict) -> list[dict]:
    return [
        {
            "window_start": summary["window_start"],
            "window_end": summary["window_end"],
            "total_runs": summary["total_runs"],
            "total_feedback_events": summary["total_feedback_events"],
            "avg_online_score_total": summary["avg_online_score_total"],
            "avg_groundedness_proxy_score": summary["avg_groundedness_proxy_score"],
            "avg_citation_validity_score": summary["avg_citation_validity_score"],
            "avg_policy_adherence_score": summary["avg_policy_adherence_score"],
            "avg_format_validity_score": summary["avg_format_validity_score"],
            "avg_retrieval_support_score": summary["avg_retrieval_support_score"],
            "avg_brand_voice_score": summary["avg_brand_voice_score"],
            "avg_tone_appropriateness_score": summary["avg_tone_appropriateness_score"],
            "review_recommended_rate": summary["review_recommended_rate"],
            "supported_answer_rate": summary["supported_answer_rate"],
            "invalid_output_rate": summary["invalid_output_rate"],
            "thumbs_up_rate": summary["thumbs_up_rate"],
            "thumbs_down_rate": summary["thumbs_down_rate"],
            "high_risk_run_count": summary["high_risk_run_count"],
        }
    ]


def _render_alert_panel(snapshot: dict):
    alerts = snapshot["alert_evaluation"]
    st.metric("Alert status", alerts["status"])
    st.metric("Rollback recommended", "yes" if alerts["rollback_recommended"] else "no")
    if alerts["triggered_alerts"]:
        st.write("Triggered alerts")
        st.dataframe(alerts["triggered_alerts"])
    else:
        st.caption("No alert thresholds are currently triggered.")
    if alerts["rollback_reasons"]:
        st.write("Rollback reasons")
        for reason in alerts["rollback_reasons"]:
            st.write(f"- {reason}")
    if snapshot.get("review_queue_item_ids"):
        st.write("Review queue items opened from this summary")
        for item_id in snapshot["review_queue_item_ids"]:
            st.write(f"- `{item_id}`")
    if snapshot.get("worst_run_ids"):
        st.write("Most affected runs")
        for run_id in snapshot["worst_run_ids"]:
            st.write(f"- `{run_id}` -> `{BASE_PATH}/runs/{run_id}`")


def render(api: ApiClient):
    st.title("Online Control")
    st.caption("One operator view for inline deterministic checks, human review pressure, and the async OpenAI judge layer.")
    st.info("Judge sampling policy: 10% of low-risk runs, 30% of medium-risk runs, and 100% of high-risk runs that were not already routed to human review.")

    try:
        snapshot = api.get(f"{BASE_PATH}/feedback/summary?limit=50")
        recent_feedback = api.get(f"{BASE_PATH}/feedback?limit=15")
        review_items = api.get(f"{BASE_PATH}/review-queue")
        judge_records = api.get(f"{BASE_PATH}/llm-judge?limit=50")
        recent_runs = _load_recent_run_details(api, 50)

        recent_run_ids = {run["run_id"] for run in recent_runs}
        recent_judge_records = [record for record in judge_records if record["run_id"] in recent_run_ids]
        judge_summary = _summarize_judge_records(recent_judge_records)
        judged_run_ids = {record["run_id"] for record in recent_judge_records}

        review_status_counts = Counter(item["review_status"] for item in review_items)
        review_source_counts = Counter(item["review_source"] for item in review_items)
        open_review_items = [item for item in review_items if item["review_status"] != "resolved"]

        overview_cards = st.columns(5)
        overview_cards[0].metric("Invalid output rate", format_pct(snapshot["summary"]["invalid_output_rate"]))
        overview_cards[1].metric("Runtime review rate", format_pct(snapshot["summary"]["review_recommended_rate"]))
        overview_cards[2].metric("Open human reviews", len(open_review_items))
        overview_cards[3].metric("Judge sample rate", format_pct(safe_rate(len(judged_run_ids), len(recent_run_ids))))
        overview_cards[4].metric("Alert status", snapshot["alert_evaluation"]["status"])

        second_cards = st.columns(4)
        second_cards[0].metric("Supported answer rate", format_pct(snapshot["summary"]["supported_answer_rate"]))
        second_cards[1].metric("Thumbs up rate", format_pct(snapshot["summary"]["thumbs_up_rate"]))
        second_cards[2].metric("Thumbs down rate", format_pct(snapshot["summary"]["thumbs_down_rate"]))
        second_cards[3].metric("Judge -> human", judge_summary["escalated_count"])

        runtime_tab, review_tab, judge_tab = st.tabs(["Inline Checks", "Human Review", "LLM Judge"])

        with runtime_tab:
            alert_col, summary_col = st.columns([1, 1])
            with alert_col:
                st.subheader("Alert Policy")
                _render_alert_panel(snapshot)
            with summary_col:
                st.subheader("Current Window")
                st.dataframe(_online_metric_rows(snapshot["summary"]))
                counts_col, risk_col = st.columns(2)
                with counts_col:
                    st.write("Outcome counts")
                    st.json(snapshot["summary"]["outcome_counts"])
                with risk_col:
                    st.write("Risk band counts")
                    st.json(snapshot["summary"]["risk_band_counts"])

            st.subheader("Average Runtime Scores")
            score_cols = st.columns(4)
            score_cols[0].metric("Groundedness", format_pct(snapshot["summary"]["avg_groundedness_proxy_score"]))
            score_cols[1].metric("Citation validity", format_pct(snapshot["summary"]["avg_citation_validity_score"]))
            score_cols[2].metric("Policy adherence", format_pct(snapshot["summary"]["avg_policy_adherence_score"]))
            score_cols[3].metric("Format validity", format_pct(snapshot["summary"]["avg_format_validity_score"]))
            score_cols_2 = st.columns(4)
            score_cols_2[0].metric("Retrieval support", format_pct(snapshot["summary"]["avg_retrieval_support_score"]))
            score_cols_2[1].metric("Brand voice", format_pct(snapshot["summary"]["avg_brand_voice_score"]))
            score_cols_2[2].metric("Tone", format_pct(snapshot["summary"]["avg_tone_appropriateness_score"]))
            score_cols_2[3].metric("Total runtime score", format_pct(snapshot["summary"]["avg_online_score_total"]))

            st.subheader("Grouped Metrics")
            group_tab_labels = [
                ("Prompt version", "by_prompt_version"),
                ("Model backend", "by_model_backend"),
                ("Outcome", "by_response_outcome"),
                ("Risk band", "by_risk_band"),
            ]
            tabs = st.tabs([label for label, _ in group_tab_labels])
            for tab, (_, key) in zip(tabs, group_tab_labels):
                with tab:
                    st.dataframe(snapshot["summary"][key])

            with st.expander("Recent feedback", expanded=False):
                st.dataframe(_flatten_feedback_rows(recent_feedback))

        with review_tab:
            review_cards = st.columns(4)
            review_cards[0].metric("Open items", len(open_review_items))
            review_cards[1].metric("Pending", review_status_counts.get("pending", 0))
            review_cards[2].metric("In review", review_status_counts.get("in_review", 0))
            review_cards[3].metric("Resolved", review_status_counts.get("resolved", 0))

            source_col, queue_col = st.columns([1, 2])
            with source_col:
                st.subheader("Queue sources")
                st.dataframe([{"review_source": source, "count": count} for source, count in sorted(review_source_counts.items())])
            with queue_col:
                st.subheader("Open review items")
                st.dataframe(open_review_items if open_review_items else review_items[:5])

            with st.expander("All review items", expanded=False):
                st.dataframe(review_items)

        with judge_tab:
            st.caption("Async OpenAI judge results over recent sampled runs.")
            judge_cards = st.columns(5)
            judge_cards[0].metric("Recent judge records", judge_summary["total_records"])
            judge_cards[1].metric("Completed", judge_summary["completed_count"])
            judge_cards[2].metric("Queued", judge_summary["queued_count"])
            judge_cards[3].metric("Failed", judge_summary["failed_count"])
            judge_cards[4].metric("Judge -> human", judge_summary["escalated_count"])

            judge_score_cards = st.columns(4)
            judge_score_cards[0].metric("Overall", format_pct(judge_summary["avg_overall_score"]))
            judge_score_cards[1].metric("Supportedness", format_pct(judge_summary["avg_supportedness_score"]))
            judge_score_cards[2].metric("Policy alignment", format_pct(judge_summary["avg_policy_alignment_score"]))
            judge_score_cards[3].metric("Response mode", format_pct(judge_summary["avg_response_mode_score"]))

            st.subheader("Recent Judge Status")
            st.dataframe(
                [
                    {"status": "completed", "count": judge_summary["completed_count"]},
                    {"status": "queued", "count": judge_summary["queued_count"]},
                    {"status": "failed", "count": judge_summary["failed_count"]},
                    {"status": "judge_recommended_human_review", "count": judge_summary["escalated_count"]},
                ]
            )

            st.subheader("Judge Results By Slice")
            slice_tabs = st.tabs(["Risk band", "Prompt version", "Model backend"])
            with slice_tabs[0]:
                st.dataframe(_group_judge_result_rows(recent_judge_records, "risk_band"))
            with slice_tabs[1]:
                st.dataframe(_group_judge_result_rows(recent_judge_records, "prompt_version"))
            with slice_tabs[2]:
                st.dataframe(_group_judge_result_rows(recent_judge_records, "model_backend"))

            with st.expander("Recent judge records", expanded=False):
                st.dataframe(_flatten_llm_judge_rows(recent_judge_records))

            st.write("Human review is created only when the judge recommends it, and queue dedupe still happens by `run_id`.")
    except Exception as exc:
        st.error(str(exc))

