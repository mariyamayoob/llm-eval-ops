from __future__ import annotations

import json

import streamlit as st

from ui_api import ApiClient
from ui_config import BASE_PATH
from ui_utils import flatten_review_queue_rows, inference_display_text


def render(api: ApiClient):
    st.title("Review Queue")
    st.caption("Lookup a review item, compare the current output vs the intended output, then mark the review complete and optionally promote it to offline eval.")

    items: list[dict] = []
    try:
        if not st.session_state.get("review_queue_demo_pruned"):
            try:
                api.post(f"{BASE_PATH}/review-queue/prune?max_open_runtime_items=20", {})
            finally:
                st.session_state["review_queue_demo_pruned"] = True
        items = api.get(f"{BASE_PATH}/review-queue")
        st.dataframe(flatten_review_queue_rows(items))
    except Exception as exc:
        st.error(str(exc))

    item_id = st.text_input("Review queue item id")
    selected_item = None
    selected_run = None
    if item_id and items:
        try:
            selected_item = next((item for item in items if item["review_queue_item_id"] == item_id), None)
            if selected_item is None:
                st.warning("Review queue item not found. Paste a `review_queue_item_id` from the table above.")
            else:
                run_payload = api.get(f"{BASE_PATH}/runs/{selected_item['run_id']}")
                selected_run = run_payload.get("run")
        except Exception as exc:
            st.error(str(exc))

    if not selected_item:
        return

    st.subheader("Current Run Result")
    current_outcome = selected_run["outcome"] if selected_run else None
    st.write(f"Run: `{selected_item['run_id']}`  |  Current outcome: `{current_outcome}`")
    if selected_run:
        st.caption("Response preview")
        st.write(inference_display_text(selected_run["response_payload"]))

    st.subheader("Reviewer Correction")
    correction_cols = st.columns(2)
    with correction_cols[0]:
        st.write("What the system returned")
        st.json(
            {
                "outcome": current_outcome,
                "risk_band": selected_run["risk_band"] if selected_run else None,
                "online_score_total": selected_run["online_score_total"] if selected_run else None,
                "suspicious_flags": selected_run["suspicious_flags"] if selected_run else None,
                "review_source": selected_item.get("review_source"),
                "review_reason": selected_item.get("review_reason"),
            }
        )

    with correction_cols[1]:
        reviewer = st.text_input("Reviewer", value=selected_item.get("reviewer_label") or "")
        notes = st.text_area("Reviewer notes", value=selected_item.get("reviewer_notes") or "")
        review_complete = st.checkbox("Review complete", value=selected_item.get("review_status") == "resolved")
        review_status = "resolved" if review_complete else "in_review"

        should_have_outcome_default = selected_item.get("should_have_outcome") or current_outcome or "supported_answer"
        should_have_outcome = st.selectbox(
            "What it should have returned (outcome label)",
            ["supported_answer", "refused_more_evidence_needed", "human_review_recommended"],
            index=["supported_answer", "refused_more_evidence_needed", "human_review_recommended"].index(should_have_outcome_default),
        )
        st.caption("This is the main correction signal used for metrics + offline export.")

        current_output_text = inference_display_text(selected_run["response_payload"]) if selected_run else ""
        should_have_response_default = selected_item.get("should_have_response_text") or current_output_text
        should_have_response_text = st.text_area(
            "What it should have returned (response text)",
            value=should_have_response_default,
            height=180,
        )

        promote_to_offline_eval = st.checkbox(
            "Add to offline eval export (PII-redacted)",
            value=bool(selected_item.get("promote_to_offline_eval", False)),
        )

        if st.button("Save review update") and reviewer:
            try:
                st.json(
                    api.post(
                        f"{BASE_PATH}/review-queue/{item_id}/annotate",
                        {
                            "reviewer_label": reviewer,
                            "reviewer_notes": notes,
                            "review_status": review_status,
                            "promote_to_offline_eval": promote_to_offline_eval,
                            "should_have_outcome": should_have_outcome,
                            "should_have_response_text": should_have_response_text,
                        },
                    )
                )
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Offline Eval Export")
    st.caption("This generates a JSON file of promoted cases with demo-grade PII redaction, so you can show the 'review -> offline eval' loop.")
    if st.button("Generate download"):
        try:
            payload = api.get(f"{BASE_PATH}/offline-eval/export")
            st.download_button(
                "Download promoted cases",
                data=json.dumps(payload, indent=2).encode("utf-8"),
                file_name="promoted_eval_cases.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(str(exc))

