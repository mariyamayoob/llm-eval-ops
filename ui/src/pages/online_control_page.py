from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timezone
from typing import Any

import streamlit as st

from ui_api import ApiClient
from ui_config import BASE_PATH
from ui_utils import avg_value, format_pct, safe_rate


_CONTROL_PLANE_CSS = """
<style>
  .cp-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 14px; }
  .cp-title { font-size: 0.95rem; font-weight: 650; color: rgba(255,255,255,0.88); margin: 0 0 8px 0; }
  .cp-title-strong { font-weight: 760; }
  .cp-kpi { font-size: 1.6rem; font-weight: 760; line-height: 1.1; margin: 0 0 2px 0; }
  .cp-subtle { color: rgba(255,255,255,0.70); font-size: 0.86rem; margin: 0; }
  .cp-line { color: rgba(255,255,255,0.84); font-size: 0.92rem; margin: 2px 0 0 0; }
  .cp-badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:0.82rem; font-weight:650; border:1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.92); }
  .cp-green { border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.12); }
  .cp-amber { border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.12); }
  .cp-red { border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.12); }
  .cp-arrow { color: rgba(255,255,255,0.45); font-size: 1.2rem; margin-top: 20px; text-align: center; }
  .cp-list { margin: 6px 0 0 0; padding-left: 18px; color: rgba(255,255,255,0.88); font-size: 0.92rem; }
  .cp-accent { border-color: rgba(59,130,246,0.45); }
  .cp-section { font-size: 1.02rem; font-weight: 760; color: rgba(255,255,255,0.86); margin: 16px 0 8px 0; }
  .cp-section-sub { font-size: 0.88rem; color: rgba(255,255,255,0.70); margin: -2px 0 10px 0; }
  .cp-divider { height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0 12px 0; }
</style>
"""

_RUNTIME_BLOCKING_FLAGS = {
    "citations_not_in_retrieval",
    "possible_policy_mismatch",
    "repair_attempted",
    "unsupported_claim_signal",
}


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


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


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _truncate(value: str | None, *, max_len: int = 120) -> str:
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _card(title: str, *, kpi: str | None = None, subtitle: str | None = None, bullets: list[str] | None = None):
    parts: list[str] = [f"<div class='cp-card'><div class='cp-title'>{_escape_html(title)}</div>"]
    if kpi is not None:
        parts.append(f"<div class='cp-kpi'>{_escape_html(kpi)}</div>")
    if subtitle is not None:
        parts.append(f"<div class='cp-subtle'>{_escape_html(subtitle)}</div>")
    if bullets:
        items = "".join([f"<li>{_escape_html(item)}</li>" for item in bullets])
        parts.append(f"<ul class='cp-list'>{items}</ul>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _accent_card(title: str, *, kpi: str | None = None, subtitle: str | None = None, bullets: list[str] | None = None):
    parts: list[str] = [
        f"<div class='cp-card cp-accent'><div class='cp-title cp-title-strong'>{_escape_html(title)}</div>"
    ]
    if kpi is not None:
        parts.append(f"<div class='cp-kpi'>{_escape_html(kpi)}</div>")
    if subtitle is not None:
        parts.append(f"<div class='cp-subtle'>{_escape_html(subtitle)}</div>")
    if bullets:
        items = "".join([f"<li>{_escape_html(item)}</li>" for item in bullets])
        parts.append(f"<ul class='cp-list'>{items}</ul>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _section(title: str, *, subtitle: str | None = None):
    st.markdown(f"<div class='cp-section'>{_escape_html(title)}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='cp-section-sub'>{_escape_html(subtitle)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='cp-divider'></div>", unsafe_allow_html=True)


def _confidence_badge(status: str) -> tuple[str, str]:
    if status == "ok":
        return "Healthy", "cp-badge cp-green"
    if status == "watch":
        return "Watch", "cp-badge cp-amber"
    if status == "action_required":
        return "Action Required", "cp-badge cp-red"
    return (status or "unknown"), "cp-badge"


def _confidence_reason(*, triggered_alerts: int, open_review_items: int, median_queue_age_hours: float | None, judge_escalations: int) -> str:
    if triggered_alerts > 0:
        return "Alert thresholds triggered"
    if open_review_items > 0 and median_queue_age_hours is not None and median_queue_age_hours >= 24:
        return "Queue aging above target"
    if judge_escalations > 0:
        return "Judge escalations elevated"
    return "All thresholds within range"


def _stage_health_card(
    *,
    title: str,
    primary: str,
    support_lines: list[str],
    highlight: str | None = None,
    accent: bool = False,
):
    container_class = "cp-card cp-accent" if accent else "cp-card"
    parts: list[str] = [f"<div class='{container_class}'>", f"<div class='cp-title'>{_escape_html(title)}</div>"]
    parts.append(f"<div class='cp-kpi'>{_escape_html(primary)}</div>")
    for line in support_lines[:4]:
        parts.append(f"<div class='cp-line'>{_escape_html(line)}</div>")
    if highlight:
        parts.append(
            f"<div class='cp-subtle' style='margin-top:8px; font-weight:650;'>{_escape_html(highlight)}</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_funnel(*, total_runs: int, deterministic_flagged: int, judged: int, human_review: int, promoted: int):
    def pct(count: int) -> str:
        if total_runs <= 0:
            return "0%"
        return f"{round(100.0 * count / total_runs)}%"

    cols = st.columns([1, 0.25, 1, 0.25, 1, 0.25, 1, 0.25, 1])
    stages = [
        ("Requests", f"{total_runs}", "Runs in current window"),
        ("Flagged by inline checks", f"{deterministic_flagged}", pct(deterministic_flagged)),
        ("Reviewed by LLM judge", f"{judged}", pct(judged)),
        ("Routed to human review", f"{human_review}", pct(human_review)),
        ("Promoted to offline evals", f"{promoted}", pct(promoted)),
    ]
    for i, (label, count, cap) in enumerate(stages):
        with cols[i * 2]:
            _card(label, kpi=count, subtitle=cap)
        if i < len(stages) - 1:
            with cols[i * 2 + 1]:
                st.markdown("<div class='cp-arrow'>&rarr;</div>", unsafe_allow_html=True)



def _offline_learning_summary(*, review_items: list[dict], export_cases: list[dict] | None) -> dict[str, Any]:
    promoted_resolved = [
        item for item in review_items if item.get("review_status") == "resolved" and item.get("promote_to_offline_eval")
    ]
    backlog = [
        item
        for item in review_items
        if item.get("review_status") == "resolved"
        and not item.get("promote_to_offline_eval")
        and (item.get("should_have_outcome") in {"refused_more_evidence_needed", "human_review_recommended"})
    ]
    return {
        "promoted_count": len(promoted_resolved),
        "promotion_backlog": len(backlog),
        "exported_count": len(export_cases or []) if export_cases is not None else None,
    }


def _top_blocking_flag(recent_runs: list[dict]) -> str:
    flagged = [
        run
        for run in recent_runs
        if any(flag in _RUNTIME_BLOCKING_FLAGS for flag in (run.get("suspicious_flags") or []))
    ]
    counts: Counter[str] = Counter()
    for run in flagged:
        for flag in run.get("suspicious_flags") or []:
            counts[str(flag)] += 1
    return counts.most_common(1)[0][0] if counts else "none"


def _top_recurring_cluster(review_items: list[dict]) -> str:
    counts: Counter[str] = Counter()
    for item in review_items:
        for code in item.get("review_reason_codes") or []:
            counts[str(code)] += 1
        if not (item.get("review_reason_codes") or []):
            for flag in item.get("suspicious_flags") or []:
                counts[str(flag)] += 1
    return counts.most_common(1)[0][0] if counts else "unavailable"


def _operator_message(
    *,
    open_review_queue: int,
    judge_escalations: int,
    median_queue_age_hours: float | None = None,
    medium_risk_judge_escalations: int = 0,
    promotion_backlog: int,
    triggered_alerts: int,
    rollback_recommended: bool,
) -> str:
    if rollback_recommended or triggered_alerts > 0:
        return f"Online alert thresholds triggered ({triggered_alerts}). Inspect worst runs and prompt/model slices."
    if (
        open_review_queue > 0
        and median_queue_age_hours is not None
        and median_queue_age_hours >= 24
    ):
        return f"Review queue aging is above target across {open_review_queue} open items (median {round(median_queue_age_hours, 1)}h)."
    if promotion_backlog > 0:
        return "Offline learning backlog growing; promote confirmed failures into offline evals."
    if judge_escalations > 0:
        if medium_risk_judge_escalations > 0:
            return f"Judge escalations are elevated in medium-risk runs ({medium_risk_judge_escalations}/{judge_escalations})."
        return f"Judge escalations present ({judge_escalations}). Review escalations and promote confirmed failures."
    if open_review_queue > 0:
        return f"Review queue has {open_review_queue} open items. Triage to keep backlog controlled."
    return "No manual action needed."


def _priority_actions(
    *,
    open_review_queue: int,
    median_queue_age_hours: float | None,
    judge_escalations: int,
    promotion_backlog: int,
    triggered_alerts: int,
    rollback_recommended: bool,
) -> list[str]:
    actions: list[str] = []
    if rollback_recommended or triggered_alerts > 0:
        actions.append("Inspect triggered thresholds and worst-run slices")
    if judge_escalations > 0:
        actions.append(f"Review {judge_escalations} LLM judge escalations")
    if open_review_queue > 0 and median_queue_age_hours is not None and median_queue_age_hours >= 24:
        actions.append(f"Triage {open_review_queue} open review items (aging)")
    elif open_review_queue > 0:
        actions.append(f"Triage {open_review_queue} open review items")
    if promotion_backlog > 0:
        actions.append(f"Promote {promotion_backlog} resolved failures into offline evals")
    if not actions and triggered_alerts > 0:
        actions.append("Validate stability across prompt/model slices")
    if not actions:
        actions.append("No manual action needed")
    return actions[:3]


def _top_issues_table(*, recent_runs: list[dict], recent_judge_records: list[dict], review_items: list[dict]) -> list[dict]:
    rows: list[dict] = []

    flagged = [
        run
        for run in recent_runs
        if any(flag in _RUNTIME_BLOCKING_FLAGS for flag in (run.get("suspicious_flags") or []))
    ]
    flag_counts: Counter[str] = Counter()
    for run in flagged:
        for flag in run.get("suspicious_flags") or []:
            flag_counts[str(flag)] += 1
    for issue, count in flag_counts.most_common(6):
        rows.append(
            {
                "issue / cluster": issue,
                "stage": "Inline Checks",
                "count": count,
                "trend": "trend unavailable",
                "recommended action": "inspect retrieval/citation/prompt logic",
            }
        )

    judge_completed = [r for r in recent_judge_records if r.get("status") == "completed" and r.get("assessment")]
    judge_escalations = [r for r in judge_completed if r["assessment"].get("human_review_recommended")]
    if judge_escalations:
        rows.append(
            {
                "issue / cluster": "judge_escalations",
                "stage": "LLM Judge",
                "count": len(judge_escalations),
                "trend": "trend unavailable",
                "recommended action": "review escalations and promote confirmed failures",
            }
        )

    open_items = [item for item in review_items if item.get("review_status") != "resolved"]
    src_counts = Counter(item.get("review_source") or "unknown" for item in open_items)
    for source, count in src_counts.most_common(4):
        rows.append(
            {
                "issue / cluster": source,
                "stage": "Human Review",
                "count": count,
                "trend": "trend unavailable",
                "recommended action": "triage and resolve; promote recurring failures",
            }
        )

    return sorted(rows, key=lambda row: (-int(row["count"]), row["stage"], row["issue / cluster"]))[:12]


def _recent_escalations_rows(*, review_items: list[dict], run_by_id: dict[str, dict]) -> list[dict]:
    open_items = [item for item in review_items if item.get("review_status") != "resolved"]
    rows: list[dict] = []
    for item in open_items[:10]:
        run = run_by_id.get(item.get("run_id"))
        rows.append(
            {
                "run_id": item.get("run_id"),
                "source": item.get("review_source"),
                "reason": _truncate(item.get("review_reason"), max_len=100),
                "risk": (run or {}).get("risk_band", "unknown"),
            }
        )
    return rows


def render(api: ApiClient):
    st.markdown(_CONTROL_PLANE_CSS, unsafe_allow_html=True)
    st.title("Online Control Plane")
    st.caption(
        "One operator view for the online eval funnel: inline checks absorb routine failures, "
        "LLM judge reviews the uncertain slice, humans handle the exceptions, "
        "and confirmed failures feed back into offline evals."
    )

    try:
        snapshot = api.get(f"{BASE_PATH}/feedback/summary?limit=50")
        recent_feedback = api.get(f"{BASE_PATH}/feedback?limit=15")
        review_items = api.get(f"{BASE_PATH}/review-queue")
        judge_records = api.get(f"{BASE_PATH}/llm-judge?limit=50")
        recent_runs = _load_recent_run_details(api, 50)

        export_cases: list[dict] | None = None
        try:
            export_payload = api.get(f"{BASE_PATH}/offline-eval/export")
            export_cases = export_payload.get("cases") or []
        except Exception:
            export_cases = None

        recent_run_ids = {run["run_id"] for run in recent_runs}
        run_by_id = {run["run_id"]: run for run in recent_runs}

        recent_judge_records = [record for record in judge_records if record.get("run_id") in recent_run_ids]
        judge_summary = _summarize_judge_records(recent_judge_records)
        judged_run_ids = {record["run_id"] for record in recent_judge_records}
        judge_completed = [record for record in recent_judge_records if record.get("status") == "completed" and record.get("assessment")]
        judge_escalations = [record for record in judge_completed if record["assessment"].get("human_review_recommended")]
        medium_risk_judge_escalations = sum(1 for record in judge_escalations if str(record.get("risk_band")) == "medium")

        review_status_counts = Counter(item["review_status"] for item in review_items)
        review_source_counts = Counter(item["review_source"] for item in review_items)
        open_review_items = [item for item in review_items if item.get("review_status") != "resolved"]

        total_runs = int(snapshot["summary"]["total_runs"])
        deterministic_flagged = sum(
            1
            for run in recent_runs
            if any(flag in _RUNTIME_BLOCKING_FLAGS for flag in (run.get("suspicious_flags") or []))
        )
        review_items_in_window = [item for item in review_items if item.get("run_id") in recent_run_ids]
        promoted_in_window = [
            item
            for item in review_items_in_window
            if item.get("review_status") == "resolved" and item.get("promote_to_offline_eval")
        ]

        offline_learning = _offline_learning_summary(review_items=review_items, export_cases=export_cases)
        top_cluster = _top_recurring_cluster(review_items)

        queue_age_hours: list[float] = []
        now = datetime.now(UTC)
        for item in open_review_items:
            created_at = _parse_dt((run_by_id.get(item.get("run_id")) or {}).get("created_at"))
            if created_at is None:
                continue
            queue_age_hours.append(max(0.0, (now - created_at).total_seconds() / 3600.0))
        median_age = _median(queue_age_hours)

        alert_eval = snapshot.get("alert_evaluation") or {}
        confidence_status = str(alert_eval.get("status") or "unknown")
        triggered_alerts = alert_eval.get("triggered_alerts") or []
        rollback_recommended = bool(alert_eval.get("rollback_recommended"))
        confidence_label, confidence_class = _confidence_badge(confidence_status)
        confidence_reason = _confidence_reason(
            triggered_alerts=len(triggered_alerts),
            open_review_items=len(open_review_items),
            median_queue_age_hours=median_age,
            judge_escalations=judge_summary["escalated_count"],
        )

        overview_tab, inline_tab, judge_tab, human_tab, offline_tab = st.tabs(
            ["Overview", "Inline Checks", "LLM Judge", "Human Review", "Offline Learning"]
        )

        with overview_tab:
            _section("Operator Status")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    "<div class='cp-card'>"
                    "<div class='cp-title'>Confidence Status</div>"
                    f"<span class='{confidence_class}'>{_escape_html(confidence_label)}</span>"
                    f"<div class='cp-line' style='margin-top:8px;'>{_escape_html(confidence_reason)}</div>"
                    f"<div class='cp-subtle' style='margin-top:6px;'>Threshold state: {len(triggered_alerts)} triggered</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                _card(
                    "Operator Message",
                    subtitle=_operator_message(
                        open_review_queue=len(open_review_items),
                        median_queue_age_hours=median_age,
                        judge_escalations=judge_summary["escalated_count"],
                        medium_risk_judge_escalations=medium_risk_judge_escalations,
                        promotion_backlog=offline_learning["promotion_backlog"],
                        triggered_alerts=len(triggered_alerts),
                        rollback_recommended=rollback_recommended,
                    ),
                )
            with c3:
                _card(
                    "Priority Actions",
                    bullets=_priority_actions(
                        open_review_queue=len(open_review_items),
                        median_queue_age_hours=median_age,
                        judge_escalations=judge_summary["escalated_count"],
                        promotion_backlog=offline_learning["promotion_backlog"],
                        triggered_alerts=len(triggered_alerts),
                        rollback_recommended=rollback_recommended,
                    ),
                )

            with st.expander("Alerts and Policy Details", expanded=False):
                st.write("Sampling policy")
                st.markdown(
                    "- Inline deterministic checks: 100% of requests\n"
                    "- LLM judge: 100% medium/high + 20% low (deterministic by run_id)\n"
                    "- Human review: only when escalated\n"
                    "- Offline eval: promote confirmed failures",
                )

                st.write("Secondary signals (alerts)")
                if not triggered_alerts:
                    st.caption("No thresholds triggered in this window.")
                else:
                    st.dataframe(triggered_alerts, use_container_width=True)
                if rollback_recommended and alert_eval.get("rollback_reasons"):
                    st.write("Rollback reasons")
                    for reason in alert_eval.get("rollback_reasons") or []:
                        st.write(f"- {reason}")

                st.divider()
                st.write("Reset demo data (danger)")
                confirm = st.text_input("Type RESET to clear all SQLite tables", value="", key="reset_confirm")
                c_reset_all, c_reset_partial = st.columns(2)
                with c_reset_all:
                    reset_all = st.button("Clear all tables", disabled=confirm.strip().upper() != "RESET")
                with c_reset_partial:
                    reset_review_judge = st.button(
                        "Clear review + judge tables",
                        disabled=confirm.strip().upper() != "RESET",
                        help="Deletes rows from Review Queue + LLM Judge (keeps run history).",
                    )

                if reset_all:
                    try:
                        result = api.post(f"{BASE_PATH}/dev/reset", {"confirm": confirm})
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success("Cleared demo tables.")
                        st.json(result.get("deleted", result))
                        st.rerun()

                if reset_review_judge:
                    try:
                        result = api.post(f"{BASE_PATH}/dev/reset/review-and-judge", {"confirm": confirm})
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success("Cleared review + judge tables.")
                        st.json(result.get("deleted", result))
                        st.rerun()

            _section(
                "Evaluation Pipeline"
            )
            _render_funnel(
                total_runs=total_runs,
                deterministic_flagged=deterministic_flagged,
                judged=len(judged_run_ids),
                human_review=len(review_items_in_window),
                promoted=len(promoted_in_window),
            )

            _section("Stage Health")
            a, b, c, d = st.columns(4)
            with a:
                _card(
                    "Inline Checks",
                    bullets=[
                        f"Invalid output rate: {format_pct(snapshot['summary']['invalid_output_rate'])}",
                        f"Deterministic review rate: {format_pct(safe_rate(deterministic_flagged, total_runs))}",
                        f"Supported answer rate: {format_pct(snapshot['summary']['supported_answer_rate'])}",
                        f"High-risk count: {snapshot['summary']['high_risk_run_count']}",
                        f"Top blocker: {_top_blocking_flag(recent_runs)}",
                    ],
                )
            with b:
                _card(
                    "LLM Judge",
                    bullets=[
                        f"Judge coverage: {format_pct(safe_rate(len(judged_run_ids), len(recent_runs)))}",
                        f"Completed: {judge_summary['completed_count']}",
                        f"Failed: {judge_summary['failed_count']}",
                        f"Judge escalations: {judge_summary['escalated_count']}",
                        f"Avg judge score: {format_pct(judge_summary['avg_overall_score'])}",
                    ],
                )
            with c:
                _card(
                    "Human Review",
                    bullets=[
                        f"Open review items: {len(open_review_items)}",
                        f"Pending: {review_status_counts.get('pending', 0)}",
                        f"In review: {review_status_counts.get('in_review', 0)}",
                        f"Resolved: {review_status_counts.get('resolved', 0)}",
                        f"Median queue age: {round(median_age, 1)}h" if median_age is not None else "Median queue age: unavailable",
                    ],
                )
            with d:
                exported_label = offline_learning["exported_count"] if offline_learning["exported_count"] is not None else "unavailable"
                _accent_card(
                    "Offline Learning",
                    bullets=[
                        f"Promoted cases: {offline_learning['promoted_count']}",
                        f"Promotion backlog: {offline_learning['promotion_backlog']}",
                        f"Recent exported cases: {exported_label}",
                        f"Top recurring cluster: {top_cluster}",
                    ],
                )

            st.subheader("Operator Detail Panels")
            left, right = st.columns([1.15, 1])
            with left:
                st.markdown("<div class='cp-card'><div class='cp-title'>Top Issues / Failure Clusters</div></div>", unsafe_allow_html=True)
                st.dataframe(_top_issues_table(recent_runs=recent_runs, recent_judge_records=recent_judge_records, review_items=review_items), use_container_width=True)
            with right:
                st.markdown("<div class='cp-card'><div class='cp-title'>Recent Escalations / Learning Loop</div></div>", unsafe_allow_html=True)
                st.write("Recent escalations")
                st.dataframe(_recent_escalations_rows(review_items=review_items, run_by_id=run_by_id), use_container_width=True)
                st.write("Recently promoted to offline eval")
                if export_cases is None:
                    st.caption("Export status unavailable.")
                else:
                    st.dataframe(
                        [
                            {
                                "case_id": case.get("case_id"),
                                "scenario": case.get("scenario"),
                                "expected_behavior": case.get("expected_behavior"),
                            }
                            for case in export_cases[:10]
                        ],
                        use_container_width=True,
                    )

        with inline_tab:
            st.subheader("Inline Checks")
            st.info(
                "Inline deterministic checks run on every request. "
                "Hard blockers are flagged deterministically and routed to the LLM judge. "
                "LLM judge runs on 100% of medium/high-risk runs and 20% of low-risk runs (deterministic by run_id)."
            )

            st.subheader("Average Inline Check Scores")
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
                    st.dataframe(snapshot["summary"][key], use_container_width=True)

            with st.expander("Recent feedback (secondary signal)", expanded=False):
                st.dataframe(_flatten_feedback_rows(recent_feedback), use_container_width=True)

        with judge_tab:
            st.subheader("LLM Judge")
            judge_cards = st.columns(5)
            judge_cards[0].metric("Judge coverage", format_pct(safe_rate(len(judged_run_ids), len(recent_runs))))
            judge_cards[1].metric("Completed", judge_summary["completed_count"])
            judge_cards[2].metric("Queued", judge_summary["queued_count"])
            judge_cards[3].metric("Failed", judge_summary["failed_count"])
            judge_cards[4].metric("Judge escalations", judge_summary["escalated_count"])

            st.subheader("Judge Results By Slice")
            slice_tabs = st.tabs(["Risk band", "Prompt version", "Model backend"])
            with slice_tabs[0]:
                st.dataframe(_group_judge_result_rows(recent_judge_records, "risk_band"), use_container_width=True)
            with slice_tabs[1]:
                st.dataframe(_group_judge_result_rows(recent_judge_records, "prompt_version"), use_container_width=True)
            with slice_tabs[2]:
                st.dataframe(_group_judge_result_rows(recent_judge_records, "model_backend"), use_container_width=True)

            with st.expander("Recent judge records", expanded=False):
                st.dataframe(_flatten_llm_judge_rows(recent_judge_records), use_container_width=True)

        with human_tab:
            st.subheader("Human Review")
            review_cards = st.columns(5)
            review_cards[0].metric("Open review items", len(open_review_items))
            review_cards[1].metric("Pending", review_status_counts.get("pending", 0))
            review_cards[2].metric("In review", review_status_counts.get("in_review", 0))
            review_cards[3].metric("Resolved", review_status_counts.get("resolved", 0))
            review_cards[4].metric("Median queue age", f"{round(median_age, 1)}h" if median_age is not None else "age unavailable")

            source_col, queue_col = st.columns([1, 2])
            with source_col:
                st.subheader("Queue sources")
                st.dataframe([{"review_source": source, "count": count} for source, count in sorted(review_source_counts.items())], use_container_width=True)
            with queue_col:
                st.subheader("Open review items")
                st.dataframe(open_review_items if open_review_items else review_items[:5], use_container_width=True)

            with st.expander("All review items", expanded=False):
                st.dataframe(review_items, use_container_width=True)

        with offline_tab:
            st.subheader("Offline Learning")
            st.info("Confirmed failures can be promoted into offline eval cases (PII-redacted portable JSON).")

            left, right = st.columns([1, 1])
            with left:
                st.metric("Promoted to offline eval", offline_learning["promoted_count"])
                st.metric("Promotion backlog", offline_learning["promotion_backlog"])
                st.metric("Top recurring cluster", top_cluster)
                if offline_learning["exported_count"] is None:
                    st.caption("Export status unavailable.")
                else:
                    st.metric("Exported cases (current)", offline_learning["exported_count"])

            with right:
                st.write("Recently exported cases")
                if export_cases is None:
                    st.caption("Export status unavailable.")
                else:
                    st.dataframe(
                        [
                            {
                                "case_id": case.get("case_id"),
                                "scenario": case.get("scenario"),
                                "expected_behavior": case.get("expected_behavior"),
                                "label_notes": _truncate(case.get("label_notes"), max_len=120),
                            }
                            for case in export_cases[:25]
                        ],
                        use_container_width=True,
                    )
    except Exception as exc:
        st.error(str(exc))
