from __future__ import annotations


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


def safe_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return float(count) / float(total)


def avg_value(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 4)


def inference_display_text(payload: dict) -> str:
    return (
        payload.get("answer")
        or payload.get("provisional_answer")
        or payload.get("missing_or_conflicting_evidence_summary")
        or payload.get("human_review_reason")
        or "No response text returned."
    )


def flatten_review_queue_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "review_queue_item_id": item["review_queue_item_id"],
            "run_id": item["run_id"],
            "review_status": item["review_status"],
            "review_priority": item["review_priority"],
            "review_source": item["review_source"],
            "online_score_total": item["online_score_total"],
            "suspicious_flags": ", ".join(item.get("suspicious_flags", [])),
            "reviewer_label": item.get("reviewer_label"),
            "should_have_outcome": item.get("should_have_outcome"),
            "promote_to_offline_eval": item.get("promote_to_offline_eval", False),
        }
        for item in items
    ]
