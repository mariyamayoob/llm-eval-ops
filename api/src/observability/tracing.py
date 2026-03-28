from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import mean
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from model.contracts import ScenarioName, SuspiciousFlag, TraceSummary


class SpanRecord(BaseModel):
    name: str
    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: float = 0.0
    attributes: dict[str, object] = Field(default_factory=dict)
    status: str = "ok"


class TraceRecord(BaseModel):
    trace_id: str
    run_id: str
    scenario: ScenarioName
    prompt_version: str
    started_at: datetime
    ended_at: datetime | None = None
    total_latency_ms: float = 0.0
    suspicious_flags: list[SuspiciousFlag] = Field(default_factory=list)
    spans: list[SpanRecord] = Field(default_factory=list)
    attributes: dict[str, object] = Field(default_factory=dict)


@dataclass
class SpanHandle:
    recorder: "TraceRecorder"
    trace_id: str
    name: str
    attributes: dict[str, object] = field(default_factory=dict)
    _started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _perf_started: float = field(default_factory=perf_counter)

    def finish(self, status: str = "ok", **attributes: object) -> float:
        ended_at = datetime.now(UTC)
        latency_ms = round((perf_counter() - self._perf_started) * 1000, 3)
        merged = dict(self.attributes)
        merged.update(attributes)
        self.recorder._record_span(
            self.trace_id,
            SpanRecord(
                name=self.name,
                started_at=self._started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                attributes=merged,
                status=status,
            ),
        )
        return latency_ms


class TraceRecorder:
    def __init__(self) -> None:
        self._traces: dict[str, TraceRecord] = {}

    def start_trace(self, *, run_id: str, scenario: ScenarioName, prompt_version: str, **attributes: object) -> str:
        trace_id = str(uuid4())
        self._traces[trace_id] = TraceRecord(
            trace_id=trace_id,
            run_id=run_id,
            scenario=scenario,
            prompt_version=prompt_version,
            started_at=datetime.now(UTC),
            attributes=dict(attributes),
        )
        return trace_id

    def start_span(self, trace_id: str, name: str, **attributes: object) -> SpanHandle:
        return SpanHandle(self, trace_id, name, dict(attributes))

    def _record_span(self, trace_id: str, span: SpanRecord) -> None:
        self._traces[trace_id].spans.append(span)

    def finalize_trace(self, trace_id: str, *, suspicious_flags: list[SuspiciousFlag], **attributes: object) -> None:
        trace = self._traces[trace_id]
        trace.ended_at = datetime.now(UTC)
        trace.total_latency_ms = round(sum(span.latency_ms for span in trace.spans), 3)
        trace.suspicious_flags = list(suspicious_flags)
        trace.attributes.update(attributes)

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces.get(trace_id)

    def list_traces(self) -> list[TraceSummary]:
        traces = sorted(self._traces.values(), key=lambda item: item.started_at, reverse=True)
        return [
            TraceSummary(
                trace_id=item.trace_id,
                run_id=item.run_id,
                scenario=item.scenario,
                prompt_version=item.prompt_version,
                total_latency_ms=item.total_latency_ms,
                suspicious_flags=item.suspicious_flags,
            )
            for item in traces
        ]

    def summary(self) -> dict[str, Any]:
        latencies = [trace.total_latency_ms for trace in self._traces.values() if trace.total_latency_ms]
        return {
            "trace_count": len(self._traces),
            "mean_total_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
        }
