# Architecture

## System Overview

`llm-eval-ops` is a bounded LLM evaluation reference system rather than a generic chatbot. It answers synthetic policy questions over a synthetic knowledge base, validates structured outputs, scores every runtime response, routes suspicious cases to review, persists run data in SQLite, and supports offline regression gating over a labeled dataset.

The repo is intentionally organized around three practical questions:

1. Is the output valid and supported by retrieved evidence?
2. Did the system choose the expected response mode (answer vs abstain vs human review)?
3. Did the candidate regress versus baseline on the same dataset?

Backend code lives under `api/src/`. The Streamlit demo UI lives under `ui/src/` (with `ui/src/streamlit_app.py` as the router and `ui/src/pages/` containing page modules).

## Runtime Outcomes

Runtime inference produces one of three outcomes:

- `supported_answer`: answer is present, citations are present, and evidence summary exists.
- `refused_more_evidence_needed`: refusal with an explicit refusal reason + evidence gap summary.
- `human_review_recommended`: the system decided a human should inspect (based on suspicious flags / low score / boundary behavior).

Note: `review_required` is tracked separately as an operator signal; a “safe abstain” can still be safe even if review is suggested.

## Request Lifecycle (Online Inference)

Entry point: `POST /policy-desk-assistant/respond` (FastAPI in `api/src/main.py`).

Main orchestration: `PolicyDeskService` (`api/src/services/qa_service.py`).

High-level sequence:

1. Create `run_id`, `trace_id`, and a question hash.
2. Retrieve evidence with deterministic lexical retrieval (`RetrievalService`).
3. Call the model backend (`ModelService`) using a prompt profile (`prompt_version`).
4. Parse + validate structured output (`ValidationService`).
5. Compute runtime score breakdown, suspicious flags, and risk band (`OnlineScoringService`).
6. Decide whether to enqueue a review item (runtime routing).
7. Persist the run record (and review item if created) via `RunService` → `StorageService` (SQLite).
8. Finalize the trace (`TraceRecorder`) with spans and summary fields.

Trace spans are recorded for: `retrieve`, `prompt_build`, `llm_call`, `output_parse`, `output_validate`, `output_repair`, `online_score`, `review_route`, and `persist_run`.

## Persistence (SQLite)

SQLite is initialized by `StorageService` (`api/src/services/storage_service.py`). Key persisted entities:

- `run_records`: full run record JSON, indexed by `run_id`.
- `review_queue`: review queue items (minimal workflow state + JSON payload).
- `reviewer_annotations`: append-only reviewer updates (history).
- `runtime_feedback_events`: user/operator feedback events linked to `run_id`.
- `llm_judge_records`: async LLM-judge records linked to `run_id`.
- `offline_eval_runs` / `offline_eval_cases`: offline eval summaries and per-case results.
- `offline_comparison_runs` / `offline_comparison_cases`: baseline-vs-candidate comparisons.

This is a reference app: no external message bus, no distributed tracing backend, no streaming ingestion.

## Review Queue

Review items are created by multiple paths:

- runtime routing (inline deterministic checks and suspicious flags)
- online control plane auto-enqueue when a high-risk run receives thumbs-down feedback
- online summary alert evaluation when status becomes `action_required` (enqueues “worst runs”)
- LLM judge when it recommends human review (enqueues with source `llm_judge`)

Dedupe is by `run_id` to avoid queue spam when multiple signals point to the same run.

Reviewer updates are written via:

- `POST /policy-desk-assistant/review-queue/{item_id}/annotate`

The review correction signal is intentionally simple:

- `should_have_outcome`: what the system should have returned (`supported_answer`, `refused_more_evidence_needed`, or `human_review_recommended`)
- `should_have_response_text`: free-form reference text for humans (not scored by offline gates today)
- `promote_to_offline_eval`: include this item in the offline export set (with demo-grade PII redaction)

There is also a demo-only queue cleanup endpoint:

- `POST /policy-desk-assistant/review-queue/prune?max_open_runtime_items=20`

It deletes older untouched `pending` runtime items (never deletes `llm_judge` items). This exists to keep the UI usable for screenshots and demos; it is not a retention policy.

## Offline Eval Harness

Offline evals replay a labeled dataset through the same runtime pipeline and score results against a versioned gate policy.

Key pieces:

- Dataset: `data/eval_dataset.json` (`EvalCase` in `api/src/evals/contracts.py`)
- Gate policy: `data/offline_gate_policy.json`
- Scoring + blocker taxonomy: `api/src/evals/scorers.py`
- Gate evaluation: `api/src/evals/gates.py`
- Runner: `api/src/evals/runner.py`

Offline endpoints:

- `GET /policy-desk-assistant/evals/offline` (single-run summary + case results)
- `POST /policy-desk-assistant/evals/offline/compare` (baseline vs candidate)
- `GET /policy-desk-assistant/evals/offline/{eval_run_id}`
- `GET /policy-desk-assistant/evals/offline/comparisons/{comparison_id}`

Case sets:

- `case_set=portable`: core release suite (works for both mock + OpenAI backends)
- `case_set=full`: portable suite plus mock-only stress cases

## Online Control Plane (Reference Implementation)

The online control plane is a lightweight operator layer that rolls up recent runs + feedback into a business-readable summary, evaluates it against thresholds, and can open review items when the system drifts.

Components:

- Feedback ingestion and persistence (`OnlineControlPlaneService.record_feedback`)
- Rolling summary over recent N runs (`OnlineControlPlaneService.build_live_summary`)
- Alert policy evaluation based on `data/online_alert_policy.json`

Endpoints:

- `POST /policy-desk-assistant/feedback`
- `GET /policy-desk-assistant/feedback`
- `GET /policy-desk-assistant/feedback/summary`

The summary is deterministic and intentionally small (counts + rates + average runtime scores) and also includes grouped metrics by:

- `prompt_version`
- `model_backend`
- `response_outcome`
- `risk_band`

`rollback_recommended` is only an operator recommendation signal. It does not perform traffic switching.

## Async OpenAI Judge (Reference Layer)

The judge is a small “LLM-as-judge” layer that samples stored runs (that were not already routed to human review), grades them against retrieved evidence, and optionally recommends human review.

Sampling:

- deterministic by `run_id`
- 10% of `low` risk runs
- 30% of `medium` risk runs
- 100% of `high` risk runs

Judge metrics:

- `supportedness_score`
- `policy_alignment_score`
- `response_mode_score`
- `overall_score` (simple average)

If the judge recommends human review, a review item is created with `review_source="llm_judge"`.

Endpoints:

- `GET /policy-desk-assistant/llm-judge`
- `GET /policy-desk-assistant/llm-judge/{judge_id}`
- `POST /policy-desk-assistant/llm-judge/run/{run_id}` (demo/operator override; bypasses sampling)

Environment flags:

- `OPENAI_API_KEY` enables judge calls when present
- `LLM_JUDGE_ENABLED=true|false` (defaults to enabled if `OPENAI_API_KEY` is present)
- `OPENAI_JUDGE_MODEL=...` (defaults to `gpt-5-mini`)

## Review → Offline Eval Export

To demonstrate how human review becomes future test coverage, the repo supports exporting “promoted” review items into a portable offline eval JSON file:

- `GET /policy-desk-assistant/offline-eval/export`

This export:

- redacts obvious PII in the question (demo-grade, not a production DLP system)
- includes `reference_response_text` for human context
- sets `expected_behavior` based on `should_have_outcome`

The offline gates do not score free-form reference text today; they score the labeled behavior expectations and other deterministic checks.

## Streamlit UI

The Streamlit UI is intentionally small and operator-oriented:

- Inference (create a run + quick thumbs up/down feedback)
- Run Explorer (inspect stored runs)
- Offline Gates (single-run and baseline-vs-candidate comparisons)
- Review Queue (reviewer correction + offline export)
- Online Control (rolling metrics + alert panel + judge slice views)

Code structure:

- `ui/src/streamlit_app.py`: routing only
- `ui/src/pages/*.py`: one module per page
- `ui/src/ui_api.py`, `ui/src/ui_config.py`, `ui/src/ui_utils.py`: shared helpers

## Non-Goals

This repo intentionally does not implement:

- an experiment router / traffic splitting system
- real rollback (traffic switching)
- external queues (Kafka/Redis/etc.)
- semantic drift detection / embeddings
- production-grade PII detection and redaction

