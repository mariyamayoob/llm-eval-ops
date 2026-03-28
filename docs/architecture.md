# Architecture

## System overview

`llm-eval-ops` is the repository for **LLM Eval Ops**, a bounded LLM evaluation reference system rather than a generic chatbot. It answers synthetic policy questions over a synthetic knowledge base, validates structured outputs, scores every runtime response, routes risky cases to review, persists run records, and supports offline evaluation over labeled scenarios.

Backend code lives under `api/src/`. The Streamlit demo UI lives under `ui/src/streamlit_app.py`.

## Why this system exists

The point of this repo is to show that LLM applications should be evaluated in two different ways:

- `online evals` run on every request and help decide whether a response should be trusted right now
- `offline evals` run over a labeled synthetic dataset and help detect regressions after prompt, model, or code changes

Together, they give the system a more production-like posture:

- online evals help contain risky runtime behavior
- offline evals help compare versions and prevent silent regressions
- reviewer outcomes can later be turned into new eval cases or prompt improvements

## The three outcomes

| Outcome | Meaning | Routing condition |
| --- | --- | --- |
| `supported_answer` | The system found enough evidence and no review trigger fired | grounded answer, valid structure, no critical flags |
| `refused_more_evidence_needed` | The system could not answer safely | retrieval miss, conflicting evidence, or validated refusal |
| `human_review_recommended` | The result was suspicious or low confidence | critical flag or low online score |

## Request lifecycle

1. `main.py` receives `POST /policy-desk-assistant/respond`.
2. `PolicyDeskService` creates a `run_id`, `trace_id`, and question hash.
3. `RetrievalService` performs deterministic lexical retrieval over `data/kb.json`.
4. `ModelService` calls either the mock adapter or optional OpenAI adapter.
5. `ValidationService` parses output, attempts one repair if needed, checks citations and invariants, and records failure signals.
6. `OnlineScoringService` computes per-run scores and suspicious flags.
7. Review routing decides whether to create a review queue item.
8. `StorageService` writes the run record and any review item to SQLite.
9. `TraceRecorder` finalizes the step-level trace.

## Core modules

- `api/src/model/`: request, response, scoring, run, and review contracts
- `api/src/services/`: retrieval, model adapters, validation, scoring, storage, and orchestration
- `api/src/observability/`: root traces and step-level spans
- `api/src/prompts/`: versioned prompts and repair prompt
- `api/src/evals/`: offline eval dataset contracts, runner, scorers, and review models

## Online scoring

Online scoring runs on every request. It computes groundedness, citation validity, policy adherence, tone, format validity, and retrieval support, then produces:

- `online_score_total`
- `risk_band`
- `suspicious_flags`
- `review_required`

This is the runtime control layer. Its job is triage, not perfect judgment.

## Offline evals

Offline evals are triggered through `GET /policy-desk-assistant/evals/offline`.

The runner:

- loads labeled synthetic cases from `data/eval_dataset.json`
- executes them through the same runtime pipeline
- scores schema validity, citation validity, retrieval hit, refusal correctness, answer fact match, unsupported claims, tone, and policy adherence
- stores eval summaries and case results in SQLite

This is the regression layer. It is used to compare prompt versions, model backends, or pipeline changes.

## Review loop

Suspicious runs are written to a review queue. Reviewers annotate them through `POST /policy-desk-assistant/review-queue/{item_id}/annotate`.

In the current implementation, reviewer outcomes are stored and inspectable, but they do not automatically change prompts or datasets. The intended future loop is explicit:

- reviewed failures become new offline eval cases
- repeated corrections identify prompt weaknesses
- updated prompts or scoring rules are validated by rerunning offline evals

## Tracing and storage

`TraceRecorder` creates a root trace and spans for:

- `retrieve`
- `prompt_build`
- `llm_call`
- `output_parse`
- `output_validate`
- `output_repair`
- `online_score`
- `review_route`
- `persist_run`

SQLite stores:

- run records
- offline eval summaries and case results
- review queue items
- reviewer annotations

Raw question text is not stored in run records by default.

## Offline evals vs online scoring

| Dimension | Offline evals | Online scoring |
| --- | --- | --- |
| Purpose | regression verification across labeled cases | per-run triage and review routing |
| Input | synthetic eval dataset | one live request |
| Output | aggregate metrics and case results | score breakdown, risk band, suspicious flags |
| Best use | compare changes across versions | inspect runtime quality and risk |

## Tradeoffs

- retrieval is lexical and deterministic by design
- traces are local summaries, not an external observability backend
- Streamlit is used only for demo and inspection
- reviewer annotations are not yet auto-promoted into prompt or dataset changes
- the OpenAI backend is optional and not used in tests
