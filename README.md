# llm-eval-ops

LLM Eval Ops is a clean-room reference app for eval-driven LLM operations: runtime checks, a review loop, and offline regression gates.

It answers synthetic policy questions over a synthetic knowledge base, validates structured outputs, scores every runtime response, persists runs/reviews to SQLite, and ships a Streamlit operator UI.

<p align="center">
  <img src="docs/medium_1.png" alt="Offline evals decide what should ship before release, online evals decide what should be trusted in live traffic, and reviewer feedback becomes future test coverage." width="980">
</p>

For deeper implementation notes, see `docs/architecture.md`.

## What This Repo Demonstrates

- evidence-bound answering with strict structured output validation (+ one repair attempt)
- deterministic runtime checks with suspicious flags and risk bands on every run
- async LLM-as-judge semantic review on a sampled slice (100% medium/high-risk, 20% low-risk; deterministic by `run_id`)
- a human review queue fed by judge + side-channel signals (feedback + aggregate alerts), with dedupe by `run_id`
- offline evals (single-run scoring and baseline-vs-candidate comparison) using the same runtime pipeline
- an “online control plane” summary view: rolling metrics + threshold-based alerts + an async OpenAI judge layer

## Quickstart

Run the API:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
python -m uvicorn main:app --reload --app-dir api/src
```

Run the UI:

```bash
streamlit run ui/src/streamlit_app.py
```

Run tests:

```bash
pytest
```

## Key Flows

### Architecture (Online Eval Funnel)

Main per-run path (never blocks on semantic judging):

- request → retrieval → model call → validation/repair → deterministic checks → risk band → persist run → return response → async semantic review (sampled) → human review only if judge/side-channels route it → promote reviewed failures into offline eval cases

Side-channel signals (control plane):

- thumbs-down feedback can confirm/escalate risky runs
- aggregate online summary alerts can surface drift and create review work
- reviewers can annotate and promote cases into offline tests

Implementation entry points:

- Deterministic runtime checks: `api/src/services/metrics_service.py` (`OnlineScoringService.evaluate`)
- Semantic review (LLM-as-judge): `api/src/services/llm_judge_service.py` (`OpenAIJudgeService`)
- Central review routing/dedupe: `api/src/services/human_review_router.py` (`HumanReviewRouter`)
- Full walkthrough + diagram: `docs/architecture.md`

Offline regression gates:

- Dataset lives in `data/eval_dataset.json`.
- Gate policy lives in `data/offline_gate_policy.json`.
- `case_set=portable` is the core release suite; `case_set=full` adds mock-only stress cases (API default is `full`).

Online control plane:

- Operator feedback uses `POST /policy-desk-assistant/feedback` (currently `thumbs_up` / `thumbs_down`).
- Rolling summary + alert evaluation comes from `GET /policy-desk-assistant/feedback/summary`.
- Alert thresholds live in `data/online_alert_policy.json`.
- `rollback_recommended` is only an operator recommendation signal (no traffic switching).

Async LLM judge:

- Runs already routed to human review can still be judged; the judge signal is merged into the existing review item (dedupe by `run_id`).
- Sampling is deterministic by `run_id`: 20% of low-risk, and 100% of medium/high-risk runs.
- The judge scores `supportedness`, `policy_alignment`, and `response_mode`, and can enqueue human review with source `llm_judge`.

Review → offline eval export:

- Review items can be marked “Add to offline eval export (PII-redacted)”.
- Export is `GET /policy-desk-assistant/offline-eval/export` (portable case skeleton; the offline gates do not score free-form reference text today).

## Example Commands

Create a run:

```bash
curl -X POST "http://127.0.0.1:8000/policy-desk-assistant/respond" -H "Content-Type: application/json" -d "{\"question\":\"How long does a customer have to request a refund?\",\"scenario\":\"normal\",\"model_backend\":\"mock\",\"prompt_version\":\"qa-prompt:v1\"}"
```

Attach thumbs-down feedback:

```bash
curl -X POST "http://127.0.0.1:8000/policy-desk-assistant/feedback" -H "Content-Type: application/json" -d "{\"run_id\":\"{run_id}\",\"event_type\":\"thumbs_down\"}"
```

Fetch the rolling online summary + alerts:

```bash
curl "http://127.0.0.1:8000/policy-desk-assistant/feedback/summary?limit=50"
```

Fetch judge records:

```bash
curl "http://127.0.0.1:8000/policy-desk-assistant/llm-judge?limit=20"
```

Force a judge run (bypasses sampling):

```bash
curl -X POST "http://127.0.0.1:8000/policy-desk-assistant/llm-judge/run/{run_id}"
```

Download promoted offline eval cases:

```bash
curl "http://127.0.0.1:8000/policy-desk-assistant/offline-eval/export"
```

## Optional OpenAI Backend

```bash
pip install -e .[openai]
```

Environment variables:

- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=...` (runtime inference backend)
- `OPENAI_JUDGE_MODEL=...` (judge; default is `gpt-5-mini`)
- `LLM_JUDGE_ENABLED=true|false` (defaults to enabled if `OPENAI_API_KEY` is present)

## Main Endpoints

- `POST /policy-desk-assistant/respond`
- `GET /policy-desk-assistant/runs`
- `GET /policy-desk-assistant/runs/{run_id}`
- `POST /policy-desk-assistant/feedback`
- `GET /policy-desk-assistant/feedback`
- `GET /policy-desk-assistant/feedback/summary`
- `GET /policy-desk-assistant/llm-judge`
- `GET /policy-desk-assistant/llm-judge/{judge_id}`
- `POST /policy-desk-assistant/llm-judge/run/{run_id}`
- `GET /policy-desk-assistant/review-queue`
- `POST /policy-desk-assistant/review-queue/{item_id}/annotate`
- `POST /policy-desk-assistant/review-queue/prune`
- `GET /policy-desk-assistant/offline-eval/export`
- `GET /policy-desk-assistant/evals/offline`
- `GET /policy-desk-assistant/evals/offline/{eval_run_id}`
- `POST /policy-desk-assistant/evals/offline/compare`
- `GET /policy-desk-assistant/evals/offline/comparisons/{comparison_id}`

## Limitations (Intentional)

- retrieval is deterministic lexical matching, not vector search
- traces are local summaries, not a distributed tracing backend
- review queue is intentionally minimal and SQLite-backed
- online alerting is threshold-based, not a streaming observability system
- rollback is only a recommendation signal and never performs traffic switching
