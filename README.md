# llm-eval-ops

`llm-eval-ops` is the repository for **LLM Eval Ops**, a clean-room reference implementation for evaluating, tracing, and reviewing bounded LLM behavior in a production-like environment.

It is intentionally centered on evaluation, not chat UX. The system answers synthetic policy questions over a synthetic knowledge base, validates structured outputs, scores every runtime response, routes risky cases to review, and replays labeled scenarios through an offline eval harness.

## What it demonstrates

- evidence-bound answering over a synthetic knowledge base
- three explicit outcomes: `supported_answer`, `refused_more_evidence_needed`, `human_review_recommended`
- strict structured output validation with one repair attempt
- online scoring on every request for runtime triage
- offline regression-style evaluation over a labeled synthetic dataset
- trace-linked run inspection and SQLite persistence
- a minimal Streamlit UI for inference, run inspection, eval summaries, and review queue inspection

## Why online and offline evals both matter

`Online evals` run on every request. They help decide whether a response should be trusted right now by producing runtime scores, suspicious flags, a risk band, and review routing.

`Offline evals` run on demand over a labeled synthetic dataset. They help detect regressions after prompt, model, or code changes by replaying representative scenarios through the same runtime pipeline.

Together, they support a more production-like workflow:
- online evals protect runtime behavior
- offline evals validate system changes
- reviewer outcomes can later be promoted into new eval cases or prompt updates

## Review loop

The current implementation stores reviewer decisions and keeps them inspectable. It does not automatically rewrite prompts or datasets.

The intended next step is explicit and controlled:
- reviewed failures become new offline eval cases
- repeated corrections identify prompt weaknesses
- scoring rules can be tightened or relaxed based on reviewer outcomes
- prompt or scoring changes are revalidated by rerunning offline evals

## Repo structure

- `api/src/`: backend package
- `api/src/model/`: request, response, scoring, run, and review contracts
- `api/src/services/`: retrieval, model adapters, validation, scoring, storage, and orchestration
- `api/src/observability/`: local trace recorder and span summaries
- `api/src/prompts/`: versioned prompts and repair prompt
- `api/src/evals/`: offline eval contracts, runner, scorers, and review queue models
- `api/src/tests/`: pytest suite
- `ui/src/streamlit_app.py`: simple operator/demo UI
- `data/`: synthetic KB and offline eval dataset
- `docs/architecture.md`: implementation notes
- `certs/`: optional local certificate material for TLS interception environments

## Run FastAPI

```bash
If you renamed or moved the repo folder, delete and recreate `.venv` first because Windows virtualenv launchers embed absolute paths.

python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
python -m uvicorn main:app --reload --app-dir api/src
```

## Run Streamlit

```bash
streamlit run ui/src/streamlit_app.py
```

## Run tests

```bash
pytest
```

## Run offline evals

```bash
curl "http://127.0.0.1:8000/policy-desk-assistant/evals/offline?model_backend=mock&prompt_version=qa-prompt:v1"
```

## Optional OpenAI backend

```bash
pip install -e .[openai]
```

Then set:
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

## Environment variables

- `MODEL_BACKEND=mock|openai`
- `PROMPT_VERSION=qa-prompt:v1|qa-prompt:v2`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=...`
- `SSL_CERT_FILE=...`
- `REQUESTS_CA_BUNDLE=...`

The mock backend is the default and the only backend used in tests.

## Main endpoints

- `POST /policy-desk-assistant/respond`
- `GET /policy-desk-assistant/runs`
- `GET /policy-desk-assistant/runs/{run_id}`
- `GET /policy-desk-assistant/evals/offline`
- `GET /policy-desk-assistant/evals/offline/{eval_run_id}`
- `GET /policy-desk-assistant/review-queue`
- `POST /policy-desk-assistant/review-queue/{item_id}/annotate`

## Limitations

- retrieval is deterministic lexical matching, not vector search
- traces are local in-memory summaries, not a distributed tracing backend
- OpenAI integration is optional and not required for tests
- the review queue is intentionally minimal and SQLite-backed
- reviewer annotations are persisted, but not yet auto-promoted into prompt or dataset changes
