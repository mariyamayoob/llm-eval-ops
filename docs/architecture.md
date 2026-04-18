# Architecture

## System overview

`llm-eval-ops` is a bounded LLM evaluation reference system rather than a generic chatbot. It answers synthetic policy questions over a synthetic knowledge base, validates structured outputs, scores every runtime response, routes risky cases to review, persists run records, and supports offline regression gating over labeled scenarios.

Backend code lives under `api/src/`. The Streamlit demo UI lives under `ui/src/streamlit_app.py`.

The codebase is intentionally organized around three practical questions:

1. Is the output valid and supported by retrieved evidence?
2. Did the system choose the expected response?
3. Did the candidate regress versus baseline?

## Why this system exists

The point of this repo is to show that LLM applications should be evaluated in two different ways:

- `online evals` run on every request and help decide whether a response should be trusted right now
- `offline evals` run over a labeled synthetic dataset and help detect regressions after prompt, model, retrieval, or knowledge-base changes

Together, they give the system a more production-like posture:

- online evals help contain risky runtime behavior
- offline evals help compare versions and prevent silent regressions
- reviewer outcomes can later be turned into new eval cases or prompt improvements

## Runtime outcomes

| Outcome | Meaning | Routing condition |
| --- | --- | --- |
| `supported_answer` | The system found enough evidence and no review trigger fired | grounded answer, valid structure, no critical flags |
| `refused_more_evidence_needed` | The system could not answer safely | retrieval miss, conflicting evidence, or validated refusal |
| `human_review_recommended` | The result was suspicious or intentionally escalated | critical flag, low online score, or boundary-sensitive behavior |

## Request lifecycle

1. `main.py` receives `POST /policy-desk-assistant/respond`.
2. `PolicyDeskService` creates a `run_id`, `trace_id`, and question hash.
3. `RetrievalService` performs deterministic lexical retrieval over `data/kb.json`.
4. `ModelService` calls either the mock adapter or the optional OpenAI adapter.
5. `ValidationService` parses output, attempts one repair if needed, checks citations and invariants, and records failure signals.
6. `OnlineScoringService` computes per-run scores and suspicious flags.
7. Review routing decides whether to create a review queue item.
8. `StorageService` writes the run record and any review item to SQLite.
9. `TraceRecorder` finalizes the step-level trace.

`retrieval_config_version` and `source_snapshot_id` flow through the request, response, stored run record, offline summaries, and comparison summaries. In the current demo they are traceability tags rather than active UI controls, and they do not switch between different KB files.

## Core modules

- `api/src/model/`: request, response, scoring, run, and review contracts
- `api/src/services/`: retrieval, model adapters, validation, scoring, storage, and orchestration
- `api/src/observability/`: root traces and step-level spans
- `api/src/prompts/`: two demo prompt profiles and the repair prompt
- `api/src/evals/contracts.py`: eval dataset schema, run summaries, comparison summaries, and behavior taxonomy
- `api/src/evals/scorers.py`: per-case offline scoring and blocker detection
- `api/src/evals/gates.py`: gate policy loader, threshold resolution, weighted scoring, and hard-fail logic
- `api/src/evals/runner.py`: single-run and comparison execution paths

## Online scoring

Online scoring runs on every request. It now emphasizes the repo's core trust story:

- valid structure and citations
- grounded retrieval-backed content
- policy and behavior safety signals

It still computes tone and brand heuristics, but those are advisory rather than primary release blockers.

Review routing is also intentionally separate from behavior scoring:

- `review_required` means "a human may want to inspect this"
- it does not automatically mean the system chose `human_review` as its behavior
- safe abstains remain abstains in the core offline gate story

Online scoring produces:

- `online_score_total`
- `risk_band`
- `suspicious_flags`
- `review_required`

This is the runtime control layer. Its job is triage, not perfect judgment.

## Offline eval dataset model

Offline eval cases are no longer just "question plus expected refusal boolean." Each case now carries enough metadata to support real regression gates:

- `bucket_id` and `bucket_name`
- `risk_tier`
- `business_criticality`
- `expected_behavior`
- `refusal_reason_expected`
- `retrieval_config_version`
- `source_snapshot_id`
- `label_notes`
- `gate_group`
- `owner`
- `case_kind`
- `supported_backends`

Legacy dataset rows still load because the case contract backfills missing fields from the older shape.

Two case groupings matter in practice:

- `portable`: real end-to-end cases that both backends can run
- `stress`: synthetic mock-only cases used to exercise failure modes like conflicting evidence, malformed output, or over-refusal

The current dataset buckets are:

- `direct-answerable`: normal answerable cases plus stress checks for wrong refusal and malformed output
- `missing-evidence-abstain`: high-risk abstain behavior when retrieval is empty
- `conflicting-evidence-refuse`: mock-only refusal behavior when evidence conflicts
- `policy-boundary-escalation`: mock-only human-review behavior for ambiguous exception handling
- `unsupported-claim-trap`: critical mock-only cases that hard-fail on unsupported claims
- `tone-brand`: low-risk advisory quality coverage

## Behavior taxonomy and blocker model

Offline scoring now distinguishes the expected and actual behavior of a run:

- `answer`
- `abstain`
- `clarify`
- `refuse`
- `human_review`

Each case result stores:

- `expected_behavior`
- `actual_behavior`
- `behavior_match`
- `regression_blockers`

Important blocker examples include:

- `unsafe_compliance`
- `over_refusal`
- `false_clarify`
- `unnecessary_escalation`
- `missed_human_review`
- `unsupported_claim`

This matters because a safe abstain and an unsafe answer should not be treated as equivalent just because they happen to land on the same average score.

For the simplified public-repo story, the core gate treats `refuse` and `abstain` as the same no-answer behavior class. The exact refusal subtype can still be inspected, but it is not the main release contract.

Not every detected regression becomes a blocking comparison failure. The scorer records a wider set of blockers for inspection, while the gate policy decides which blocker dimensions count as release-blocking in a given bucket or risk tier.

## Offline single-run evals

Single-run offline evals are triggered through `GET /policy-desk-assistant/evals/offline`.

The runner:

- loads labeled synthetic cases from `data/eval_dataset.json`
- executes them through the same runtime pipeline
- scores output validity and evidence support
- checks whether the system chose the expected response for the case
- records unsupported claims and other blockers
- computes weighted case scores using the gate policy
- aggregates by scenario and by bucket
- applies bucket-specific thresholds and hard-fail rules
- stores eval summaries and case results in SQLite

The single-run output answers: "Should this candidate clear the configured release gate on its own?"

The `case_set` input controls whether the runner evaluates:

- `portable`: only the real end-to-end cases
- `full`: portable cases plus mock-only stress cases

The Streamlit UI surfaces these as `Core release suite` and `Stress demo suite`. The API default remains `full`, but the demoable release-gate path typically uses `portable`.

The summary payload now includes:

- `aggregate_metrics` with `valid_grounded_score`, `behavior_score`, `advisory_quality_score`, `weighted_overall`, and `pass_rate`
- `by_bucket_breakdown` with per-bucket thresholds, blocker counts, and release decisions
- `failure_taxonomy_counts` and `behavior_taxonomy_counts`
- `worst_case_ids` and `skipped_cases`

The Streamlit single-run screen is driven directly from that payload:

- `render_story_cards()` turns `valid_grounded_score`, `behavior_score`, and `pass_rate` into the three top-line cards
- the middle card caption is derived from bucket-level `behavior_match_rate * case_count`, which is why the UI shows counts such as `4/4 cases matched answer vs abstain/refuse/escalate`
- the right card caption is derived from bucket-level `pass_rate * case_count`, which is why the UI shows counts such as `4/4 cases cleared thresholds with no blocking issues`
- `render_release_banner()` then shows a separate `Release decision` metric and any `decision_reasons`
- the page follows with `Core gate metrics`, `Advisory quality signals`, `Bucket gates`, `Worst cases`, and `All cases`

## Offline comparison evals

Comparison evals are triggered through `POST /policy-desk-assistant/evals/offline/compare`.

The comparison runner:

- runs a baseline config and a candidate config over the same dataset
- requires both configs to use the same `case_set`
- narrows comparison to the common supported cases when the two backends do not support the same stress cases
- persists both underlying single-run evals
- computes per-case deltas and bucket deltas
- identifies new failures introduced by the candidate
- identifies failures fixed by the candidate
- produces a final release decision with reasons

This output answers: "What got worse, what got better, and is the candidate safe to ship?"

This avoids falsely counting backend-specific stress cases as regressions when comparing a portable OpenAI candidate against the mock baseline.

The comparison summary also makes scope explicit through:

- `compared_case_ids`
- `excluded_case_ids`

Advisory-only regressions can still appear in score deltas or worst-regression lists, but they do not count as `new_blocking_failures` unless the gate policy marks that blocker dimension as release-blocking.

In the current UI, single-run offline results are summarized with three cards:

- `Output validity & support`
- `Expected response choice`
- `Gate pass rate`

Comparison results then add:

- a four-card regression header for `Release decision`, `New failures`, `Fixed failures`, and `New blockers`
- baseline and candidate story cards side by side
- `Core deltas`
- `Bucket regressions`
- `New failures`
- `Worst regressions`

That makes the comparison screen readable as both an engineering tool and a demo surface.

The prompt layer is intentionally minimal for the public demo:

- `qa-prompt:v1` is the stable baseline
- `qa-prompt:v2` is the new candidate

Those prompt profiles are defined once in `api/src/prompts/registry.py`.

The mock adapter reads the same structured profile fields that are passed to the OpenAI adapter, so the implementation stays readable and avoids hidden prompt-version tricks spread across the codebase.

## Gate policy design

The gate policy lives in `data/offline_gate_policy.json`.

It supports:

- global defaults
- risk-tier-specific thresholds
- bucket-specific thresholds
- weighted dimensions
- blocker dimensions
- hard-fail rules

The policy is intentionally lightweight JSON plus Pydantic validation. The goal is readability and inspectability, not a full policy engine.

The current weighted dimensions are:

- `citation_valid`
- `retrieval_hit`
- `behavior_match`
- `answer_fact_match`
- `unsupported_claim_penalty`
- `policy_adherence_match`

The default blocker dimensions are:

- `behavior_mismatch`
- `unsafe_compliance`
- `unsupported_claim`

Bucket and risk overlays then tighten those defaults for specific cases such as `policy-boundary-escalation` and `unsupported-claim-trap`.

The current hard-fail rules are intentionally small and explicit:

- `critical_any_blocker`
- `unsupported_claim_hard_fail`
- `safety_boundary_hard_fail`

Tone and brand signals remain available for inspection, but they are not meant to dominate the release story.

That separation is one of the stronger product qualities in the code right now: advisory quality metrics still show up in single-run and comparison views, but they do not automatically turn into blocking regressions unless the gate policy says they should.

## Why averages are not enough

Averages remain useful, but they are not the release contract.

The release decision now combines:

- weighted score thresholds
- bucket pass-rate thresholds
- behavior-match thresholds
- blocker counts
- hard-fail rules for critical buckets
- new failures in comparison mode
- new blocking failures in comparison mode

This means score deltas remain informative, but the actual release verdict is driven by bucket thresholds and blocker policy rather than by a single average or by advisory quality drift alone.

This is what makes the system behave like a real regression gate rather than a single flat score demo.

The important simplification is that the public-facing explanation stays small even though the gate has policy detail underneath it:

- output validity and support
- expected response choice
- gate pass rate for single-run views

## Release decision categories

Both single-run and comparison evals end in one of:

- `pass`
- `warn`
- `fail`

Decision payloads include:

- `decision_reasons`
- `failed_buckets`
- `new_failures`
- `new_blocking_failures`
- `fixed_failures`

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
- offline comparison summaries and case deltas
- review queue items
- reviewer annotations

Raw question text is not stored in run records by default.

## Offline evals vs online scoring

| Dimension | Offline evals | Online scoring |
| --- | --- | --- |
| Purpose | regression verification across labeled cases | per-run triage and review routing |
| Input | synthetic eval dataset | one live request |
| Output | aggregate metrics, bucket gates, and comparison deltas | score breakdown, risk band, suspicious flags |
| Best use | compare changes across versions | inspect runtime quality and risk |

## Tradeoffs

- retrieval is lexical and deterministic by design
- traces are local summaries, not an external observability backend
- Streamlit is used only for demo and inspection
- reviewer annotations are not yet auto-promoted into prompt or dataset changes
- the OpenAI backend is optional and not used in tests
- the mock backend still powers synthetic stress cases, but prompt-specific outcome cheats have been removed from the mock adapter
