# Spec review updated

Goal alignment
- Pipeline stays on coastline segment labeling with canonical labels: rock, pebble, cobble, sand, built, unknown.
- AI step now classifies cached imagery too.
- Manifest now includes run_summary and supports explicit output path.

Implemented
- pipeline/steps/ai_vision.py: cached-image skip bug fixed, deterministic ordering, richer counters, safer needs_ai transition.
- pipeline/ai/providers.py: structured output validation, retry and backoff wrapper, request metadata (latency retries request_id), reduced output token default.
- pipeline/ai/factory.py: explicit provider routing and ambiguity guard, config-driven retry token image params.
- pipeline/config.py: added retry token budget image tunables.
- run_pipeline.py: resume stop controls, explicit manifest path, run summary section.
- pipeline/qa/metrics.py: quality KPI module.

Remaining gaps
- Hard budget enforcement (ai_budget_max_tokens or usd) not yet wired to runtime abort.
- QA JSON is not automatically emitted by run_pipeline yet.
- Resume semantics are step-level only, no per-segment checkpoint cursor.
- EU enrichment calibration safeguards still basic.

Gold-label policy
- Strict rocky is enforced: only rock is positive. Pebble/cobble are distinct non-rocky labels.
- Gold evaluator validates gold_label against the canonical set and reports invalid predictions.
