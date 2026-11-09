# CoastPilot

CoastPilot is a geospatial pipeline for coastline segmentation and shoreline surface classification.

It processes coastal geometry, enriches it, and applies AI vision classification to unresolved segments.

## What it does

1. Collects coastline and beach-related geometry.
2. Splits coastline into fixed-length segments.
3. Runs initial rule-based classification.
4. Enriches segments with external coastal reference data.
5. Runs AI vision classification for unresolved segments.
6. Produces versioned artifacts and manifest metadata.

Canonical surface labels used by AI vision and gold labels:
- rock
- pebble
- cobble
- sand
- built
- unknown

Strict rocky policy: only `rock` counts as rocky. `pebble` and `cobble` are separate non-rocky classes for evaluation and recommendation QA.

## Main features

- Deterministic segmentation and reproducible outputs.
- Cached imagery reuse for faster reruns.
- Configurable AI providers and models.
- Retry/backoff and structured-output validation for model responses.
- Run manifest with step-level details and summary.
- Unit tests for core AI pipeline behavior.

## Repository structure

- pipeline/
  - ai/: provider routing and model-call logic
  - steps/: pipeline execution steps
  - vision/: imagery resolve/download/cache helpers
  - qa/: quality metrics helpers
- run_pipeline.py: CLI entrypoint
- tests/: unit tests
- docs/: fix plan, quality review, spec review

## Requirements

- Python 3.11+
- Linux recommended
- Dependencies from requirements.txt
- API keys for token-billed providers (when using hosted models)

## Setup

1. cd /opt/apps/coastpilot
2. python3 -m venv .venv
3. source .venv/bin/activate
4. pip install -r requirements.txt

## Usage

Run full pipeline:
- python run_pipeline.py --step all

Run AI step only:
- python run_pipeline.py --step ai --vision-limit 100

Resume/stop controls:
- python run_pipeline.py --step all --resume-from eu --stop-after ai

Explicit manifest output:
- python run_pipeline.py --step all --manifest-path /tmp/coastpilot_manifest.json

Optional quality report:
- python run_pipeline.py --step classify --quality-report /tmp/coastpilot_quality.json

After a successful run, `--quality-report PATH` writes JSON metrics from
`data/segments_classified.geojson`: label coverage, unknown ratio, class
distribution, and confidence bands. The classified file must exist and the
output directory must already exist. Without the flag, no quality report is
generated.

## Review labeler

Start the standalone labeler with `python tools/labeler_app.py`. It binds to
`127.0.0.1:8787` by default; use an SSH local forward for remote review rather
than exposing the development server.

- `COASTPILOT_LABEL_HOST` overrides the bind address. Non-loopback addresses
  (including `0.0.0.0` and `::`) require a non-blank `COASTPILOT_LABEL_TOKEN`;
  otherwise startup exits before opening a listener.
- When a token is configured, every route requires it in the `token` query
  parameter or `X-Label-Token` header. Keep tokens out of source control and
  shared logs; use a private, high-entropy value and HTTPS for network access.
- The startup guard applies to the standalone script. If using an alternative
  WSGI launcher or reverse proxy, configure a token explicitly before making
  the app reachable beyond localhost; those launchers control their own bind.
- `COASTPILOT_LABEL_INPUT` and `COASTPILOT_LABEL_OUTPUT` select the CSV paths.

## Outputs

Typical outputs:
- data/segments.geojson
- data/segments_classified.geojson
- data/manifest_v<version>.json

Manifest includes:
- configuration context
- per-step records/details
- AI vision metadata
- run summary

## AI cost notes

Current token-cost path is hosted AI vision inference (for example Gemini/OpenAI-compatible endpoints).

The project includes:
- response contract validation
- retry/backoff
- timeout controls
- usage logging (gemini_usage.log when Gemini route is used)
- reduced default output-token settings

## Quality checks

- python -m compileall run_pipeline.py pipeline tests
- python -m unittest discover -s tests -v

## Current limitations

- Hard runtime budget kill-switch (token/USD) is not fully wired yet.
- Resume is step-level, not per-segment checkpoint-level.
- Full long-run benchmark depends on environment and provider limits.

## License

No license file is currently included.
Add one before public/commercial distribution.
