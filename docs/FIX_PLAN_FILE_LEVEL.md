# Coastpilot full file-level remediation plan

## Goal
Ship a reproducible, cost-controlled coastline-surface classification pipeline with high label coverage and deterministic artifacts.

## Current blockers (verified)
-  skips cached imagery candidates instead of classifying them.
- AI progress stalls around 20 labels out of 22,149.
- No git history in app dir (fixed in this repo bootstrap).
- Manifest/details inconsistent across runs.

## File-level implementation plan

### 1)  (P0 blocker)
- Remove logic that s when cached image exists.
- Candidate selection should:
  - include rows where  OR  missing;
  - prefer cached images first (for speed), but still classify;
  - preserve  behavior.
- Add deterministic batch ordering by .
- Add run counters in :
  - 
  - 
  - 
  - 
  - 
- Guarantee  only when classification returned valid label/confidence.

### 2)  (P0 cost + correctness)
- Add strict structured output contract:
  - required keys: , , .
  - label constrained to: .
- Add response validation + fallback path:
  - if parse fails: return  + reason.
- Add token usage accumulator writer hardening:
  - include request id, latency, retries, and model per call.
- Add per-call timeout + retry budget with exponential backoff.

### 3)  (P1 provider routing)
- Make provider selection explicit and auditable:
  - env + cli precedence documented.
- Add dedicated providers:
  -  (current style)
  - 
  - 
- Reject ambiguous config at startup with clear error.

### 4)  (P1 tunables)
- Add fields:
  - 
  - 
  - 
  - 
  - 
- Keep defaults backward-compatible.

### 5)  (P1 orchestration)
- Add  and  options.
- Add  for explicit artifact targeting.
- Persist run summary section:
  - total segments
  - ai labeled total
  - newly labeled this run
  - remaining unlabeled
  - estimated token cost

### 6)  (P2 quality)
- Preserve prior  if EU match quality is weak.
- Add thresholded EU confidence calibration into metadata.

### 7) New:  (P1)
- Compute quality KPIs:
  - label coverage %
  - class distribution
  - high/low confidence bands
  - unknown ratio
- Emit .

### 8) New tests
- 
  - cached-image candidate must still be classified.
  -  transition rules.
- 
  - malformed model output handling.
  - label normalization and confidence constraints.
- 
  - required manifest keys and type checks.

### 9) New docs
- 
-  (batch execution + resume + troubleshooting)

## Rollout phases
1. P0 bugfix + tests
2. P1 manifest/cost instrumentation
3. P1 batch resume + QA metrics
4. P2 enrichment tuning

## Acceptance criteria
- AI-labeled coverage >95% on current 22,149 segments.
- Deterministic rerun reproducibility (same inputs => same outputs).
- Cost report generated every run.
- Test suite green.
