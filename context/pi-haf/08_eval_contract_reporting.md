# Evaluation Contract and Reporting

Last reviewed: 2026-04-08
Scope: Synthetic and real evaluation policy for Stage A and Stage B.

## Evaluation Modes

1. Synthetic dense depth evaluation.
2. Real tuple ranking evaluation (primary real benchmark gate via P/T/Q).

## Synthetic Evaluation Contract

1. Report per-layer and aggregate metrics.
2. Keep metric schema stable across local and server runs.
3. Disclose subset/full-data context in report metadata.

## Real Tuple Evaluation Contract

1. Evaluate on retained real test and validation materialization for local checks.
2. Use canonical full real splits for server reporting.
3. Evaluate tuple roots:
- `layer_all` (multi-layer benchmark subset)
- `layer_first` (first-layer benchmark subset when populated)

## Local Tuple Layer Coverage Requirement

Definition:
- Each tuple point references `(x, y, layer_id)`.
- A tuple is valid only when all required `layer_id` predictions are present for that sample.

Policy (locked):
- Per-image required-layer expansion with deduped layer inference across tuple points.

Reporting requirements:

1. Required layer IDs observed.
2. Predicted layer IDs generated.
3. Missing-layer tuple count.
4. Tuple totals used for scoring.
5. Tuple P/T/Q metrics (`pairs_acc`, `trips_acc`, `quads_acc`) as primary report fields.

Expected Stage A behavior:
- Missing-layer tuple count should be zero under auto-expand.

## Stage A Evaluation Gate

All of the following are required:

1. Tuple metrics are non-zero.
2. Tuple metrics show improving trend over fixed 5-epoch Stage A window.
3. No eval-time schema or layer-coverage contract violations.
4. Local validation reporting is present at end-of-epoch for Stage A.

## Stage B Evaluation Gate

1. Full synthetic and real reports are generated.
2. Reporting schema is consistent with Stage A.
3. Any skipped/missing tuples are explicitly accounted for.
4. Validation reporting cadence is every 10 epochs during Stage B.
5. Final-stop full real benchmark evaluation is mandatory at Stage B stop.

## Stage B Stop/Eval Runtime Policy (Approved Target)

1. Stage B uses dual cap: `min(24h, 30 epochs)`.
2. 25 epochs is preferred when budget and stability permit.
3. A final full real evaluation (1200-image contract) must run at stop regardless of stop trigger.

## Reporting Schema (Minimum)

1. Run metadata:
- date/time
- config hash
- backbone variant
- data materialization summary

Implementation note (2026-04-07): `scripts/eval/eval_synth_depth.py` and `scripts/eval/eval_real_tuples.py` now embed run metadata manifests in report payloads.

2. Synthetic section:
- per-layer metrics
- aggregate metrics

3. Real tuple section:
- layer key metrics
- tuple totals
- missing/coverage diagnostics

4. Decision section:
- pass/fail against gate criteria
- next action
