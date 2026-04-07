# Evaluation Contract and Reporting

Last reviewed: 2026-04-07
Scope: Synthetic and real evaluation policy for Stage A and Stage B.

## Evaluation Modes

1. Synthetic dense depth evaluation.
2. Real tuple ranking evaluation.

## Synthetic Evaluation Contract

1. Report per-layer and aggregate metrics.
2. Keep metric schema stable across local and server runs.
3. Disclose subset/full-data context in report metadata.

## Real Tuple Evaluation Contract

1. Evaluate on retained real test and validation materialization for local checks.
2. Use canonical full real splits for server reporting.
3. Evaluate tuple roots:
- `layer_all`
- `layer_first` (when available)

## Local Tuple Layer Coverage Requirement

Definition:
- Each tuple point references `(x, y, layer_id)`.
- A tuple is valid only when all required `layer_id` predictions are present for that sample.

Policy (locked):
- Auto-expand predictions to all layer IDs required by tuples per sample.

Reporting requirements:

1. Required layer IDs observed.
2. Predicted layer IDs generated.
3. Missing-layer tuple count.
4. Tuple totals used for scoring.

Expected Stage A behavior:
- Missing-layer tuple count should be zero under auto-expand.

## Stage A Evaluation Gate

All of the following are required:

1. Tuple metrics are non-zero.
2. Tuple metrics show improving trend over 4-5 epochs.
3. No eval-time schema or layer-coverage contract violations.

## Stage B Evaluation Gate

1. Full synthetic and real reports are generated.
2. Reporting schema is consistent with Stage A.
3. Any skipped/missing tuples are explicitly accounted for.

## Reporting Schema (Minimum)

1. Run metadata:
- date/time
- config hash
- backbone variant
- data materialization summary

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
