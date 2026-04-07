# Stage A Local Notebook Protocol

Last reviewed: 2026-04-07
Scope: Error-proof local bring-up on limited local shards before server execution.

## Goal

Validate end-to-end correctness and learning behavior on local setup in 4-5 epochs, with strict observability.

## Data Contract (Local)

1. Synthetic data:
- Use locally retained synthetic train subset (currently one train shard).
- Validation subset can be partial, but must pass schema checks.

2. Real data:
- Use retained reindexed real shards for tuple evaluation checks.
- Treat local shard counts as runtime availability, not dataset definition.

## Required Startup Logs

Before epoch 1, log:

1. Dataset identity and split names.
2. Observed sample counts per loaded split.
3. Available shard counts (if discoverable from cache path).
4. Required schema keys:
- Synthetic: `image.png`, `depth_1.png` (and discovered `depth_k` fields when present).
- Real: `image.png`, `tuples.json` roots (`layer_all`, `layer_first` when available).

## Stage A Epoch Plan (4-5 Epoch)

1. Epoch window:
- Run 4 or 5 epochs (fixed before run starts).

2. Loss monitoring requirement:
- All components must be tracked in logs, even if staged weighting applies by epoch.
- Minimum monitored set:
  - Flow
  - SSI
  - Wavelet edge
  - Ordinal
  - Total loss
  - Gradient norm

3. Gradient and optimizer health:
- Record finite/non-finite checks every step.
- Record scaler behavior if AMP is enabled.

## Local Tuple Layer Coverage Requirement (User-Selected)

Definition:
- For each tuple point `(x, y, layer_id)`, evaluation must have a predicted depth map for `layer_id`.

Policy:
- Auto-expand predictions to all layer IDs found in tuples for each evaluated image.

Mandatory logs:

1. Required layer IDs observed per sample.
2. Predicted layer IDs generated per sample.
3. Missing-layer tuple count.

Acceptance expectation:
- Missing-layer tuple count should be zero under auto-expand policy.

## Stage A Acceptance Gate (Strict)

All conditions must hold:

1. No runtime errors (shape/schema/index/IO).
2. All monitored loss components remain finite.
3. Gradients remain finite and non-zero.
4. Local tuple metrics are non-zero and show improving trend over the 4-5 epoch window.

## Required Outputs

1. Notebook artifact path and execution timestamp.
2. Config snapshot used for Stage A.
3. Local run summary:
- loss curves
- gradient health summary
- tuple metrics table with trend notes
4. Decision note: pass/fail for Stage B promotion.

## Failure Handling

If Stage A fails any gate:

1. Stop promotion to Stage B.
2. Record failure category:
- data contract
- backbone access/runtime
- model shape contract
- loss instability
- eval coverage/metric failure
3. Record minimal reproduction snippet and corrective action plan.
