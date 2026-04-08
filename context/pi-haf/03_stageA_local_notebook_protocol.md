# Stage A Local Notebook Protocol

Last reviewed: 2026-04-08
Scope: Error-proof local bring-up on limited local shards before server execution.

## Goal

Validate end-to-end correctness and learning behavior on local setup in fixed 5 epochs, with strict observability.

## Implemented Workflow Hook (2026-04-07)

1. Stage A policy workflow is now available via CLI:
- `python cli.py stage-a --config configs/local/dev.yaml --epochs 5`
2. This workflow applies stage-policy runtime config generation and strict checks for:
- fixed 5-epoch window,
- end-of-epoch periodic real eval cadence,
- run-manifest metadata emission.
3. Existing `train`/`train-eval` commands remain backward compatible.

## Hardware-Aware Local Guidance (RTX 4060 8GB, Range-Based)

1. Batch size: keep within low-memory range (for example 1-4) based on stability.
2. Gradient accumulation: use a small range (for example 1-8) to reach effective batch targets without OOM.
3. Input sizing: prefer conservative resolution/crop ranges during Stage A bring-up.
4. Precision: mixed precision is allowed when finite-loss and gradient checks remain healthy.
5. If OOM or instability appears, reduce batch/resolution first before changing objective schedule.

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

## Stage A Epoch Plan (Fixed 5 Epoch)

1. Epoch window:
- Run exactly 5 epochs (fixed before run starts).

2. Loss and validation monitoring requirement:
- All components must be tracked in logs each epoch, even if staged weighting applies by epoch.
- Validation metrics must be logged at end-of-epoch for every Stage A epoch.
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
- Per-image required-layer expansion with deduped layer inference across tuple points.

Mandatory logs:

1. Required layer IDs observed per sample.
2. Predicted layer IDs generated per sample.
3. Missing-layer tuple count.
4. Tuple totals used for scoring (`pairs`, `trips`, `quads`).

Acceptance expectation:
- Missing-layer tuple count should be zero under auto-expand policy.

## Stage A Acceptance Gate (Strict)

All conditions must hold:

1. No runtime errors (shape/schema/index/IO).
2. All monitored loss components remain finite.
3. Gradients remain finite and non-zero.
4. Local tuple metrics are non-zero and show improving trend over the fixed 5-epoch window.
5. End-of-epoch validation logs are present for all Stage A epochs.

## Migration Boundary Reminder

1. Current runtime still contains mixed legacy/flow-capable paths by profile.
2. The PI-HAF flow target contract (inverse-depth `[-1,1]` space and empty-space rule) is approved and must be treated as required migration work before claiming full Stage A parity.

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
