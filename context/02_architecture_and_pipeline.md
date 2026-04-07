# Architecture and Pipeline

Last reviewed: 2026-04-06

## Model Path

- Wrapper: `src/lbp_project/models/wrapper.py`
- Decoder core: `src/lbp_project/models/unet.py`
- DINO source:
  - On-the-fly via `torch.hub.load(...)` when precompute is disabled.
  - Bypass path via `forward(..., precomputed_dino=...)` when precompute is enabled.

## Current Baseline (Non-PIHAF) Implementation

- Active wrapper class: `DINOSFIN_Architecture_NEW`.
- Conditioning strategy is prompt-concat with fused channels at decoder input:
  - RGB image channels: 3
  - DINO features: 384
  - Layer prompt map: 1
  - Total fused input channels: 388
- Layer conditioning is global per pass using `target_layer` prompt routing.
- Training uses two forward passes per sample (`target_layer=1` and `target_layer=2`) and supervises both outputs.
- Encoder/decoder update strategy:
  - DINO encoder frozen.
  - SFIN/RHAG U-Net decoder trainable.

## Training Objective and Optimization

- Current supervised loss: `SILogLoss` (`src/lbp_project/utils/losses.py`).
- Multi-stage supervision uses bottleneck, decoder, and final outputs per target layer.
- Optimizer: AdamW.
- Scheduler: cosine annealing (config-driven), not OneCycle.
- Curriculum weighting decays decoder and bottleneck auxiliary terms after midpoint.

## Data Path

- Loader: `src/lbp_project/data/dataset.py`
- Train/val loaders use:
  - `data.train_dataset_name`, `data.train_split`
  - `data.val_dataset_name`, `data.val_split`
- Real tuple eval uses `scripts/eval/eval_real_tuples.py` with split/layer fallback logic.

### Supervision Split Contract

- Dense supervised depth training/validation: `princeton-vl/LayeredDepth-Syn`.
- Real benchmark tuple evaluation: `princeton-vl/LayeredDepth` only.
- Real tuple data is not used for dense supervised training losses.

## Execution Path

- CLI entry: `src/lbp_project/cli.py`
- Commands dispatch to:
  - `train.py`
  - `scripts/training/train_eval.py`
  - `scripts/quickcheck/run.sh`
  - `scripts/eval/eval_real_tuples.py`
  - `scripts/eval/eval_checkpoints.py`

## Tuple Evaluation Semantics

- Tuple annotations are loaded from `tuples.json` with layer roots such as `layer_all` and `layer_first`.
- Accuracy is reported for:
  - pairs (P)
  - trips (T)
  - quads (Q)
- Correctness is computed from pairwise relative-depth ordering constraints among tuple points.
- Real eval scripts support split/layer-key fallback to avoid empty-report failure modes.

## Why This Matters

The project has two evaluation modalities:

1. Synthetic dense-depth validation (numeric depth metrics).
2. Real tuple ranking evaluation (pair/trip/quad ordering logic).

These should not be mixed when reasoning about model quality.
