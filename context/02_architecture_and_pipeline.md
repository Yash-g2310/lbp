# Architecture and Pipeline

Last reviewed: 2026-04-05

## Model Path

- Wrapper: `src/lbp_project/models/wrapper.py`
- Decoder core: `src/lbp_project/models/unet.py`
- DINO source:
  - On-the-fly via `torch.hub.load(...)` when precompute is disabled.
  - Bypass path via `forward(..., precomputed_dino=...)` when precompute is enabled.

## Data Path

- Loader: `src/lbp_project/data/dataset.py`
- Train/val loaders use:
  - `data.train_dataset_name`, `data.train_split`
  - `data.val_dataset_name`, `data.val_split`
- Real tuple eval uses `scripts/eval/eval_real_tuples.py` with split/layer fallback logic.

## Execution Path

- CLI entry: `src/lbp_project/cli.py`
- Commands dispatch to:
  - `train.py`
  - `scripts/training/train_eval.py`
  - `scripts/quickcheck/run.sh`
  - `scripts/eval/eval_real_tuples.py`
  - `scripts/eval/eval_checkpoints.py`

## Why This Matters

The project has two evaluation modalities:

1. Synthetic dense-depth validation (numeric depth metrics).
2. Real tuple ranking evaluation (pair/trip/quad ordering logic).

These should not be mixed when reasoning about model quality.
