# Indexing and Precompute

Last reviewed: 2026-04-05

## What "Indexing" Means Here

Indexing refers to precomputed DINO feature lookup via `index.json` and shard `.pt` files, not database indexing.

## Producer Path

- Script: `src/lbp_project/data/precompute_dino.py`
- Writes:
  - `{split}_shard_{id}.pt`
  - `index.json`
- Index schema root key: `samples` grouped by split.

## Consumer Paths

- `src/lbp_project/data/dataset.py`
  - `PrecomputedFeatureStore` resolves `split + sample_id` to shard entry.
- `src/lbp_project/models/wrapper.py`
  - uses `precomputed_dino` tensor in forward pass when provided.

## Config Toggles

- `data.use_precomputed_dino`
- `data.precomputed_index_path`
- `evaluation.real_use_precomputed_dino`
- `evaluation.real_precomputed_index_path`

## Local vs Server Behavior

- Local configs currently disable precomputed DINO by default.
- Server configs enable precomputed DINO for train/val path and typically disable for real tuple eval.
- Server index paths point to mounted locations that may be unavailable on local machines.

## Practical Checkpoints

1. Index file exists at configured path.
2. Shard paths inside index exist and are readable.
3. Index contains expected split keys and sample coverage.
