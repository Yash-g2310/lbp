# Data Segments and Usage

Last reviewed: 2026-04-06
Authority: latest generated data-truth report + config and script trace.

## Four Key Segments

1. LayeredDepth-Syn train
2. LayeredDepth-Syn validation
3. LayeredDepth validation
4. LayeredDepth test

## Reconciliation Snapshot

| Segment | Remote parquet | Local arrow family | Local arrow present |
|---|---:|---:|---:|
| LayeredDepth-Syn train | 56 | 45 | 1 |
| LayeredDepth-Syn validation | 2 | 2 | 1 |
| LayeredDepth validation | 9 | 6 | 1 |
| LayeredDepth test | 31 | 20 | 20 |

Interpretation:

- Remote parquet count is source-repo metadata.
- Local arrow family is expected cache materialization shape.
- Local arrow present is what currently exists on disk.

## Schema Contract

| Dataset | Primary fields | Intended role |
|---|---|---|
| LayeredDepth-Syn | `image.png`, `depth_1.png` ... `depth_8.png`, `__key__` | dense supervised train/val |
| LayeredDepth | `image.png`, `tuples.json`, `__key__`, `__url__` | sparse tuple ranking eval |

`tuples.json` roots include `layer_all` and `layer_first`, each with `pairs`, `trips`, and `quads` annotations.

## Split Role Clarification

- Synthetic train (14.8k) is the supervised training source.
- Synthetic validation (500) is the dense-depth validation source.
- Real validation (300) is used for tuple-eval tuning checks and periodic reporting.
- Real test (1.2k) is used for final zero-shot tuple evaluation.

No real split is used as dense supervised loss target in the current training path.

## Local Cache Reality (Important)

- Local synthetic train currently has one materialized arrow shard.
- Local synthetic validation currently has one materialized arrow shard.
- Local real validation is partial.
- Local real test is largely materialized.

This mixed cache state is expected in current local debug workflow and should be treated as a local availability fact, not a dataset-definition change.

## Usage Matrix (4x4 View)

| Surface | Syn train | Syn val | Real val | Real test |
|---|---|---|---|---|
| `notebooks/experiment1.ipynb` | partial subset workflow | no stable path | yes | no |
| `notebooks/unet_connect.ipynb` | intended, timeout-prone | implicit | yes | no |
| Local scripts (`configs/local/*`) | yes | yes | yes | no |
| Server scripts (`configs/server/*`) | yes | yes | yes | yes |

## Where To Inspect In Code

- Config mapping: `configs/local/*.yaml`, `configs/server/*.yaml`
- Dataloaders: `src/lbp_project/data/dataset.py`
- Real tuple eval: `scripts/eval/eval_real_tuples.py`
- Checkpoint batch eval: `scripts/eval/eval_checkpoints.py`
