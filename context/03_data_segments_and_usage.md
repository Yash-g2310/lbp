# Data Segments and Usage

Last reviewed: 2026-04-05
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
