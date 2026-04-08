# Data Segments and Usage

Last reviewed: 2026-04-08
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

`tuples.json` roots include `layer_all` and `layer_first`, each exposing `pairs`, `trips`, and `quads` containers; population can vary by split/materialization.

## Preprocessing Contract Status

### Implemented now

1. Depth tensors are loaded from synthetic depth images, scaled by `depth_scale`, and clipped to configured finite ranges.
2. Invalid depth pixels (non-finite or `<= 0`) are preserved as invalid/zero-masked values in loader outputs.
3. Dynamic multilayer payloads (`depth_targets`, `depth_layer_mask`) are available for training/eval paths.
4. Empty-space contract is enforced in collation for front-layer supervision: when layer-1 is invalid and layer-2 is valid, preprocessing applies $L_1 = L_2$.
5. Flow-mode training path maps depth targets to inverse-depth normalized space in $[-1,1]$ before flow objective application.

### Approved target (immediate implementation requirement)

1. **Flow-space mapping**:
- Metric depth $d$ must be transformed before loss application:

$$
d_{inv} = \frac{1}{\max(d, \epsilon)}
$$

$$
d_{norm} = 2 \cdot \frac{d_{inv}}{d_{inv,max}} - 1
$$

- Target training space is bounded to $[-1,1]$.

2. **Migration note**:
- This depth-space migration is a retraining boundary.
- Existing positive-depth checkpoints should be treated as legacy and not promoted across this boundary.

## Split Role Clarification

- Synthetic train (14.8k) is the supervised training source.
- Synthetic validation (500) is the dense-depth validation source.
- Real validation (300) is used for tuple-eval tuning checks and periodic reporting.
- Real test (1.2k) is used for final zero-shot tuple evaluation.
- Real benchmark reporting uses tuple P/T/Q as primary metrics.

No real split is used as dense supervised loss target in the current training path.

## Local Cache Reality (Important)

- Local synthetic train currently has one materialized arrow shard.
- Local synthetic validation currently has one materialized arrow shard.
- Local real validation is partial.
- Local real test is largely materialized.

This mixed cache state is expected in current local debug workflow and should be treated as a local availability fact, not a dataset-definition change.

## Clash Notes To Preserve

1. Current loader behavior and target flow-space behavior are intentionally different during migration.
2. Inverse-depth normalization is active in flow-mode training; legacy metric-depth path still exists for backward compatibility.
3. Gate thresholds will require recalibration after migration because training-space statistics change.

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
