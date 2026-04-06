# Data Configuration Reference

This file is the source of truth for configured dataset usage across local and server workflows.

## Configured Split Mapping

| Workflow | Train | Validation | Real Eval |
|---|---|---|---|
| Local (`configs/local/dev.yaml`) | `princeton-vl/LayeredDepth-Syn:train` | `princeton-vl/LayeredDepth-Syn:validation` | `princeton-vl/LayeredDepth:validation` |
| Local quickcheck (`configs/local/quickcheck.yaml`) | `princeton-vl/LayeredDepth-Syn:train` | `princeton-vl/LayeredDepth-Syn:validation` | `princeton-vl/LayeredDepth:validation` |
| Server (`configs/server/default.yaml`) | `princeton-vl/LayeredDepth-Syn:train` | `princeton-vl/LayeredDepth-Syn:validation` | `princeton-vl/LayeredDepth:{validation,test}` |
| Server quickcheck (`configs/server/quickcheck.yaml`) | `princeton-vl/LayeredDepth-Syn:train` | `princeton-vl/LayeredDepth-Syn:validation` | `princeton-vl/LayeredDepth:{validation,test}` |

## Four-Segment Reconciliation Snapshot

Based on the latest data-truth run:

| Segment | Remote parquet | Local arrow family | Local arrow present | Status |
|---|---:|---:|---:|---|
| LayeredDepth-Syn train | 56 | 45 | 1 | local materialized partial |
| LayeredDepth-Syn validation | 2 | 2 | 1 | local materialized partial |
| LayeredDepth validation | 9 | 6 | 1 | local materialized partial |
| LayeredDepth test | 31 | 20 | 20 | local materialized complete |

Interpretation rule:

- Remote parquet counts and local arrow counts are different layers.
- Missing local shards mean not present locally now; this does not prove deletion history.

## Where Each Segment Is Used

| Execution surface | Syn train | Syn val | Real val | Real test |
|---|---|---|---|---|
| `notebooks/experiment1.ipynb` | partial (`.take(1000)` style workflow) | no stable direct val loop | yes | no |
| `notebooks/unet_connect.ipynb` | intended, but timeout-prone | implicit | yes | no |
| Local scripts (`configs/local/*`) | yes | yes | yes | no |
| Server scripts (`configs/server/*`) | yes | yes | yes | yes |

## Field Semantics

- Synthetic supervised path uses depth maps (`depth_1` and `depth_2`) from LayeredDepth-Syn.
- Real tuple evaluation uses ranking annotations in `tuples.json` (`layer_all`, `layer_first`) on LayeredDepth validation/test.

## Related References

- Reporting commands and artifact policy: `reporting_and_artifacts.md`
- Indexing/precompute details for future chat handoff: `../../context/04_indexing_precompute.md`
