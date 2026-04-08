# Data Shards and Split Contract

Last reviewed: 2026-04-08
Scope: Canonical split definitions and runtime materialization rules for local and server stages.

## Canonical Dataset Definitions (Do Not Change)

1. LayeredDepth-Syn:
- Train: 14800
- Validation: 500

2. LayeredDepth (real tuples):
- Test: 1200
- Validation: 300

These definitions are authoritative regardless of local cache/shard reindex state.

## Runtime Materialization Rule

1. Local cache/shard counts may differ due to reindexing/materialization.
2. Local materialization state must be logged and treated as runtime availability only.
3. Do not rewrite canonical split definitions based on local shard layout.

## Stage A Local Contract

1. Synthetic training can use retained subset shards for bring-up.
2. Retained real shards are used for tuple evaluation sanity checks.
3. Mandatory startup logs:
- observed sample counts per split
- shard presence summary
- schema key checks
4. Local loader policy:
- load from available local Arrow shards when partial mode is enabled,
- do not trigger remote dataset shard downloads in local Stage A paths.

## Stage B Server Contract

1. Full split availability required before launch.
2. If precomputed features are enabled, index coverage must match all referenced samples.
3. Fail-fast on:
- missing split
- missing required keys
- index sample-ID misses
4. Server loader policy:
- partial-shard mode is disabled,
- full split materialization is required under server cache/staging roots.

## Required Schema Keys

### Synthetic

- `image.png`
- `depth_1.png` (required)
- optional additional `depth_k` fields

### Real

- `image.png`
- `tuples.json`
- tuple roots expected for reporting: `layer_all`, `layer_first` when available

## Precomputed Feature Index Contract

1. Index file exists and is readable.
2. Split entries exist for used splits.
3. Spot-check sample IDs resolve to valid shard paths.
4. Missing sample IDs are hard failures for precomputed mode.

## Data-Contract Risks

1. Silent confusion between canonical split definition and local cache state.
2. Partial real shard coverage causing skewed tuple metrics if not disclosed.
3. Precomputed index mismatch causing runtime failure late in run.

## Required Data-Contract Artifacts

1. Data truth snapshot per run.
2. Split/materialization summary in report header.
3. Precomputed index verification log if enabled.
