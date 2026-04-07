# Notebook State

Last reviewed: 2026-04-06

## Status Summary

| Notebook | Status | Main Risk |
|---|---|---|
| `notebooks/experiment1.ipynb` | stale outputs, legacy assumptions | mixed local parquet + HF assumptions |
| `notebooks/unet_connect.ipynb` | stale outputs, legacy failed attempt traces | dataset streaming/timeouts and incomplete training path |

## Attribution Note

- The prior failed model effort referenced by notebook history was the legacy DINOv2 + SFIN + RHAG U-Net path.
- It should not be interpreted as a PI-HAF failure.

## Current Role

Treat notebooks as exploratory references, not as canonical production execution paths.

## Recommended Use

1. Validate schemas and sample keys quickly.
2. Avoid using notebook outputs as source of truth for experiment tracking.
3. Use script/CLI pipelines for reproducible runs.

## If Notebook Work Is Needed Again

- Re-validate dataset load assumptions from current configs.
- Prefer explicit split definitions over ad-hoc subsets.
- Keep notebook outputs untracked under current git policy.
