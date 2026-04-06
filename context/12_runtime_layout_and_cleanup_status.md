# Runtime Layout And Cleanup Status

Last reviewed: 2026-04-06

## Final Layout Decision

- Runtime operational outputs live under `runs/current/`.
- Curated outputs stay under `artifacts/` (notably `artifacts/insights/`).
- Legacy `project_root/` is removed from active repository layout.

## Completed Changes

1. Config runtime paths moved to `runs/current` for checkpoints and reports.
2. Slurm output/error defaults moved to `runs/current/logs`.
3. Quickcheck and eval wrappers aligned to `runs/current` output paths.
4. W&B local output directory aligned to `runs/current/wandb`.
5. Ignore rules extended for generated runtime artifact folders:
   - `runs/`
   - `artifacts/reports/`
   - `artifacts/archive/`
   - `artifacts/tmp/`

## Validation Snapshot

Completed in this migration cycle:

1. Config schema checks for active local and server profiles.
2. Server preflight checks for train, quickcheck, and eval templates.
3. Slurm dry-run checks with `runs/current/logs` output targets.
4. End-to-end quickcheck run using server quickcheck profile.
5. Lightweight training sanity run (`check_train.py` with one train step).
6. Post-delete checks confirming no active dependency on `project_root/`.

## Important Caveat

- Local quickcheck config on this host may fail due cache path permissions under `/home/yash_gupta/...`.
- Server quickcheck config is the portable verification path on this machine.

## Use This File When

- You need the shortest migration completion summary.
- You want to verify that runtime output design and cleanup decisions are already applied.