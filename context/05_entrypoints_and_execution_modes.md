# Entrypoints and Execution Modes

Last reviewed: 2026-04-08

## Unified CLI

File: `src/lbp_project/cli.py`

Commands:

- `train`
- `train-eval`
- `stage-a`
- `stage-b`
- `quickcheck`
- `eval-real`
- `eval-checkpoints`

## Stage Intent (Math Contract)

1. **Stage A**:
- Error-free bring-up and geometric warmup.
- Target objective: Flow + SSI.
- Target schedule: fixed 5 epochs.

2. **Stage B**:
- Physics lockdown and structural refinement.
- Target objective: Flow + SSI + Wavelet Edge + Ordinal.
- Target server policy: 25 epochs preferred, can extend up to 30 within runtime budget.

3. **Dual-cap stop policy (target)**:
- Hard stop at `min(24h, 30 epochs)`.
- Always run final full real benchmark evaluation at stop.

## Local Mode

Typical configs:

- `configs/local/dev.yaml`
- `configs/local/quickcheck.yaml`

Behavior:

- Real eval is validation-only by default.
- Precomputed DINO is disabled by default.
- Stage-A command path is implemented, but full PI-HAF flow-space migration is still pending.

## Server Mode

Typical configs:

- `configs/server/default.yaml`
- `configs/server/quickcheck.yaml`
- `configs/server/balanced.yaml`
- `configs/server/stable.yaml`

Behavior:

- Real eval includes validation and test by default.
- Precomputed DINO is enabled for training path in server profiles.
- Server profiles explicitly set `architecture.backbone_fallback_approved: false` to preserve strict no-silent-fallback policy.
- `stage-b` mode performs a promotion-gate precheck and requires prior Stage A evidence reports.
- Server preflight enforces `hardware.min_vram_gb` against detected GPU VRAM before launch.
- Current implementation supports periodic real eval cadence control.
- Dual-cap stop policy and mandatory final 1200-image full eval are approved target behavior pending full implementation lock.

## Script Groups

- Config checks: `scripts/config/`
- Evaluation: `scripts/eval/`
- Quickcheck pipeline: `scripts/quickcheck/`
- Reporting: `scripts/reporting/`
- Server checks/bootstrap: `scripts/server/`
- Training wrappers: `scripts/training/`

## Slurm

- Submit wrapper: `slurm/submit.sh`
- Templates: `slurm/templates/*.sbatch`

## Runtime Outputs

- Canonical runtime root: `runs/current/`
- Checkpoints: `runs/current/checkpoints/`
- Slurm logs: `runs/current/logs/`
- Evaluation reports: `runs/current/reports/`
- Quickcheck outputs: `runs/current/quickcheck/`
- W&B local metadata: `runs/current/wandb/`
- Stage runtime configs: `runs/current/generated/`
- Stage manifests: `runs/current/reports/manifests/`
- Optional gate-fixture evidence (runtime-generated) uses existing `runs/current/reports/real_tuple_eval_epoch_*.json` paths.

## Layout Status

- `project_root/` is removed from the active repository layout.
- Curated shareable artifacts remain under `artifacts/` (for example `artifacts/insights/`).
