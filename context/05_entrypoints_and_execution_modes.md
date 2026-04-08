# Entrypoints and Execution Modes

Last reviewed: 2026-04-07

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

## Local Mode

Typical configs:

- `configs/local/dev.yaml`
- `configs/local/quickcheck.yaml`

Behavior:

- Real eval is validation-only by default.
- Precomputed DINO is disabled by default.

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
