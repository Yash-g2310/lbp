# Entrypoints and Execution Modes

Last reviewed: 2026-04-05

## Unified CLI

File: `src/lbp_project/cli.py`

Commands:

- `train`
- `train-eval`
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
