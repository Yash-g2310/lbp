# Verification Playbook

Last reviewed: 2026-04-07

Use these commands after major config or data path changes.

## Docs and Reporting

```bash
cd /home/yash_gupta/lbp/lbp
python scripts/reporting/data_inventory.py --output docs/generated/data_inventory_report.json
python scripts/reporting/data_truth_report.py --output-json docs/generated/data_truth_report.json --output-md docs/generated/data_truth_report.md
```

## Quickcheck

```bash
cd /home/yash_gupta/lbp/lbp
python cli.py quickcheck --config configs/local/quickcheck.yaml
```

If local cache paths are not writable on the current host, run server quickcheck config with explicit env python:

```bash
cd /home/yash_gupta/lbp/lbp
PYTHON_BIN=/mnt/home2/home/yash_g/.conda/envs/monodepth/bin/python \
bash scripts/quickcheck/run.sh configs/server/quickcheck.yaml
```

Known host caveat:

- On non-server hosts, `configs/server/quickcheck.yaml` may fail with permissions on `/mnt/home2/...`.
- Use fallback cache override when needed:

```bash
cd /home/yash_gupta/lbp/lbp
FALLBACK_CACHE_DIR=./runs/current/quickcheck/hf_cache \
python cli.py quickcheck --config configs/server/quickcheck.yaml
```

- To emit strict Stage-B gate fixture evidence from existing quickcheck outputs (no new fixture files):

```bash
cd /home/yash_gupta/lbp/lbp
EMIT_STAGE_GATE_FIXTURE=1 python cli.py quickcheck --config configs/server/quickcheck.yaml
```

## Local Train+Eval Smoke

```bash
cd /home/yash_gupta/lbp/lbp
python cli.py train-eval --config configs/local/dev.yaml --skip-train
```

## Stage A Policy Run

```bash
cd /home/yash_gupta/lbp/lbp
python cli.py stage-a --config configs/local/dev.yaml --epochs 5 --skip-train
```

Precondition for `--skip-train`:

- `runs/current/checkpoints/best_checkpoint.pth` or `runs/current/checkpoints/latest_checkpoint.pth` should exist.
- If missing, orchestration falls back to quickcheck checkpoint fixture at `runs/current/quickcheck/checkpoints/quickcheck/sanity_roundtrip.pth` when available.

## Stage B Policy Dry-Orchestration

```bash
cd /home/yash_gupta/lbp/lbp
python cli.py stage-b --config configs/server/default.yaml --skip-train
```

Preconditions for Stage B dry-run:

- Promotion evidence must exist under `runs/current/reports/` as periodic files (`real_tuple_eval_epoch_*.json`) or a valid `final_report.json`.
- Without evidence files, the Stage B gate is expected to fail before training/eval orchestration.
- Hardware policy must pass: detected GPU VRAM must meet `hardware.min_vram_gb` for the selected server profile.

## Server Preflight

```bash
cd /home/yash_gupta/lbp/lbp
python scripts/server/preflight.py --config configs/server/default.yaml --sbatch slurm/templates/train.sbatch
python scripts/server/check_env.py --config configs/server/default.yaml
```

## Slurm Dry-Run Path Check

```bash
cd /home/yash_gupta/lbp/lbp
sbatch --test-only --chdir="$PWD" --output="$PWD/runs/current/logs/test-%j.out" --error="$PWD/runs/current/logs/test-%j.err" slurm/templates/train.sbatch
```
