# Verification Playbook

Last reviewed: 2026-04-05

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

## Local Train+Eval Smoke

```bash
cd /home/yash_gupta/lbp/lbp
python cli.py train-eval --config configs/local/dev.yaml --skip-train
```

## Server Preflight

```bash
cd /home/yash_gupta/lbp/lbp
python scripts/server/preflight.py --config configs/server/default.yaml --sbatch slurm/templates/train.sbatch
python scripts/server/check_env.py --config configs/server/default.yaml
```
