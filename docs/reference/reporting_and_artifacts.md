# Reporting and Artifacts Reference

This file defines where generated reports go and what should be versioned.

## Runtime Output Locations

Runtime outputs from training/evaluation pipelines are stored under `runs/current/`:

- Checkpoints: `runs/current/checkpoints/`
- Slurm logs: `runs/current/logs/`
- Evaluation reports: `runs/current/reports/`
- Quickcheck outputs: `runs/current/quickcheck/`
- W&B run metadata: `runs/current/wandb/`

`project_root/` is not part of the active runtime layout.

## Reporting Scripts

- Data inventory: `scripts/reporting/data_inventory.py`
- Data truth reconciliation: `scripts/reporting/data_truth_report.py`

## Default Output Locations

Generated outputs are written to `docs/generated/`:

- `docs/generated/data_inventory_report.json`
- `docs/generated/data_truth_report.json`
- `docs/generated/data_truth_report.md`

## Regenerate Commands

```bash
cd /home/yash_gupta/lbp/lbp
python scripts/reporting/data_inventory.py --output docs/generated/data_inventory_report.json
python scripts/reporting/data_truth_report.py --output-json docs/generated/data_truth_report.json --output-md docs/generated/data_truth_report.md
```

## Versioning Policy

- Curated docs and references are tracked in git.
- Machine-generated report artifacts are ignored by `.gitignore`.
- Raw chat transcript dumps are not copied into project docs.
- Curated artifacts under `artifacts/insights/` remain trackable by policy.
- Generated artifact folders such as `artifacts/reports/`, `artifacts/archive/`, and `artifacts/tmp/` are ignored.

## Why This Split

1. Keep documentation review noise low.
2. Keep context portable across machines.
3. Prevent large generated artifacts from polluting git history.
