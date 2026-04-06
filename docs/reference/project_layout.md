# Project Layout Reference

This file describes structure and ownership boundaries, not runbook steps.

## Canonical Root

The canonical project layout is repository-root based.

- Active modules live at root (`src/`, `scripts/`, `configs/`, `slurm/`, `docs/`, `context/`).
- Legacy tracked paths under `project_root/` are considered deprecated migration history.
- Reconciliation mapping is documented in `../../context/10_repository_layout_reconciliation.md`.

## Top-Level Structure

```text
lbp/
├── cli.py
├── pyproject.toml
├── train.py
├── configs/
│   ├── base/
│   ├── experiments/
│   ├── local/
│   └── server/
├── scripts/
│   ├── config/
│   ├── eval/
│   ├── quickcheck/
│   ├── reporting/
│   ├── server/
│   └── training/
├── src/lbp_project/
│   ├── cli.py
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── slurm/
│   ├── submit.sh
│   └── templates/
├── docs/
└── context/
```

## Canonical Command Surface

- Local training: `python cli.py train --config configs/local/dev.yaml`
- Local train+eval: `python cli.py train-eval --config configs/local/dev.yaml`
- Local quickcheck: `python cli.py quickcheck --config configs/local/quickcheck.yaml`
- Real tuple eval: `python cli.py eval-real --config configs/server/default.yaml --checkpoint checkpoints/best_checkpoint.pth --output artifacts/reports/real_tuple_eval.json`
- Batch checkpoint scan: `python cli.py eval-checkpoints --config configs/server/default.yaml`

## Related Docs

- Workflow guides: `../guides/quickstart.md`, `../guides/quickcheck_guide.md`, `../guides/slurm_operations.md`
- Data/index reference: `data_configuration.md`
