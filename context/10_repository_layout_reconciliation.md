# Repository Layout Reconciliation

Last reviewed: 2026-04-06

## Why This File Exists

This file captures the completed reconciliation from the old tracked `project_root/` layout to the repository-root modular layout.

This file captures the intended migration mapping so subsequent chats and commits stay consistent.

## Canonical Layout Decision

Canonical project root is the repository root:

- `src/`
- `scripts/`
- `configs/`
- `slurm/`
- `docs/`
- `context/`
- `train.py`
- `cli.py`

The legacy `project_root/` tracked structure is treated as deprecated history.

Status update: `project_root/` has been fully removed from the active runtime structure.

## High-Level Migration Mapping

| Legacy tracked path | Canonical path |
|---|---|
| `project_root/train.py` | `train.py` |
| `project_root/.gitignore` | `.gitignore` |
| `project_root/configs/*.yaml` | `configs/local/*.yaml`, `configs/server/*.yaml` |
| `project_root/data/*.py` | `src/lbp_project/data/*.py` |
| `project_root/models/*.py` | `src/lbp_project/models/*.py` |
| `project_root/utils/*.py` | `src/lbp_project/utils/*.py` |
| `project_root/scripts/*` | `scripts/*` (domain-grouped folders) |
| `project_root/slurm/*.sbatch` | `slurm/templates/*.sbatch` |
| `project_root/slurm/submit_server.sh` | `slurm/submit.sh` |
| `project_root/docs/*.md` | `docs/guides/*.md`, `docs/reference/*.md`, compatibility pages in `docs/*.md` |

## Runtime Layout (Current)

- Active runtime outputs live under `runs/current/`.
- Optional historical runtime outputs live under `runs/archive/`.
- Curated outputs intended for sharing remain under `artifacts/`.
- `src/lbp_project/` is the active implementation/import root for current execution phases.
- `src/pi-haf/` is reference-only in the current phase; new untracked files under this tree are ignored by `.gitignore` policy.

## Notes About Notebooks and Reports

- Root notebooks that were historically tracked are now intentionally ignored by policy (`*.ipynb`).
- Generated reports are now written under `docs/generated/` and ignored by policy.

## Commit-Time Checklist

1. Stage migration changes intentionally (root canonical, legacy removed).
2. Verify no active scripts still reference `project_root/` runtime paths.
3. Ensure docs and context links resolve.
4. Keep generated files and notebooks untracked as per policy.

## Verification Status

- Markdown links across `docs/` and `context/`: passing.
- Reporting scripts write to `docs/generated/`: passing.
- `.gitignore` policy includes notebooks and generated artifacts: applied.
- Runtime and Slurm paths are standardized on `runs/current/`: applied.
- Legacy `project_root/` directory removal: applied.
- Post-removal config/preflight and Slurm dry-run checks: passing.
