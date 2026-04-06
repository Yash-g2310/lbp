# Repository Layout Reconciliation

Last reviewed: 2026-04-05

## Why This File Exists

The repository index currently reflects an older tracked layout rooted at `project_root/`, while active development is now happening at repository root (`src/`, `scripts/`, `configs/`, `docs/`, `context/`).

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
