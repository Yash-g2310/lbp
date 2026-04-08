# Environment Footprint Cleanup Runbook

Last reviewed: 2026-04-07
Scope: Safe cleanup and verification for conda/pip GPU training environment.

## Goal

Reduce environment footprint without breaking torch/cuda runtime required for Stage A and Stage B.

## Safety Principles

1. Always snapshot environment before cleanup.
2. Use package-manager cleanup commands, not manual deletion of internal env files.
3. Verify torch/cuda imports immediately after cleanup.
4. Block implementation runs until verification passes.

## Pre-Cleanup Snapshot

Recommended:

1. Export package snapshot.
2. Record torch/torchvision/cuda versions.
3. Record `nvidia-smi` output.

## Safe Cleanup Sequence

1. Conda dry run:
- `conda clean --all --dry-run`

2. Conda cleanup:
- `conda clean --all --yes`

3. Pip cache cleanup:
- `pip cache purge`

4. Duplicate CUDA wheel check (safe audit):
- `pip list | grep -E "^(torch|torchvision|torchaudio|nvidia-.*)"`
- `conda list | grep -E "^(pytorch|torchvision|pytorch-cuda|cuda|cudnn)"`
- If duplicate package versions are detected, remove only outdated duplicates via package manager commands.

5. Optional additional cleanup only if needed:
- `conda clean --packages --yes`
- `conda clean --tarballs --yes`

## Forbidden / High-Risk Actions

1. Manual deletion of conda package internals.
2. Untracked removal of CUDA runtime libs.
3. Running training without post-clean verification.

## Post-Cleanup Verification Checklist

1. `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
2. `python -c "import torchvision; print(torchvision.__version__)"`
3. `python -c "import torch; print(torch.cuda.is_available())"`
4. If GPU expected: verify visible device count and name.
5. `pip check` to confirm dependency consistency after cleanup.
6. Run one minimal model import/forward smoke check.

## Current Session Note

- Pip cache purge has been executed in this session.
- Continue with full verification before training runs.
- Strict dependency pin normalization can still produce CUDA ABI/runtime regressions even when metadata appears cleaner; treat import/forward smoke checks as final authority.

## Recovery Protocol

If verification fails:

1. Stop all planned runs.
2. Restore known-good package set from snapshot.
3. Re-run post-clean verification.
4. Record incident in risks/decisions log.
