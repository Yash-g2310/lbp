# Stage B Server Script Protocol

Last reviewed: 2026-04-07
Scope: Full-data scripted training/evaluation after Stage A pass.

## Goal

Execute full-data run with reproducibility, strict preflight, and stable reporting.

## Implemented Workflow Hook (2026-04-07)

1. Stage B policy workflow is now available via CLI:
- `python cli.py stage-b --config configs/server/default.yaml --periodic-eval-every 10`
2. Slurm train template now dispatches through Stage B workflow command.
3. Stage-policy runtime config generation and run-manifest emission are now part of `scripts/training/train_eval.py`.

## Entry Preconditions

Stage B is allowed only if Stage A gate is passed:

1. No runtime/schema/index errors in Stage A.
2. All losses finite and gradients healthy in Stage A.
3. Non-zero and improving local tuple metrics.

## Full-Data Contract

1. Synthetic nominal definitions:
- Train: 14800
- Validation: 500

2. Real nominal definitions:
- Test: 1200
- Validation: 300

3. Important:
- Nominal definitions are fixed dataset facts.
- Server materialization must be verified independently from local cache state.

## Hardware-Aware Server Guidance (48GB Class, Range-Based)

1. Batch size: choose a throughput-oriented range based on memory headroom.
2. Gradient accumulation: keep minimal when memory allows; increase only if needed for stability.
3. Precision mode: use stable mixed precision policy validated in Stage A.
4. Evaluation cadence should balance signal and cost; locked default is every 10 epochs.
5. Track runtime memory counters so server settings remain reproducible.

## Mandatory Server Preflight

1. Data presence checks:
- Required splits resolvable.
- Required schema keys present.

2. Precomputed feature checks (if enabled):
- Index path exists.
- Split coverage complete for referenced splits.
- Sample ID lookup integrity verified on spot-check.

3. Backbone checks:
- Primary backbone weight access/load test passes.
- If failed, apply interactive fallback protocol from Phase 9 docs.

4. Environment checks:
- Torch/CUDA import and device visibility.
- Critical package versions logged.
5. Hardware policy checks:
- Detected GPU VRAM must satisfy profile minimum via `hardware.min_vram_gb`.
- Profile mismatch is a hard failure before launch.

## Runtime Protocol

1. Training run is script-driven (no notebook dependency).
2. Validation cadence is locked before launch:
- loss components logged each epoch.
- validation logged every 10 epochs and at end-of-run.
3. Logs must include:
- per-loss components
- gradient/optimizer health
- memory/runtime counters
- preflight hardware diagnostics (requested profile class, detected VRAM, pass/fail status)

## Abort and Recovery Rules

Abort conditions:

1. Repeated non-finite loss/gradient events beyond configured patience.
2. Unrecoverable index/key mismatches.
3. Backbone access failure without approved fallback.
4. Hardware profile VRAM check failure.

Recovery requirements:

1. Persist failure snapshot (config + logs + checkpoint metadata).
2. Classify failure category.
3. Record next action and owner in decision log.

## Required Stage B Outputs

1. Config and environment manifest.
2. Training report with staged-loss behavior.
3. Synthetic evaluation report.
4. Real tuple evaluation report.
5. Promotion recommendation for long-run or ablation branch.

## Stage B Completion Gate

1. Reports complete and consistent with eval contract.
2. No unresolved critical runtime contract violations.
3. Decision note recorded for next branch (ablation or optimization).
