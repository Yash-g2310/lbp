# Scope and Current State

Last reviewed: 2026-04-07
Scope: repository `lbp/lbp` only.

## What Is Implemented

1. Modular docs structure added under `docs/guides`, `docs/reference`, and `docs/generated`.
2. Data truth reporting script exists and is integrated in documentation.
3. Data reconciliation logic has been corrected for split counting and alias handling.
4. Runtime output paths are standardized under `runs/current` across configs, scripts, and Slurm.
5. Legacy `project_root/` has been removed from active repository layout.
6. Stage-policy workflows are active via `cli.py stage-a` and `cli.py stage-b`, including gate and manifest wiring.
7. Phase 9A ablation scaffolding is implemented (module + runtime payload + experiment presets).
8. Server profiles now explicitly pin `architecture.backbone_fallback_approved: false` for no-silent-fallback visibility.
9. Server preflight now enforces VRAM-aware hardware policy via `hardware.min_vram_gb` and fails early on profile mismatch.
10. Fixture handling is implemented through existing quickcheck pipeline switches (no new fixture files in repo).

## Current Focus Areas

- Keep docs and context modular for future chat handoff.
- Keep generated runtime artifacts out of tracked curated paths.
- Keep actionable technical context in small topic files under `context/`.
- Preserve clean boundary: runtime (`runs/`) vs curated artifacts (`artifacts/insights/`).
- Keep dry-run prerequisites explicit (checkpoint availability for Stage A skip-train and Stage B gate evidence reports).
- Keep strict fail-hard semantics for stage gates while making failure messages actionable.

## High-Signal Facts

- Local and server configs intentionally differ in real-eval split policy.
- Real tuple evaluation and synthetic depth validation consume different fields and datasets.
- Active runtime writes should target `runs/current` only.
- Host-path permissions can block quickcheck depending on selected profile (`/home/...` or `/mnt/home2/...`).
- Stage A dry-run with `--skip-train` resolves checkpoints from configured paths first and then quickcheck checkpoint fixture fallback.
- Stage B dry-run enforces promotion evidence (`real_tuple_eval_epoch_*.json` or `final_report.json`) before orchestration proceeds.
- Existing quickcheck script can emit stage-gate fixture evidence via `EMIT_STAGE_GATE_FIXTURE=1`.

## Use This File When

- A new chat needs a one-minute orientation.
- You want to know where authoritative project context now lives.
