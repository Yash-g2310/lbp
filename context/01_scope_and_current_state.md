# Scope and Current State

Last reviewed: 2026-04-06
Scope: repository `lbp/lbp` only.

## What Is Implemented

1. Modular docs structure added under `docs/guides`, `docs/reference`, and `docs/generated`.
2. Data truth reporting script exists and is integrated in documentation.
3. Data reconciliation logic has been corrected for split counting and alias handling.
4. Runtime output paths are standardized under `runs/current` across configs, scripts, and Slurm.
5. Legacy `project_root/` has been removed from active repository layout.

## Current Focus Areas

- Keep docs and context modular for future chat handoff.
- Keep generated runtime artifacts out of tracked curated paths.
- Keep actionable technical context in small topic files under `context/`.
- Preserve clean boundary: runtime (`runs/`) vs curated artifacts (`artifacts/insights/`).

## High-Signal Facts

- Local and server configs intentionally differ in real-eval split policy.
- Real tuple evaluation and synthetic depth validation consume different fields and datasets.
- Active runtime writes should target `runs/current` only.
- Local quickcheck config may fail on this server due `/home/yash_gupta/...` cache path permissions.

## Use This File When

- A new chat needs a one-minute orientation.
- You want to know where authoritative project context now lives.
