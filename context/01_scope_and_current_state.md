# Scope and Current State

Last reviewed: 2026-04-05
Scope: repository `lbp/lbp` only.

## What Is Implemented

1. Modular docs structure added under `docs/guides`, `docs/reference`, and `docs/generated`.
2. Data truth reporting script exists and is integrated in documentation.
3. Data reconciliation logic has been corrected for split counting and alias handling.

## Current Focus Areas

- Keep docs and context modular for future chat handoff.
- Keep generated artifacts out of versioned docs.
- Keep actionable technical context in small topic files under `context/`.

## High-Signal Facts

- Local and server configs intentionally differ in real-eval split policy.
- Real tuple evaluation and synthetic depth validation consume different fields and datasets.
- Local cache state is currently partial for three of the four key segments.

## Use This File When

- A new chat needs a one-minute orientation.
- You want to know where authoritative project context now lives.
