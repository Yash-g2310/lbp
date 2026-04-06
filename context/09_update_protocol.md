# Context Update Protocol

Last reviewed: 2026-04-05

## Goal

Keep context modular, durable, and useful for subsequent chats.

## Rules

1. Update only the module affected by the change.
2. Add date stamps to touched files.
3. Keep entries factual and tied to repo paths.
4. Do not copy raw chat transcript blocks.
5. Do not store machine-specific secrets or tokens.

## Update Triggers

- Config key change: update `03_data_segments_and_usage.md` and `05_entrypoints_and_execution_modes.md`.
- Precompute/index change: update `04_indexing_precompute.md`.
- New failure mode: update `07_known_risks_and_open_items.md`.
- New validation command: update `08_verification_playbook.md`.

## Recommended Cadence

- After major refactor: review full `context/` set.
- After bugfix or config change: review only impacted modules.
