# Context Hub

This folder holds curated, modular context for future chats.

## Start Order For New Chats

1. `01_scope_and_current_state.md`
2. `10_repository_layout_reconciliation.md`
3. `12_runtime_layout_and_cleanup_status.md`
4. `03_data_segments_and_usage.md`
5. `04_indexing_precompute.md`
6. `05_entrypoints_and_execution_modes.md`
7. `07_known_risks_and_open_items.md`

For LayeredDepth experiment planning and paper alignment:

8. `13_layereddepth_paper_baseline.md`
9. `14_paper_vs_current_delta.md`

## File Index

- `01_scope_and_current_state.md`: current status and what changed recently.
- `02_architecture_and_pipeline.md`: model and training/eval data flow.
- `03_data_segments_and_usage.md`: four-segment usage matrix and counts.
- `04_indexing_precompute.md`: DINO precompute/index behavior and paths.
- `05_entrypoints_and_execution_modes.md`: CLI, scripts, and local/server execution paths.
- `06_notebook_state.md`: notebook reliability and limitations.
- `07_known_risks_and_open_items.md`: unresolved risks and next fixes.
- `08_verification_playbook.md`: reproducible verification commands.
- `09_update_protocol.md`: update rules so context stays modular.
- `10_repository_layout_reconciliation.md`: old tracked layout to canonical root mapping.
- `11_transparent_surface_cues_lab.md`: experiment notebook scope, outputs, and first-run insights.
- `12_runtime_layout_and_cleanup_status.md`: final runtime-path migration status and deletion summary.
- `13_layereddepth_paper_baseline.md`: concise LayeredDepth paper baseline summary for reproducible references.
- `14_paper_vs_current_delta.md`: explicit paper-vs-current implementation mapping and implications.

## Subfolders

- `pi-haf/`: PI-HAF layered-depth modular context pack.
	- Core order now starts from `pi-haf/01_program_state_and_experiment_protocol.md` and `pi-haf/02_experiment_master_plan.md`.
	- Includes Stage A local protocol, Stage B server protocol, architecture/loss specs, shard/eval contracts, risks log, cleanup runbook, and ablation matrix.
	- Active PI-HAF contract is supervised-first with tuple P/T/Q as primary real benchmark gates, interactive backbone fallback, and stage-specific logging cadence.

## Guardrails

- Do not copy raw chat transcripts into these files.
- Keep content task-oriented and repo-relative.
- Update only impacted modules to avoid monolithic growth.
