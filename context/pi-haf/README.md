# PI-HAF Context Pack

Last reviewed: 2026-04-07
Scope: PI-HAF layered-depth experiment program in `lbp/lbp`.

## Start Order For New Chats

1. `01_program_state_and_experiment_protocol.md`
2. `02_experiment_master_plan.md`
3. `03_stageA_local_notebook_protocol.md`
4. `04_stageB_server_script_protocol.md`
5. `05_architecture_spec_sfwin_rhag_backbone.md`
6. `06_loss_stack_math_and_schedule.md`
7. `07_data_shards_and_split_contract.md`
8. `08_eval_contract_reporting.md`
9. `09_risks_questions_decisions.md`
10. `10_env_cleanup_runbook.md`
11. `11_ablation_matrix_wavelet_freq_hybrid.md`
12. `GSoC2026_Proposal_YashGupta_DeepLense_final.tex`

## File Index

- `01_program_state_and_experiment_protocol.md`: canonical state, execution gates, and program-level protocol.
- `02_experiment_master_plan.md`: phase-wise master plan, locked decisions, and expected final change set.
- `03_stageA_local_notebook_protocol.md`: local 4-5 epoch bring-up protocol and strict Stage A acceptance gates.
- `04_stageB_server_script_protocol.md`: server full-data scripted execution protocol and preflight/abort rules.
- `05_architecture_spec_sfwin_rhag_backbone.md`: architecture contract for backbone, SFWIN-first path, and RHAG compatibility.
- `06_loss_stack_math_and_schedule.md`: current vs target loss inventory, equations, and staged schedule.
- `07_data_shards_and_split_contract.md`: canonical split definitions vs local materialization rules and index checks.
- `08_eval_contract_reporting.md`: synthetic/real eval contract including tuple-layer coverage policy and reporting schema.
- `09_risks_questions_decisions.md`: locked decisions, active risks, and unresolved questions.
- `10_env_cleanup_runbook.md`: environment footprint cleanup and verification runbook.
- `11_ablation_matrix_wavelet_freq_hybrid.md`: post-baseline ablation roadmap.
- `GSoC2026_Proposal_YashGupta_DeepLense_final.tex`: submitted GSoC proposal with detailed technical and mathematical formulation.

## Maintenance Rule

- Use `02` to `11` as the working modular implementation pack.
- Keep `01_program_state_and_experiment_protocol.md` as high-level operational summary.
- Keep proposal `.tex` as immutable submitted reference; add notes in markdown files instead of rewriting proposal text.
