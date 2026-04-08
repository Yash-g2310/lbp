# Experiment Master Plan: PI-HAF LayeredDepth Program

Last reviewed: 2026-04-07
Scope: Planning source-of-truth for the updated PI-HAF program with DINO backbone and wavelet-first interaction.
Status: Approved for implementation start.

## Scope Lock Update (2026-04-07)

1. Active implementation in this phase is restricted to `src/lbp_project`.
2. `src/pi-haf` remains reference-only and is excluded from new active-workflow artifacts via `.gitignore` policy.
3. Phase 2 and Phase 3A startup wiring has been implemented in `lbp_project` with modular modules for:
- config validation,
- backbone policy and loading,
- data/precomputed compatibility preflight,
- stage boundary scheduling,
- tuple required-layer expansion helpers.
4. Current implementation cutoff has been extended through Phase 7 workflow wiring in `lbp_project`:
- Stage A and Stage B stage-policy orchestration,
- stage-aware CLI entrypoints,
- manifest-backed reproducibility metadata in train/eval reports,
- Stage B periodic real-eval cadence policy defaults (10 epochs) in server profiles.

## Program Objective

Build a theoretically sound, implementation-safe layered depth pipeline using an updated PI-HAF architecture, with:

- Frozen DINO backbone family.
- Wavelet-first interaction blocks in the updated PI-HAF path.
- Two-stage execution strategy:
  - Stage A: local 4-5 epoch notebook bring-up with exhaustive logging and hard correctness gates.
  - Stage B: server full-data script execution with reproducibility and promotion gates.

## Locked Decisions

1. Stage A local run length: 4-5 epochs.
2. Real data handling:
- Nominal dataset definition remains 1200 test + 300 validation.
- Local shard materialization is reindexed and must be treated as local runtime state, not dataset redefinition.
- Retained real test shards are used for local evaluation checks.
3. Layer coverage in local tuple evaluation:
- Auto-expand predictions to all layer IDs referenced by tuples.
4. Backbone choice:
- Primary baseline: DINOv3 ConvNeXt-Small distilled.
- First fallback: DINOv3 ViT-S16+ distilled.
- If primary/fallback are unavailable: stop and ask user interactively for next action.
- DINOv2 temporary plumbing fallback is disabled by default and requires explicit one-off approval.
5. Wavelet defaults:
- Default family: `sym4`.
- Fallback family: `bior3.5`.
- Default decomposition level: 2.

## Phase Map

1. Phase 0: Context Modularization
- Build modular docs under `context/pi-haf/` and index them in `context/pi-haf/README.md`.

2. Phase 1: Baseline Capture
- Freeze current losses, architecture contracts, and evaluation assumptions before changes.

3. Phase 2: Data and Shard Contract
- Separate nominal split definitions from local cache/shard materialization.
- Define Stage A and Stage B data preflight checks.

4. Phase 3: Architecture Contract
- Specify updated PI-HAF + SFWIN + RHAG path and backbone integration constraints.

5. Phase 4: Loss Math and Schedule
- Capture current losses and target staged loss schedule.
- Define monitoring and stability checks for every loss component.
- Lock stage boundaries and weights as config-driven guardrails.

6. Phase 5: Stage A Local Notebook Protocol
- Define runbook, logging requirements, and acceptance gates for 4-5 epoch local run.
- Lock local validation reporting cadence to end-of-epoch.

7. Phase 6: Stage B Server Script Protocol
- Define full-data prerequisites, launch preflight, reporting and abort criteria.
- Lock server validation reporting cadence to every 10 epochs.

8. Phase 7: Evaluation and Reporting Contract
- Lock synthetic and real tuple metrics schema and interpretation rules.

9. Phase 8: Environment Footprint and Recovery
- Define safe cleanup and post-clean verification for conda/pip GPU stack.

10. Phase 9: Ablation Roadmap
- Define post-baseline ablations: frequency-only, wavelet-only, hybrid wavelet+frequency.

## Final Expected Change Set (Implementation Target)

1. Context and planning artifacts:
- New modular docs 02-11 under `context/pi-haf/`.
- Updated `context/pi-haf/README.md` index and read order.

2. Architecture and config changes (implementation stage):
- Updated PI-HAF interaction path from frequency-first to wavelet-first.
- ConvNeXt-small distilled backbone as baseline config.
- Explicit compatibility checks for SFIN/SWIN/SFWIN + RHAG combinations.

3. Training and validation changes (implementation stage):
- Staged Flow+SSI then Flow+SSI+Wavelet+Ordinal schedule.
- Stage schedule/weights loaded from config only.
- Full component-level logging and gradient health checks in Stage A.
- Local validation logging end-of-epoch and server validation logging every 10 epochs.

4. Evaluation contract updates (implementation stage):
- Auto-expanded layer coverage for tuple evaluation.
- Non-zero and improving tuple metric requirement in Stage A gate.

5. Ops and reproducibility changes:
- Environment cleanup runbook integrated with validation commands.
- Manifest requirements for local and server runs.

## Promotion Gates

1. G1 Planning Gate:
- All modular docs approved; no unresolved critical ambiguity.

2. G2 Stage A Gate:
- No runtime/schema/index errors.
- All monitored losses finite.
- Gradients finite and non-zero.
- Local tuple metrics non-zero and improving across Stage A window.

3. G3 Stage B Gate:
- Full-data preflight passes.
- Backbone access/runtime checks pass.
- Reproducibility artifacts configured.

4. G4 Baseline Completion Gate:
- Server run report complete with synthetic and real tuple sections.
- Risk notes and next-step ablation decision recorded.
