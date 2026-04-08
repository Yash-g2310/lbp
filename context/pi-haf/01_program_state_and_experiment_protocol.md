# PI-HAF Program State And Experiment Protocol

Last reviewed: 2026-04-08
Scope: repository `lbp/lbp` only.

## Current Program Status (2026-04-07)

1. Planning-to-implementation transition has started through a modular context pack under `context/pi-haf/`.
2. Active implementation source-of-truth is now split across `02` to `11` modules (master plan, stage protocols, architecture/loss specs, data/eval contracts, risks, env runbook, ablations).
3. This file remains the high-level operational anchor; detailed execution contracts live in the modular docs.
4. Runtime implementation for this wave is locked to `src/lbp_project`; `src/pi-haf` is reference-only.
5. Phase 3A startup modules are active in `src/lbp_project` for config validation, backbone policy/loader, preflight compatibility checks, stage scheduling, and tuple-layer expansion helpers.
6. Phase 5-7 workflow wiring is now active in `src/lbp_project` and scripts via stage-aware commands, policy runtime configs, and run manifests.
7. Phase 9A scaffold wiring is active in orchestration: ablation payload logging, Stage-A post-eval gate checks, and Stage-B pre-gate checks.
8. Server profiles now explicitly pin `architecture.backbone_fallback_approved: false` (`default`, `balanced`, `stable`) to make no-silent-fallback policy visible in config.
9. Dated implementation note (2026-04-07): server preflight now enforces VRAM-aware profile checks (`hardware.min_vram_gb`) and quickcheck supports runtime fixture evidence emission through existing scripts.

## Implemented vs Target Snapshot (2026-04-08)

1. Implemented now:
- Stage command orchestration, gate wiring, manifests, and tuple P/T/Q reporting contracts.
- ConvNeXt-S distilled backbone policy as active baseline.

2. Approved target (pending full implementation lock):
- Dataloader-level empty-space rule (`L1 = L2` for opaque/no-front-layer cases).
- Inverse-depth normalized training space bounded to `[-1,1]` for flow trajectory training.
- AdaLN-Zero full-scope conditioning with `(layer_id, timestep)` through decoder residual blocks.
- Stage B dual-cap runtime policy: 25 preferred, up to 30 if budget remains, hard stop at `min(24h, 30 epochs)` with mandatory final full real evaluation.

3. Migration boundary decision:
- Depth-space migration is a retraining boundary; legacy positive-depth checkpoints are not promoted across this boundary.

## Confirmed Current State

1. Local and server branches are synchronized after push/pull in both directions.
2. Proposal reference lives at `context/pi-haf/GSoC2026_Proposal_YashGupta_DeepLense_final.tex`.
3. PI-HAF source code is present at `src/pi-haf/`.
4. `src/pi-haf/` is not used as an active import/runtime target in this phase.
5. Notebook history for PI-HAF exists outside this source package; source tree currently contains the runnable modules and training/eval code.
6. Local debug data has been reduced to one shard each for train and validation.
7. The prior failed attempt was a separate legacy DINOv2 + SFIN + RHAG U-Net run, not the PI-HAF codepath.
8. Dry-run verification confirms stage runtime config generation and gate enforcement behavior, with expected failures when checkpoints or gate evidence files are absent.
9. Stage A skip-train orchestration can use quickcheck checkpoint fallback when configured checkpoint paths are missing.

## Program Goal

Run multiple controlled supervised experiments across models and pipelines, then pick the strongest candidate under the current server budget contract (Stage B dual-cap runtime policy).

## Mandatory Workflow For Each Experiment

1. Planning and verification gate:
- Define hypothesis and expected failure modes.
- Verify math, objective terms, assumptions, and scale/units.
- Verify full pipeline contract (data -> model -> loss -> metrics -> artifacts).
- Add optimization and stability plan (memory, AMP dtype, gradient clip, LR schedule).
- Add explicit error checks and acceptance criteria.
2. Local notebook gate:
- Run the experiment notebook on local reduced data (1 shard train + 1 shard val).
- Fix runtime and numerical issues until stable.
- Record timing, memory, and metric sanity trends.
3. Pre-scale hardening gate:
- Apply recommendations from local run.
- Re-run smoke + short training to confirm no regressions.
4. Full-data medium run gate:
- Train on complete data for 15-20+ epochs (about 1 day budget).
- Validate and test.
5. Selection gate:
- Compare against prior experiments on common metrics and physics checks.
- Decide whether experiment is promoted to the final 100-epoch candidate pool.

## Recommended Experiment Directory Convention

Use one folder per experiment with immutable IDs.

`artifacts/experiments/pi-haf/exp_YYYYMMDD_XX_<slug>/`

Required subfolders:

- `configs/`: frozen config snapshot used for the run.
- `notebooks/`: experiment notebook used for local runs.
- `logs/`: stdout logs and training traces.
- `metrics/`: csv/json metrics per epoch and final summary.
- `plots/`: learning curves and qualitative panels.
- `reports/`: short markdown report with conclusion.
- `manifests/`: run metadata (git commit, seed, host, device, data shard info).

Heavy outputs (recommended to keep out of tracked artifacts unless explicitly required):

- `checkpoints/experiments/pi-haf/exp_YYYYMMDD_XX_<slug>/`

## Naming Convention

Experiment ID:

- `exp_YYYYMMDD_XX_<model>_<objective>_<domain>`

Examples:

- `exp_20260406_01_pihaf_flow_sim`
- `exp_20260406_02_pihaf_wavelet_tuple`

Run tags inside each experiment:

- `local_smoke`
- `local_short`
- `server_mid_20ep`
- `server_full_100ep`

## PI-HAF Technical Anchors To Keep Consistent

1. Rectified-flow training target with Euler ODE sampling (`src/pi-haf/engine/train_sr.py`).
2. Active supervised path for this cycle: staged Flow/SSI/Wavelet/Ordinal objective (`context/pi-haf/06_loss_stack_math_and_schedule.md`).
3. PI-HAF config constraints (including expected LR shape logic) (`src/pi-haf/models/pi_haf/config.py`).
4. Real tuple evaluation contract with per-image required-layer expansion (`scripts/eval/eval_real_tuples.py` and `src/lbp_project/utils/metrics.py`).
5. Task6B/LoRA track is out of scope for the current supervised baseline and remains inactive.

## Minimal Verification Checklist Per New Experiment

1. Data contract checks:
- LR/HR shapes satisfy exact 2x relation where required.
- Split sizes and shard usage are logged.
- No NaN/Inf in loaded tensors.
2. Objective checks:
- Every loss term finite and numerically scaled.
- Loss stage boundaries, components, and weights are config-driven (no runtime mutation).
- Loss components logged each epoch.
3. Optimization checks:
- Learning rate and stage transitions are logged.
- AMP mode and dtype are recorded.
- Gradient norms monitored; clip threshold justified.
- Hardware-aware settings stay within documented local/server ranges.
4. Metrics checks:
- Real benchmark gates use tuple metrics P/T/Q (`pairs`, `trips`, `quads`) as primary.
- Local validation is logged at end-of-epoch; server validation is logged every 10 epochs.
 - Final-stop full real evaluation is mandatory under Stage B closeout policy.
5. Reproducibility checks:
- Git commit hash, seed, and config fingerprint stored in manifests.

## Immediate Open Questions To Confirm Before Experiment 01 Build

1. Exact local one-shard paths for train and validation.
2. Exact full-data root path to be used for server 15-20 epoch run.
3. Primary supervised objective for Experiment 01:
- Stage 1: Flow + SSI
- Stage 2: Flow + SSI + Wavelet Edge + Ordinal
4. Priority baseline comparators for this cycle (for example: PI-HAF, UNet-SR, EDSR-like, prior failed setup).
5. Acceptance criteria for promotion to full run (tuple P/T/Q trend thresholds and stability thresholds).
6. Artifact tracking policy for this PI-HAF program:
- which of `metrics/`, `plots/`, `reports/`, and `manifests/` are always tracked in git.
- whether any checkpoints are tracked or all checkpoints remain untracked.

## Notes For Future Chats

- This file is the operational source of truth for PI-HAF experiment execution.
- Keep updates concise and append only facts that change execution or decisions.
