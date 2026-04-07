# PI-HAF Program State And Experiment Protocol

Last reviewed: 2026-04-07
Scope: repository `lbp/lbp` only.

## Current Program Status (2026-04-07)

1. Planning-to-implementation transition has started through a modular context pack under `context/pi-haf/`.
2. Active implementation source-of-truth is now split across `02` to `11` modules (master plan, stage protocols, architecture/loss specs, data/eval contracts, risks, env runbook, ablations).
3. This file remains the high-level operational anchor; detailed execution contracts live in the modular docs.

## Confirmed Current State

1. Local and server branches are synchronized after push/pull in both directions.
2. Proposal reference lives at `context/pi-haf/GSoC2026_Proposal_YashGupta_DeepLense_final.tex`.
3. PI-HAF source code is present at `src/pi-haf/`.
4. Notebook history for PI-HAF exists outside this source package; source tree currently contains the runnable modules and training/eval code.
5. Local debug data has been reduced to one shard each for train and validation.
6. The prior failed attempt was a separate legacy DINOv2 + SFIN + RHAG U-Net run, not the PI-HAF codepath.

## Program Goal

Run multiple controlled experiments across models and pipelines, then pick the strongest candidate for a full 100-epoch train on complete data.

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
- `exp_20260406_02_pihaf_unsup_real`

Run tags inside each experiment:

- `local_smoke`
- `local_short`
- `server_mid_20ep`
- `server_full_100ep`

## PI-HAF Technical Anchors To Keep Consistent

1. Rectified-flow training target with Euler ODE sampling (`src/pi-haf/engine/train_sr.py`).
2. Task6B two-stage LoRA fine-tuning contract and strict shape checks (`src/pi-haf/engine/task6b_train.py`).
3. PI-HAF config constraints (including expected LR shape logic) (`src/pi-haf/models/pi_haf/config.py`).
4. Evaluation contract expects SR outputs in [0,1] and HR converted from [-1,1] (`src/pi-haf/engine/eval_sr.py`).

## Minimal Verification Checklist Per New Experiment

1. Data contract checks:
- LR/HR shapes satisfy exact 2x relation where required.
- Split sizes and shard usage are logged.
- No NaN/Inf in loaded tensors.
2. Objective checks:
- Every loss term finite and numerically scaled.
- Loss weights and schedules logged each epoch.
3. Optimization checks:
- Learning rate and stage transitions are logged.
- AMP mode and dtype are recorded.
- Gradient norms monitored; clip threshold justified.
4. Metrics checks:
- PSNR/SSIM trends and sample counts consistent.
- For unsupervised runs, include proxy/physics consistency diagnostics.
5. Reproducibility checks:
- Git commit hash, seed, and config fingerprint stored in manifests.

## Immediate Open Questions To Confirm Before Experiment 01 Build

1. Exact local one-shard paths for train and validation.
2. Exact full-data root path to be used for server 15-20 epoch run.
3. Primary objective for Experiment 01:
- supervised flow + mass (+ optional freq), or
- unsupervised cycle + patch + mass.
4. Priority baseline comparators for this cycle (for example: PI-HAF, UNet-SR, EDSR-like, prior failed setup).
5. Acceptance criteria for promotion to full run (minimum PSNR/SSIM or physics metric thresholds).
6. Artifact tracking policy for this PI-HAF program:
- which of `metrics/`, `plots/`, `reports/`, and `manifests/` are always tracked in git.
- whether any checkpoints are tracked or all checkpoints remain untracked.

## Notes For Future Chats

- This file is the operational source of truth for PI-HAF experiment execution.
- Keep updates concise and append only facts that change execution or decisions.
