# Risks, Questions, and Decisions Log

Last reviewed: 2026-04-08
Scope: Track locked decisions, active risks, and unresolved questions for this program.

## Locked Decisions

1. Planning-only branch approved to move into implementation.
2. Stage A local run: fixed 5 epochs.
3. Real split facts remain canonical: 1200 test + 300 validation.
4. Local real shards retained and reindexed for local evaluation checks.
5. Tuple layer coverage policy: auto-expand predictions to all required tuple layer IDs.
6. Primary backbone: DINOv3 ConvNeXt-Small distilled.
7. Fallback 1: DINOv3 ViT-S16+ distilled.
8. If primary or fallback backbone is unavailable: stop and ask user interactively for next action (no silent fallback).
9. DINOv2 temporary fallback: disabled by default, explicit one-off approval only.
10. Wavelet defaults: `sym4` primary, `bior3.5` fallback, level 2 default.
11. Stage A strict gate includes non-zero/improving tuple metrics and end-of-epoch validation logs.
12. Stage B validation cadence is every 10 epochs, with loss components logged each epoch.
13. Loss stage boundaries and weights are config-driven (anti-loss-soup guardrail).
14. Current implementation wave runs from `src/lbp_project` only; `src/pi-haf` is reference-only.
15. Startup preflight now enforces mandatory assets and precomputed-feature compatibility via shared modules.
16. `.gitignore` now blocks new untracked additions under `src/pi-haf/` while leaving already tracked files unchanged.
17. Stage workflows are now first-class commands: `cli.py stage-a` and `cli.py stage-b`.
18. Train/eval orchestration now emits run manifests with config hash + git metadata for reproducibility tracking.
19. Stage B server profiles default to `periodic_real_eval_every_epochs=10`.
20. Server configs `default`, `balanced`, and `stable` explicitly set `architecture.backbone_fallback_approved=false`.
21. Stage A/Stage B dry-runs are validated as wiring checks only when required artifacts exist (checkpoint for Stage A skip-train, evidence reports for Stage B gate).
22. Server startup checks now enforce `hardware.min_vram_gb` against detected GPU VRAM (hard failure on mismatch).
23. Fixture support uses existing quickcheck paths and switches (`EMIT_STAGE_GATE_FIXTURE`) rather than introducing dedicated tracked fixture files.
24. Stage A skip-train checkpoint resolution now includes quickcheck checkpoint fallback after configured checkpoint paths.
25. Documentation mode is locked to "implemented now" vs "approved target" labeling.
26. PI-HAF flow migration target is immediate requirement: inverse-depth normalization to `[-1,1]`.
27. Empty-space handling target is dataloader-enforced `L1 = L2` contract for opaque/no-front-layer cases.
28. AdaLN-Zero scope is full conditioning path with `(layer_id, timestep)` through decoder residual blocks.
29. Precision target is mixed BF16 global + FP32 bubble on wavelet-sensitive blocks (documented as target pending implementation).
30. Stage B runtime contract target is dual-cap: 25 preferred, up to 30, hard stop at `min(24h, 30 epochs)`.
31. Final-stop full real evaluation is mandatory at Stage B stop (1200-image benchmark contract).
32. Depth-space migration is a retraining boundary; no checkpoint compatibility shim.

## Clash Options and Chosen Paths

1. Depth representation clash:
- Current: positive-depth default supervision.
- Options considered: keep positive, inverse-positive, inverse-normalized `[-1,1]`.
- Selected: inverse-normalized `[-1,1]` as migration target.

2. Empty-space supervision clash:
- Current: independent per-layer supervision.
- Options considered: doc-only, loss-only consistency, dataloader enforcement, hybrid.
- Selected: dataloader-level enforcement (`L1 = L2` for opaque/no-front-layer regions).

3. Conditioning clash:
- Current: entry-level layer prompt conditioning.
- Options considered: RHAG-only partial AdaLN, stub-first, full scope.
- Selected: full-scope AdaLN-Zero target.

4. Precision-policy clash:
- Current: mixed precision with stability guards, but not full target policy contract.
- Options considered: global FP32, global BF16, mixed BF16 + FP32 bubble.
- Selected: mixed BF16 global + FP32 wavelet bubble target.

5. Stage boundary/runtime clash:
- Current: stage policy and cadence support exists.
- Options considered: fractional, fixed, hybrid.
- Selected: fixed Stage A (5 epochs) + Stage B dual-cap runtime policy.

## Key Risks

1. Backbone access/runtime risk:
- Distilled checkpoint access or dependency mismatch can block Stage A start.

2. Architecture transfer risk:
- ConvNeXt hierarchical feature contract can conflict with assumptions from token-grid pipelines.

3. Wavelet integration risk:
- Precision/memory instability under mixed precision if block/loss path is not numerically guarded.

4. Loss-coupling risk:
- Flow + SSI + wavelet + ordinal can destabilize if stage boundary and weights are not controlled.

5. Data interpretation risk:
- Confusing local materialization with canonical split definitions can invalidate conclusions.

6. Eval validity risk:
- Missing layer coverage in tuple evaluation can yield misleading scores if not explicitly reported.

7. Environment drift risk:
- Conda/pip cleanup or package changes can silently break torch/cuda compatibility.

8. Dry-run fixture risk:
- Stage-gate verification commands can look like runtime regressions when required evidence/checkpoint artifacts are missing.
9. Hardware-profile mismatch risk:
- Running server profiles on lower-VRAM hosts now fails early by policy; operators must select the correct profile for available GPU class.
10. Migration-gate drift risk:
- Depth-space migration changes loss/metric distributions; promotion thresholds may drift unless recalibrated before promotion.
11. Documentation drift risk:
- If target behavior is documented as active too early, operators may assume unsupported runtime behavior.

## Open Questions (Still Explicit)

1. Exact wavelet component weight schedule across Stage 1 and Stage 2.
2. Exact numeric thresholds for declaring Stage A tuple trend "improving" after depth-space migration.
3. Whether Stage B launches ablation branch immediately or only after baseline completion.
4. Whether fixture evidence generation should be part of default quickcheck CI path or remain opt-in per run.

## Decision Update Protocol

When any decision changes:

1. Update this file first.
2. Update `02_experiment_master_plan.md` if phase scope changes.
3. Update affected protocol docs (`03`, `04`, `08`) if runtime behavior changes.
4. Record date and rationale for the change.
