# Risks, Questions, and Decisions Log

Last reviewed: 2026-04-07
Scope: Track locked decisions, active risks, and unresolved questions for this program.

## Locked Decisions

1. Planning-only branch approved to move into implementation.
2. Stage A local run: 4-5 epochs.
3. Real split facts remain canonical: 1200 test + 300 validation.
4. Local real shards retained and reindexed for local evaluation checks.
5. Tuple layer coverage policy: auto-expand predictions to all required tuple layer IDs.
6. Primary backbone: DINOv3 ConvNeXt-Small distilled.
7. Fallback 1: DINOv3 ViT-S16+ distilled.
8. If fallback 1 unavailable: stop and wait for access/runtime fix.
9. DINOv2 temporary fallback: disabled by default, explicit one-off approval only.
10. Wavelet defaults: `sym4` primary, `bior3.5` fallback, level 2 default.
11. Stage A strict gate includes non-zero and improving tuple metrics.

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

## Open Questions (Still Explicit)

1. Exact wavelet component weight schedule across Stage 1 and Stage 2.
2. Exact numeric thresholds for declaring Stage A tuple trend "improving".
3. Exact command-level protocol for interactive fallback switch and logging.
4. Whether Stage B launches ablation branch immediately or only after baseline completion.

## Decision Update Protocol

When any decision changes:

1. Update this file first.
2. Update `02_experiment_master_plan.md` if phase scope changes.
3. Update affected protocol docs (`03`, `04`, `08`) if runtime behavior changes.
4. Record date and rationale for the change.
