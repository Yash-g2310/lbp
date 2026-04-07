# Ablation Matrix: Wavelet vs Frequency vs Hybrid

Last reviewed: 2026-04-07
Scope: Post-baseline ablation roadmap. Not part of Stage A/B critical baseline path.

## Purpose

After baseline completion, compare interaction/loss variants to quantify quality vs stability trade-offs.

## Ablation Tracks

1. Wavelet-only baseline (primary target path)
- Interaction: SFWIN
- Edge regularization: wavelet edge loss

2. Frequency-only branch
- Interaction: frequency-first path
- Edge regularization: frequency edge/focal term

3. Hybrid branch
- Interaction: wavelet + frequency combined
- Edge regularization: blended wavelet/frequency schedule

## Controlled Variables

Keep fixed across ablations when possible:

1. Data splits and materialization policy.
2. Backbone variant (unless backbone ablation is the explicit objective).
3. Stage schedule boundaries.
4. Eval/reporting schema.

## Backbone Sub-Ablation Note

If required after baseline:

1. ConvNeXt-Small distilled (primary baseline).
2. ViT-S16+ distilled (fallback baseline).

Only run backbone ablation when baseline pipeline is stable and reproducible.

## Evaluation Outputs Per Ablation

1. Synthetic per-layer and aggregate metrics.
2. Real tuple metrics with coverage diagnostics.
3. Stability summary:
- finite-loss rate
- gradient health
- runtime/memory notes

## Promotion Criteria

Ablation variant is promoted only if:

1. Improves key eval metrics without violating stability gates.
2. Maintains valid tuple coverage behavior.
3. Does not introduce unacceptable runtime or memory regression.

## Priority Order

1. Complete baseline first.
2. Run frequency-only branch.
3. Run hybrid branch.
4. Run any additional backbone ablation only if needed.
