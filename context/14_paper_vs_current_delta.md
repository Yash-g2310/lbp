# Paper vs Current Delta

Last reviewed: 2026-04-06
Scope: explicit mapping between LayeredDepth paper baseline and current non-PIHAF implementation in this repository.

## Summary Table

| Dimension | Paper baseline | Current implementation | Status | Why it matters |
|---|---|---|---|---|
| Conditioning input | RGB + uniform index map (4ch total) | RGB (3) + DINO features (384) + prompt map (1) = 388ch fused input | mismatch | model capacity, runtime, and inductive bias differ |
| Layer command style | uniform index map per pass | scalar `target_layer` projected to uniform spatial prompt map | partial match | both are global per-pass conditioning |
| Backbone family | pretrained monocular depth models (DepthAnything/ZoeDepth/Metric3D) | `DINOSFIN_Architecture_NEW` (frozen DINO encoder + SFIN/RHAG U-Net decoder) | mismatch | baseline comparison must not assume same architecture class |
| Freeze strategy | model-dependent (Metric3D decoder-only best) | encoder frozen, decoder trainable | partial match | closest to Metric3D-style partial freezing |
| Supervised objective | SSI-style scale/shift-invariant depth supervision | `SILogLoss` implementation | mismatch | metric trends are comparable in spirit, not identical in formula |
| Optimizer | AdamW | AdamW | match | comparable optimizer family |
| Scheduler | OneCycle | cosine annealing | mismatch | convergence dynamics and LR trajectory differ |
| Training epochs | 10 | config-driven (local 20, server 100) | mismatch | run budget and final metrics not directly comparable |
| Training data source | LayeredDepth-Syn only | LayeredDepth-Syn only | match | preserves synthetic-only supervised training contract |
| Real-data fine-tune | not used | not used for supervised loss | match | preserves zero-shot real benchmark framing |
| Real evaluation | tuple ranking P/T/Q | tuple ranking P/T/Q (`pairs`, `trips`, `quads`) | match | same benchmark family |
| Multi-pass inference | one forward pass per requested layer index | two-pass training/eval in current path (`target_layer=1/2`) | partial match | current code uses explicit layer routing per pass |

## Verified Current Anchors

- Wrapper and conditioning: `src/lbp_project/models/wrapper.py`
- Train loop and optimizer/scheduler: `src/lbp_project/training/main.py`
- Loss implementation: `src/lbp_project/utils/losses.py`
- Tuple evaluation logic: `src/lbp_project/utils/metrics.py`

## Practical Recommendation

When reporting experiment results, keep two labels explicit:

1. `paper-aligned baseline intent` (synthetic-only, multi-pass layer conditioning, tuple P/T/Q real eval)
2. `repository-specific implementation` (DINO+SFIN/RHAG design choices and SILog/cosine training stack)

This avoids accidental one-to-one claims where architecture or objective differs.
