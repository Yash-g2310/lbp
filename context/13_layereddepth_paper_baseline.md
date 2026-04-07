# LayeredDepth Paper Baseline

Last reviewed: 2026-04-06
Scope: concise baseline from the paper associated with the LayeredDepth datasets.

## Training Input/Output

- Input is index-concat conditioning:
  - RGB image (`H x W x 3`)
  - uniform layer-index map (`H x W x 1`), same index value at all pixels in a pass
- Effective model input: 4 channels.
- Output per pass: one dense depth map for the requested layer index.

## Backbone and Freezing Strategy

Paper reports three monocular depth backbones with model-specific fine-tune strategy:

- DepthAnything: full fine-tune.
- ZoeDepth: full fine-tune.
- Metric3D: encoder frozen, decoder fine-tuned.

The paper does not explicitly specify how the new 4th input channel was initialized.

## Optimization Setup

- Optimizer: AdamW.
- Weight decay: 0.01.
- Initial learning rate: `1e-5`.
- Scheduler: OneCycle.
- Batch size: 24.
- Hardware: 8 x A100 GPUs.
- Epochs: 10.

No explicit LoRA/distillation pipeline is reported for this baseline.

## Supervised Loss

- Loss family: scale/shift-invariant depth supervision (SSI-style baseline in paper discussion).
- Practical meaning: evaluate depth-shape consistency after scale/shift alignment, rather than relying on absolute scale only.

## Inference and Evaluation Protocol

- Multi-pass inference per image:
  - pass with layer index 1 -> layer-1 depth map
  - pass with layer index 2 -> layer-2 depth map
  - continue per required layer
- Real benchmark uses tuple-based sparse ranking constraints.
- Reported tuple metrics are categorized as:
  - Pair-wise (P)
  - Triplet (T)
  - Quadruplet (Q)

## Dataset Role Split

- LayeredDepth-Syn train (14.8k): supervised training source.
- LayeredDepth-Syn validation (500): supervised validation source.
- LayeredDepth real validation (300): tuning/eval checks before final reporting.
- LayeredDepth real test (1.2k): final zero-shot tuple evaluation.

## High-Level Takeaway

The paper baseline is synthetic-only supervised training with zero-shot transfer to real tuple-ranking evaluation.
