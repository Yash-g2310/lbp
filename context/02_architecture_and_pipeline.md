# Architecture and Pipeline

Last reviewed: 2026-04-08
Scope: repository `lbp/lbp`, active runtime path `src/lbp_project`.

## Status Labels

- **Implemented now**: verified in current code.
- **Approved target (pending implementation)**: accepted design, not fully active in runtime yet.
- **Deferred**: intentionally postponed.

## Model Path

- Wrapper: `src/lbp_project/models/wrapper.py`
- Decoder core: `src/lbp_project/models/unet.py`
- Backbone policy: `src/lbp_project/models/backbone_policy.py`
- Backbone loader/adapter: `src/lbp_project/models/backbone_loader.py`
- DINO source:
  - On-the-fly via policy-driven loader (`torch.hub` or `timm` HF path) when precompute is disabled.
  - Bypass path via `forward(..., precomputed_dino=...)` when precompute is enabled.

## Implemented Now (Code-Verified)

- Active wrapper class: `DINOSFIN_Architecture_NEW`.
- Default backbone path uses distilled DINOv3 ConvNeXt-Small (`timm/convnext_small.dinov3_lvd1689m`) via modular backbone policy/loader.
- Layer conditioning is currently done by layer-index embedding map fused at decoder input per pass (`target_layer` routed globally for each forward call).
- AdaLN-Zero conditioning is **not** fully wired across decoder blocks in current runtime.
- Training uses two forward passes per sample (`target_layer=1` and `target_layer=2`) and supervises both outputs.
- Encoder/decoder update strategy:
  - DINO encoder frozen.
  - SFIN/RHAG U-Net decoder trainable.
- Current default output path predicts positive depth maps (softplus depth output).
- Flow velocity head exists and is used only when flow mode path is enabled.
- Flow-mode path supports inverse-depth normalized supervision in $[-1,1]$ with signed SSI and wavelet/ordinal staged objectives.

## Training Objective and Optimization

- Training code supports both:
  - Legacy positive-depth supervision (`SILogLoss` + auxiliary terms), and
  - Flow-capable staged objectives (Flow/SSI/Wavelet/Ordinal branches) when configured.
- Local/server primary profiles now default to flow mode via `training.staged_losses.mode: flow_staged`.
- Optimizer: AdamW.
- Scheduler: cosine annealing (config-driven), not OneCycle.
- Curriculum weighting decays decoder and bottleneck auxiliary terms after midpoint.

## Approved Target (Pending Implementation)

1. **Conditioning redesign**:
- Replace entry-only layer prompt conditioning as primary design with full AdaLN-Zero style conditioning through decoder residual blocks.
- Conditioning input should include both layer index and flow timestep $t$.

2. **Depth-space contract for flow matching**:
- Train in inverse-depth normalized space bounded to $[-1, 1]$.
- Implemented for flow-enabled training path.
- Existing positive-depth checkpoints are not compatible with this migration and require retraining from scratch.

3. **Two-stage objective contract**:
- Stage A (epochs 0-4): Flow + SSI geometric warmup.
- Stage B (epoch 5 onward): Flow + SSI + Wavelet Edge + Ordinal.

4. **Stage runtime policy target**:
- Stage A fixed at 5 epochs.
- Stage B server policy: 25 epochs preferred, up to 30 if budget allows, with hard stop at `min(24h, 30 epochs)` and mandatory final full real evaluation.

5. **Precision policy target**:
- Mixed BF16 globally with FP32 bubble around wavelet-sensitive subregions.
- This is approved as target but not yet fully active as default runtime behavior.

## Explicit Clash Matrix (Current vs Target)

1. Conditioning:
- Current: layer embedding fused at input only.
- Target: AdaLN-Zero through decoder blocks with `(layer_id, t)`.

2. Output and training space:
- Current: dual-path support (legacy positive-depth path and flow-mode inverse-depth normalized path).
- Target: inverse-depth normalized to $[-1,1]$ for flow.

3. Loss activation profile:
- Current: mixed support exists, but default profile activation is not fully PI-HAF target.
- Target: fixed Stage A/B flow curriculum semantics.

4. Runtime stop contract:
- Current: epoch-loop driven.
- Target: dual cap (`24h` or `30 epochs`) with mandatory final full real eval.

## Data Path

- Loader: `src/lbp_project/data/dataset.py`
- Startup preflight and feature-index compatibility checks: `src/lbp_project/data/preflight.py`
- Train/val loaders use:
  - `data.train_dataset_name`, `data.train_split`
  - `data.val_dataset_name`, `data.val_split`
- Real tuple eval uses `scripts/eval/eval_real_tuples.py` with split/layer fallback logic.

### Supervision Split Contract

- Dense supervised depth training/validation: `princeton-vl/LayeredDepth-Syn`.
- Real benchmark tuple evaluation: `princeton-vl/LayeredDepth` only.
- Real tuple data is not used for dense supervised training losses.

## Execution Path

- CLI entry: `src/lbp_project/cli.py`
- Commands dispatch to:
  - `train.py`
  - `scripts/training/train_eval.py`
  - `scripts/quickcheck/run.sh`
  - `scripts/eval/eval_real_tuples.py`
  - `scripts/eval/eval_checkpoints.py`

## Tuple Evaluation Semantics

- Tuple annotations are loaded from `tuples.json` with layer roots such as `layer_all` and `layer_first`.
- `layer_all` and `layer_first` are benchmark subset keys; implementation treats the selected `layer_key` generically.
- Accuracy is reported for:
  - pairs (P)
  - trips (T)
  - quads (Q)
- Correctness is computed from pairwise relative-depth ordering constraints among tuple points.
- Multi-pass real eval uses per-image required-layer extraction with deduped layer inference.
- Missing-layer tuples are tracked explicitly in reporting diagnostics.
- Real eval scripts support split/layer-key fallback to avoid empty-report failure modes.

## Why This Matters

The project has two evaluation modalities:

1. Synthetic dense-depth validation (numeric depth metrics).
2. Real tuple ranking evaluation (pair/trip/quad ordering logic).

These should not be mixed when reasoning about model quality.

For PI-HAF flow migration, keep an explicit distinction between:

1. What is active in runtime today.
2. What is approved target and pending implementation.
