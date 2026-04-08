# Architecture Specification: SFWIN + RHAG + DINO Backbone

Last reviewed: 2026-04-08
Scope: Target architecture contract for the updated PI-HAF layered-depth path.

Implementation note (2026-04-07): active runtime imports for this phase are from `src/lbp_project` only.

Status note (2026-04-08): this file contains both implemented contracts and approved target contracts. Target contracts must not be treated as active defaults until implementation is merged.

## High-Level Architecture Intent

1. Keep DINO backbone family frozen.
2. Use updated PI-HAF path for layered-depth modeling.
3. Replace frequency-first interaction path with spatial wavelet interaction (SFWIN-first).
4. Keep RHAG active as a required component.
5. Ensure SFIN/SWIN/SFWIN compatibility is explicit and testable.

## Backbone Contract

Primary baseline:

- DINOv3 ConvNeXt-Small distilled.

Fallback policy:

- Primary gate: verify DINOv3 ConvNeXt-Small distilled weights are accessible before launch.
- If primary is unavailable: stop and ask user interactively for next action (path fix, fallback approval, or defer run).
- First fallback: DINOv3 ViT-S16+ distilled (user-approved only).
- If fallback is also unavailable: stop and ask user interactively; do not continue with silent fallback.
- Temporary DINOv2 fallback remains disabled by default and requires explicit one-off approval.

Wiring note:

- Backbone policy/loader is modularized in `lbp_project` and supports strict primary-stop mode plus explicit opt-in fallback candidates.

## Distillation Clarification

1. ViT-distilled and ConvNeXt-distilled are different student families.
2. ConvNeXt backbone yields hierarchical feature characteristics.
3. Integration contract must state feature selection/fusion interface for downstream decoder path.

## Interaction Block Contract

1. Required primary interaction path: SFWIN.
2. Frequency-first path is not primary in this baseline.
3. Frequency-only and hybrid interaction remain ablation branches.

## RHAG Contract

1. RHAG stays active in baseline path.
2. RHAG compatibility with SFWIN path must be validated in Stage A.

## Compatibility Matrix (Must Be Tested)

Required matrix in implementation verification:

1. SFWIN + RHAG (primary baseline)
2. SFIN + RHAG (compatibility sanity)
3. SWIN + RHAG (compatibility sanity)
4. SFIN/SWIN/SFWIN interface checks:
- tensor shape contracts
- dtype/precision behavior
- memory footprint at target resolution

## Conditioning Contract

1. Layer-index conditioning must remain explicit and deterministic.
2. Invalid or out-of-range layer IDs must trigger clear runtime behavior.
3. Local eval uses per-image required-layer expansion with deduped layer inference per tuple requirements.

## AdaLN-Zero Contract (Approved Target, Full Scope)

1. Conditioning input for normalization blocks includes:
- layer index embedding,
- flow timestep embedding.

2. Conditioning application requirement:
- propagate adaptive modulation through residual blocks (not entry-only conditioning).

3. Initialization requirement:
- zero-initialized modulation path to preserve stable identity behavior at startup.

4. Implementation status:
- current runtime is not yet full AdaLN-Zero in decoder blocks; migration work is required.

## Dimensional Bridge and Window Safety

1. Explicit bridge is required where 2D feature maps and attention token/window operations interact.
2. Dynamic right/bottom padding is required when feature-map size is not divisible by `window_size`.
3. Hard-fail divisibility checks are treated as pre-migration behavior and should be replaced by safe padding flow in target implementation.

## Tensor Contract Checklist

To be documented and verified in implementation:

1. Input image shape and normalization contract.
2. Backbone output interface contract.
3. Decoder/interactor expected channels and spatial scales.
4. Loss input target contract (per-layer depth targets + masks).
5. Eval contract for per-layer prediction maps.

## Architecture Risks

1. ConvNeXt hierarchical features may not align with prior token-grid assumptions.
2. Wavelet interaction may introduce stability or memory shifts in mixed precision.
3. RHAG window or feature-grid constraints can conflict with selected resolution.

Mitigation:

- Enforce Stage A shape and precision probes before server promotion.
