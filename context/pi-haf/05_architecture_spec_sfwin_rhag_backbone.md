# Architecture Specification: SFWIN + RHAG + DINO Backbone

Last reviewed: 2026-04-07
Scope: Target architecture contract for the updated PI-HAF layered-depth path.

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

- First fallback: DINOv3 ViT-S16+ distilled.
- If unavailable, stop and wait for access/runtime fix.
- Temporary DINOv2 fallback is disabled by default and requires explicit one-off approval.

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
3. Local eval uses auto-expanded layer prediction per tuple requirements.

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
