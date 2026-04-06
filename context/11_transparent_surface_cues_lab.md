# Transparent Surface Cues Lab

Last reviewed: 2026-04-05

## Implementation Surface

Notebook:

- `notebooks/transparent_surface_cues_lab.ipynb`

Output artifacts:

- `artifacts/insights/transparent_cues/`

## Implemented Experiments

1. FFT high-pass frequency segregation.
2. FRFT order sweep (local, numpy-based, resized for fusion).
3. HSV specular split (V/S thresholding).
4. Distortion and line-bending cue map.
5. Blur/focus residual cue map.
6. Multiscale detail persistence cue.
7. Cross-cue weighted fusion map.

## Local Data Strategy

Uses directly available Arrow shards from local cache to avoid missing-shard failures in HF split loaders:

- Synthetic: first available train Arrow shard.
- Real: first available validation Arrow shard.

## Quantitative Sanity Checks in Notebook

- Synthetic depth-edge alignment ratio for FFT and fusion maps.
- Real tuple probe via `evaluate_tuple_sample` on fusion cue map (direct/inverted orientation tested).
- Cross-cue correlation matrix on first sample.

## Current Observations (First Run)

- Pipeline executed on synthetic and real subsets with saved panel visualizations.
- Validation checks in notebook passed.
- Fusion cue currently shows mixed tuple performance on real samples, indicating useful signal but not sufficient as standalone depth proxy.

## Suggested Next Integration Step

Use top cue channels as auxiliary model inputs:

- `fft_map`
- `specular_mask`
- `distortion_map`
- `blur_map`

and test a lightweight cue-consistency auxiliary loss around high-cue regions.
