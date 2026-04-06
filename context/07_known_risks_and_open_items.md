# Known Risks and Open Items

Last reviewed: 2026-04-05

## Data and Cache Risks

1. Local cache is partial for three of the four key segments.
2. Offline load viability differs by split because local arrow availability differs.
3. Remote parquet counts and local cache shape can diverge without implying corruption.

## Evaluation Risks

1. Real tuple evaluation depends on split/layer-key fallback behavior across local and server configs.
2. Comparing synthetic depth metrics and tuple ranking metrics directly can cause wrong conclusions.

## Performance and Stability Risks

1. FFT and precision choices can affect memory and stability.
2. Precomputed DINO index integrity is a hard dependency when enabled.

## Open Follow-Ups

1. Optional cache repair matrix for failing local splits.
2. Optional dedicated CLI command for data-truth reporting.
3. Optional strict checks that fail fast when configured split/index prerequisites are missing.
