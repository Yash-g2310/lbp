# Known Risks and Open Items

Last reviewed: 2026-04-08

## Data and Cache Risks

1. Local cache is partial for three of the four key segments.
2. Offline load viability differs by split because local arrow availability differs.
3. Remote parquet counts and local cache shape can diverge without implying corruption.
4. `configs/local/quickcheck.yaml` can fail on this server because `/home/yash_gupta/...` is not writable here.
5. `configs/server/quickcheck.yaml` can fail on local hosts when `/mnt/home2/...` is not writable.
6. Fixture-assisted quickcheck runs still depend on valid dataset/index availability; fixture emission does not bypass broken data contracts.

## Evaluation Risks

1. Real tuple evaluation depends on split/layer-key fallback behavior across local and server configs.
2. Comparing synthetic depth metrics and tuple ranking metrics directly can cause wrong conclusions.
3. Inconsistent validation cadence between local and server runs can make trend comparisons misleading.
4. Stage A dry-run with `--skip-train` fails when required checkpoint artifacts are absent.
5. Stage B dry-run fails early by design when promotion evidence reports are missing.
6. Fixture-emitted gate evidence can still fail strict thresholds if `pairs_acc` is below configured gate minimum.

## Performance and Stability Risks

1. FFT and precision choices can affect memory and stability.
2. Precomputed DINO index integrity is a hard dependency when enabled.
3. ConvNeXt-distilled checkpoint availability can block startup until interactive fallback decision is made.
4. Local (8GB-class) and server (48GB-class) hardware profiles require different safe operating ranges.

## PI-HAF Flow Migration Risks (High Priority)

1. **Dimensionality clash risk**:
- ConvNeXt path emits 2D feature grids `[B, C, H, W]` while RHAG attention operations may expect tokenized or window-partitioned layouts.
- Mitigation target: explicit shape bridge and validated reshape contract between wavelet/spatial branches and attention blocks.

2. **Mixed-precision underflow risk (wavelet-sensitive regions)**:
- High-frequency coefficients can collapse under low-precision paths in unstable AMP configurations.
- Mitigation target: mixed BF16 global path with explicit FP32 precision bubble for wavelet-sensitive computation blocks.
- Implementation status: approved target, pending full default migration.

3. **RHAG window divisibility crash risk**:
- Window partition currently hard-fails if `H` or `W` is not divisible by `window_size`.
- Mitigation target: dynamic right/bottom padding before window partition and safe unpadding after reverse projection.

4. **Depth-space migration risk**:
- PI-HAF flow contract requires inverse-depth normalized to `[-1,1]`; current active path uses positive depth assumptions.
- Migration will change optimization statistics and gate behavior.

5. **Checkpoint compatibility risk**:
- Depth representation migration is a hard boundary.
- Policy lock: retrain from scratch; legacy positive-depth checkpoints are not promoted across the boundary.

## Open Follow-Ups

1. Optional cache repair matrix for failing local splits.
2. Optional dedicated CLI command for data-truth reporting.
3. Optional harmonization of local config cache paths for multi-host portability.
4. Optional profile guidance matrix for mapping `default/balanced/stable` to detected VRAM classes in operator runbooks.
5. Implement and validate dataloader-level empty-space enforcement (`L1 = L2` rule for opaque/no-front-layer cases).
6. Implement and validate inverse-depth normalization to `[-1,1]` and post-migration gate recalibration.

Implemented note (2026-04-07): stage-policy commands (`stage-a`, `stage-b`) and strict runtime policy checks are now wired into orchestration/preflight paths.
Implemented note (2026-04-07): server configs `default/balanced/stable` now explicitly set `architecture.backbone_fallback_approved: false`.
Implemented note (2026-04-07): server preflight now enforces `hardware.min_vram_gb`; quickcheck can emit runtime gate fixtures through existing scripts.
Decision note (2026-04-08): context now tracks implemented-vs-target status explicitly; PI-HAF flow migration items are approved target and must not be treated as already active defaults.
