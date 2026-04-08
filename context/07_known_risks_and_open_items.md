# Known Risks and Open Items

Last reviewed: 2026-04-07

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

## Open Follow-Ups

1. Optional cache repair matrix for failing local splits.
2. Optional dedicated CLI command for data-truth reporting.
3. Optional harmonization of local config cache paths for multi-host portability.
4. Optional profile guidance matrix for mapping `default/balanced/stable` to detected VRAM classes in operator runbooks.

Implemented note (2026-04-07): stage-policy commands (`stage-a`, `stage-b`) and strict runtime policy checks are now wired into orchestration/preflight paths.
Implemented note (2026-04-07): server configs `default/balanced/stable` now explicitly set `architecture.backbone_fallback_approved: false`.
Implemented note (2026-04-07): server preflight now enforces `hardware.min_vram_gb`; quickcheck can emit runtime gate fixtures through existing scripts.
