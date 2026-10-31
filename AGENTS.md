# Agent notes for Hyperspectral Image Reader V2

This file is for coding agents maintaining the repository. The public project overview and user instructions belong in `README.md`.

## Architecture map

- `read_dataset_image.py`: public `load_hypervision_matrix()` wrapper, `DATASET_EXTENSIONS` registry, output scaling and command-line interface.
- `hyperspectral_pipelines.py`: `LoadHyperspectralImage`, format dispatch, dataset-specific readers, HWC layout handling, wavelength resolution, optional channel append and metadata.
- `wavelength_configs.py`: `WAVELENGTH_CONFIGS`, exact channel-center definitions, and `resolve_wavelengths()`.
- `false_rgb.py`: RGB band selection, fixed display profiles, per-channel image min-max fallback, and FalseRGB metadata.
- `utils/ds_load.py`, `utils/hsi3d.py`: inherited dataset-specific loading and image conversion helpers.
- `tests/test_false_rgb.py`: regression coverage for FalseRGB selection, display scaling, loader behavior and representative real dataset files.

## Contracts to preserve

- Cube convention is generally `(H,W,C)`. Some files have dataset-specific axis conventions; do not assume `_ensure_hwc()` can resolve every ambiguous 3D layout.
- `append_rgb=False` preserves the historical per-image HSI min-max mapping to `[0,255]` in the wrapper.
- New spectral FalseRGB output is appended after HSI channels. Output shape is `(H,W,C+3)` and returned `wavelengths` describes only the `C` spectral channels.
- Default `rgb_value_mode="display"` maps appended RGB to `[0,255]`: use a verified dataset/product profile when available, otherwise independently min-max normalize each selected RGB channel over the current image. Constant channels become zero. This fallback is visualization normalization, not physical calibration, and values are not comparable across images.
- Explicit `rgb_value_mode="copy_hsi"` copies selected spectral samples and does not guarantee an RGB range of `[0,255]`.
- Old dataset branches may append paired color images using their legacy path. Preserve the source behavior where compatibility matters, while keeping appended values in the advertised output range.
- Metadata is part of the API. Preserve the distinction between source HSI wavelengths, appended channel descriptors, value mapping, and spatial alignment.
- Wavelength provenance is exposed as `wavelength_source`, `wavelength_verified`, and `wavelength_status` (`verified`, `estimated`, or `unknown`). Range-interpolated values may be returned for display/compatibility, but must not be passed to wavelength-targeted FalseRGB selection; auto mode may use ordered channel quantiles when channel order is known.
- Explicit estimated-center matching is opt-in via `allow_estimated_wavelengths=True`; record that choice in FalseRGB metadata. `auto` mode continues to avoid estimated-center matching even when the flag is set, and falls back to channel quantiles when order is known.

## Data registration and extension points

For a new dataset, inspect the actual file layout, dtype, axis order, channel count, wavelength source, companion files and product variants before changing code.

1. Register its dataset identifier and accepted suffix in `DATASET_EXTENSIONS`.
2. Implement or route loading in `LoadHyperspectralImage.load()` and the relevant format helper.
3. Add wavelength resolution in `WAVELENGTH_CONFIGS` or consume trustworthy file metadata. For multiple products, branch on validated channel counts or product path information.
4. Add the dataset to the ordered-spectrum policy only when its channel axis is known to follow spectral order. Do not assume a plausible interpolated wavelength vector proves the physical wavelength centers.
5. Register a fixed display profile in `_PROFILE_SPECS` only with product-level evidence and enforce path, dtype and channel-count restrictions. Otherwise retain the RGB per-image min-max fallback.
6. Keep the source cube separate from already appended channels. FalseRGB selection requires the cube's channel count to match the spectral wavelength vector.
7. Add/update tests for layout, wavelength length, RGB selection, `[0,255]` output, constant-channel behavior, channel order and metadata. Real-sample tests should state which product/sample they cover.
8. Update `README.md` with user-relevant support and usage information; put decoder, calibration and fixture implementation notes here.

## Dataset-specific implementation notes

### Archived and tabular products

- `GHIFVD`: compressed `pca/*.mat` cubes are decoded using `wc`, `pcc`, `wid`, `hei` and the local `matlabHyper/hyperWavelengths.csv`. The equivalent reconstruction is `pcc @ wc.T`, then MATLAB-compatible column-major HWC reshape. Validate all dimensions and wavelengths. Color-space and PCA display products are not interchangeable with the HSI cube.
- `HyperLeaf2024`: the source TIFF is `(204,48,352)`. Preserve the historical no-append CHW layout and report `spectral_axis=0`; RGB append explicitly transposes to HWC and reports `spectral_axis=-1`. Both paths must return the same 204 official centers in source order. Do not infer the uint16 scale multiplier/offset from observed sample extrema. Its FalseRGB display fallback normalizes only selected RGB channels.
- `Hyper-Skin`: channel count and VIS/NIR/MSI product identify wavelength configuration. The full-cube path must remain separately qualified from products that have local samples.
- `HFD100`: 31-channel truth and 256-channel full-cube products have different wavelength configurations.
- `BeyondRGB`: HDF5 `MIS` contains a 2D 4×4 CFA mosaic. `_unpack_beyondrgb_cfa()` reversibly unpacks 16 phase-sampled channels from the documented pattern without interpolation, demosaicing or registration. Numerical spectral response centers are unavailable locally; keep wavelength centers unknown and preserve phase offsets/alignment metadata.
- `Ladybird Cobbitty`: Pika-XC2 PNG rows are spectral samples and columns are spatial positions. The reader transposes one PNG to `(1, spatial, spectral)`, using `calibrations/pika-xc2/spectra.txt`. It intentionally does not combine neighboring scans into a scene.
- `Almond/Pistachio/Garlic`: split-local dark/white references are validated before computing `(raw-dark)/(white-dark)`. Calibration requires matching headers, dimensions, wavelengths, exposure and acquisition date. If calibration cannot be established, display mode retains raw HSI values and falls back to RGB-only per-image normalization. Record the failure reason in metadata.

### Dataset files requiring companion handling

- HSIFoodIngr-64 and Hydrocarbon Spill are commonly distributed in archives. An extracted sample needs its ENVI header and matching binary data file; callers may pass the header or matching data path.
- CloudPatch-7 BIP files may use a paired header with an appended suffix (for example, `.bip.hdr`); select the header paired with the actual sample instead of an unrelated header in the directory.
- HyperPRI nested binary files can share the unique dataset-root `hinalea_hsi.hdr`; reject ambiguous root-header matches.

## Wavelength and FalseRGB rules

- `resolve_wavelengths()` prioritizes valid header wavelengths, then dataset configuration. Its interpolation/fallback behavior exists for legacy compatibility and must not automatically be treated as measurement evidence in RGB selection.
- `rgb_mode="auto"` first seeks unique channels nearest 650/550/450 nm within 30 nm; otherwise it uses 85/50/15% spectral-channel positions only when spectral ordering is known.
- `rgb_mode="visible"` requires three distinct channels close to the fixed visible targets. `rgb_mode="non_visible"` requires three explicit R/G/B targets and three distinct selected bands outside 400–700 nm, within 30 nm.
- Two-band inputs use long-wave / mean / short-wave composition; one-band inputs repeat the band. Metadata must identify degeneracy and source components.
- Fixed profiles map from documented product-level ranges. Unknown scales use per-image min-max independently per selected RGB channel. Do not substitute cube-wide HSI normalization, paired RGB, gamma or guessed physical calibration into that fallback.

### Wavelength audit notes (2026-10)

- `fiftyoutdoor`: confirmed local MAT sample shape is `(250,329,33)`. The published dataset describes 33 samples from 400 to 720 nm at 10 nm steps; keep all 33 centers in the fallback vector.
- `umos`: configuration now follows the documented 33-band, 400–720 nm table entry, but no UM-OS fixture was located. A previously inspected 33-band Time-Lapse MAT belongs to UM-RI 2015 and is not evidence for UM-OS.
- `hsidrive20`: confirmed local NPY sample shape is `(216,409,25)`. The sensor is specified as 25 bands over 600–975 nm. Do not use the previous 400–760 nm fallback.
- `aphid`: confirmed local NPY sample has 237 channels. The current fallback spans the reported 436–965 nm interval, but the checked sample tree has no `meta.yaml`; exact centers remain unverified. If the dataset metadata becomes available, parse its `wave_lengths` vector and prefer it over the fallback.
- `arad_1k_16`: a local v7.3 MAT sample reads through the full reader as `(512,480,16)`. The published 400–1000 nm endpoints are represented by an estimated range fallback, not verified per-channel centers.
- `hs_sod`: the published product has 81 channels over 380–780 nm at 5 nm spacing. The current data mount has only a cached `(384,512,84)` NPZ; all dataset caches append 3 RGB channels, so this is 81 HSI + 3 RGB. No original MAT/HDF5 sample is present for a fresh full-reader check. Do not count cached RGB channels as HSI bands.
- `hykov2vis` and `hykov2nir`: checked MAT samples contain file wavelength vectors, which take precedence. The VIS sample has 15 channels and a vector from about 468.20 to 642.55 nm; the NIR sample has 25 channels and a vector from about 673.37 to 944.82 nm. The paper's active sensor ranges are nominal system specifications and differ from these sample vectors. Preserve embedded vectors; do not replace them with the nominal ranges when present.
- `vnihdhiatlimafb` and `virginia_tech_tree`: checked local ENVI headers contain 204 / 420 wavelength entries, spanning 397.32–1003.58 nm / 394.7–1005.86 nm. The reader should prefer those header vectors; fallback configs now reflect the checked sample products.
- `hyperdrivevnir`: a local NPZ contains `cube` with shape `(1016,1671,33)` and no wavelength vector. The reader supports NPZ cubes and reports the 660–900 nm fallback as estimated; this conflicts with the 24-band dataset description, so confirm sample/version identity and exact center wavelengths before precise spectral selection.
- `hsiroad` and HOT variants: published versions and system specifications differ. Exact per-channel centers still need matching local samples or release metadata. HOT RedNIR currently decodes a 4×4 PNG mosaic and drops its final decoded slot, returning 15 channels. A raw RedNIR frame is absent from the current mount, so do not claim the resulting final channel is zero or that only 14 channels are valid without checking the actual frame and official layout.
- `hydrocarbon_spill`: 124 checked ENVI headers declare 20 wavelengths in the same non-monotonic order: channel 0 is 1000 nm, channels 1–19 span 400–968.4 nm. Preserve the ENVI cube and header order. `wavelength_verified=True` does not imply monotonically ordered channels; automatic FalseRGB may match explicit visible targets by their header wavelengths but must not fall back to ordered-channel quantiles when matching fails.
- Real V1 samples loaded through the full API include 50 Outdoor, HSI Drive v2, Agricultural Plant, HyKo v2 VIS/NIR, HyperspectralCity v2, ARAD1K16, HyperDrive VNIR, HSI Road and HyperBlood. The HyperDrive fixture has 33 cube channels despite the 24-band dataset description and contains no wavelength vector; report the 660–900 nm range as estimated pending identity/version confirmation.
- Apple Fire Blight and Virginia Tech Tree now load through the full ENVI reader as `(512,512,204)` float32 and `(401,800,420)` uint16. Their header vectors match their last HWC axes and both cubes are finite. Do not apply `_ensure_hwc()` to arrays returned by `spectral.io.envi`; it can transpose valid `(lines,samples,bands)` cubes when dimensions are small.
- HyperBlood's checked sample returns 113 filtered bands. `ds_load.get_data()` and `LoadHyperspectralImage.load()` produce identical cubes and wavelength vectors, confirming that noisy-band filtering is applied to both.
- HyperLeaf's checked TIFF has 204 official centers in both modes. Preserve `(204,48,352)` with `spectral_axis=0` when `append_rgb=False`; append mode transposes to HWC and records `spectral_axis=-1`.
- The `hsi_reader` environment is cloned from HyperVision's Conda environment: Python 3.10.20, PyTorch 2.10.0+cu128, MMCV 2.1.0 and MMDetection 3.3.0. Reader dependencies import, `mmcv.ops.nms` runs, but CUDA is not visible in this runtime.

## Tests and local verification

Primary suite: `conda run -n hsi_reader python -m unittest discover -s tests`. The latest local run passes all 37 tests, including regression coverage for non-monotonic wavelength vectors and a real Hydrocarbon Spill ENVI sample. Set `HSI_DATASET_ROOT` to the local extracted dataset directory and `HYPERVISION_TEST_DATA_ROOT` to the optional archived fixture directory to enable real-sample checks; without those environment variables, unavailable sample checks skip. Avoid loading very large cubes unnecessarily in new tests.

## Detailed ingestion notes (developer reference)

### Archive extraction and ENVI pairs

HSIFoodIngr-64 and Hydrocarbon Spill are distributed as archive collections. Each extracted sample must retain the ENVI `.hdr` and its matching `.dat` (or other declared binary) file. The reader resolves wavelengths and cube layout through the header. Archive extraction itself is outside this library's read call.

### BeyondRGB mosaic

The `MIS.h5` two-dimensional sensor mosaic uses the 4×4 channel pattern documented in the dataset supplement. The loader unpacks it into 16 phase-sampled channels. It performs no demosaicing or cross-phase registration; no numeric spectral response table is present in the local files, so do not invent wavelength centers. FalseRGB channel choice relies on the declared channel order, and alignment metadata must state the CFA phase sampling caveat.

### Ladybird Cobbitty scanline

The local Pika-XC2 example treats each PNG row as one wavelength and each column as a spatial sample. `spectra.txt` supplies the row wavelengths. A PNG therefore represents one native scanline with output `(1, spatial, spectral)`; do not stack files or claim a reconstructed 2D scene without explicit scan/timing geometry.
