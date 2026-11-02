# Hyperspectral Image Reader V2

> **Recommended citation:** Guanyiman Fu et al., “HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone,” arXiv:2605.17286, 2026. [Paper](https://arxiv.org/abs/2605.17286) · [BibTeX](#citation)

Hyperspectral Image Reader V2 is a collection of readers for ground-based hyperspectral datasets. It brings together support for the datasets included in V1 and adds 27 more datasets. The readers convert supported file formats into NumPy cubes with a consistent `(H, W, C)` layout, resolve spectral wavelengths, and can append three RGB channels to an HSI cube. This repository contains the reader collection; it does not include the HyperVision backbone.

## Supported Datasets

### Datasets retained from V1

The V1 table lists 31 dataset/subset entries across 26 dataset groups.

| Dataset Name | `--dataset` Key | # Bands | Wavelengths | Expected Extension | File Loader Detail / Required Accompanying Files |
|---|---|---:|---|---|---|
| 50 Outdoor | `fiftyoutdoor` | 33 | 400–720 nm | `.mat` | Custom MATLAB loader |
| Agricultural Plant | `aphid` | 237 | 436–965 nm | `.npy` | NumPy array loader |
| ARAD1K16 | `arad_1k_16` | 16 | 400–1000 nm | `.mat` | MATLAB/HDF5 loader |
| ARAD1K31 | `arad_1k_31` | 31 | 400–700 nm | `.mat` | MATLAB/HDF5 loader |
| CAVE | `cave` | 31 | 400–700 nm | `.mat` | MATLAB loader |
| DeepHS-NIR | `deephsnir` | 252 | 950–1700 nm | `.bin` | ENVI binary; requires matching `.hdr` |
| DeepHS-VIS | `deephsvis` | 224 | 400–1000 nm | `.bin` | ENVI binary; requires matching `.hdr` |
| DeepHS-VISCOR | `deephsviscor` | 249 | 400–1000 nm | `.bin` | ENVI binary; requires matching `.hdr` |
| Harvard | `harvard` | 31 | 420–720 nm | `.mat` | MATLAB loader |
| HOT-2024-NIR | `hotnir` | 25 | 665–960 nm | `.png` | PNG mosaic loader; paired false-color `.jpg` used by legacy RGB path |
| HOT-2024-RedNIR | `hotrednir` | 15 returned channels | 600–850 nm (estimated centers) | `.png` | Decodes the 4×4 PNG mosaic and drops its final decoded slot, matching the existing reader path; paired false-color `.jpg` is used by the legacy RGB path |
| HOT-2024-VIS | `hotvis` | 16 | 470–600 nm | `.png` | PNG mosaic loader; paired false-color `.jpg` used by legacy RGB path |
| HSI Drive v2.0 | `hsidrive20` | 25 | 600–975 nm | `.npy` | NumPy loader; dataset may provide a paired pseudocolor `.png` |
| HSI Road | `hsiroad` | 25 | 600–960 nm | `.tif` | TIFF loader |
| HSODBIT v2 | `hsodbitv2` | 200 | 400–1000 nm | `.mat` | MATLAB/HDF5 loader |
| HSSOD | `hs_sod` | 81 | 380–780 nm | `.mat` / `.h5` | MATLAB v7.3 / HDF5 loader |
| HyKo v2-NIR | `hykov2nir` | 25 | 600–975 nm | `.mat` | MATLAB loader |
| HyKo v2-VIS | `hykov2vis` | 15 in the checked v2 sample | 470–630 nm sensor range | `.mat` | MATLAB loader; uses per-file wavelengths when present |
| HyperBlood | `hyperblood` | 128 raw (113 returned after default cleanup) | 377–1046 nm raw sensor range | `.mat` | Dataset-specific loader; removes noisy bands and damaged lines by default |
| HyperDrive-VNIR | `hyperdrivevnir` | 24 nominal; checked cube has 33 | 660–900 nm nominal | `.npz` | NumPy archive cube; actual channel count comes from the stored cube |
| HyperspectralCity v2 | `hyperspectralcityv2` | 128 | 450–950 nm | `.hsd` | Custom HSD binary loader |
| ICVL | `icvl` | 31 | 400–700 nm | `.h5` | HDF5 loader |
| LIB-HSI | `libhsi` | 204 | 400–1000 nm | `.hdr` | ENVI header; requires matching raw binary (`.raw`/`.dat`) |
| UM-EMM | `umemm` | 33 | 400–720 nm | `.mat` | MATLAB loader |
| UM-LD 2015 | `umld2015` | 33 | 400–720 nm | `.mat` | MATLAB loader |
| UM-NS 2002 | `umns2002` | 31 | 410–710 nm | `.mat` | MATLAB loader |
| UM-NS 2004 | `umns2004` | 33 | 400–720 nm | `.mat` | MATLAB loader |
| UM-OS | `umos` | 33 | 400–720 nm | `.mat` | MATLAB loader |
| UM-RI 2015 | `umri2015` | 33 | 400–720 nm | `.mat` | MATLAB loader |
| Virginia Tech Tree | `virginia_tech_tree` | 420 | 394.7–1005.86 nm in checked ENVI sample | `.hdr` | ENVI header; requires matching raw binary |
| Apple Fire Blight | `vnihdhiatlimafb` | 204 | 397.32–1003.58 nm in checked ENVI sample | `.hdr` | ENVI header; requires matching raw binary |

For datasets with wavelength values embedded in a file or header, the reader uses those values instead of the nominal ranges shown above. Some ranges describe a sensor's active range and should not be interpreted as a verified list of evenly spaced channel centers. The checked HyKo v2-VIS MAT sample contains 15 channels; other releases or processing variants may differ.

The reader exposes `wavelength_source`, `wavelength_verified`, `wavelength_status`, and `spectral_axis` in its wavelength metadata. Estimated wavelength ranges remain available as reference values, but automatic FalseRGB selection does not use them for wavelength-based band matching. Explicit `visible` or `non_visible` wavelength selection rejects estimates by default; `allow_estimated_wavelengths=True` (or `--allow-estimated-wavelengths`) opts in when approximate centers are acceptable.

### Dataset-specific validation notes

- **BeyondRGB:** the source CFA has 16 filter phases. The reader always unpacks all 16 phases first; `beyondrgb_band_count=12` (default) retains Band 0–11, while `beyondrgb_band_count=16` keeps all bands. Approximate response peaks are `[400, 420, 450, 470, 510, 530, 550, 590, 640, 660, 680, 710, 680, 520, 630, 530]` nm. These are estimates from Figure S1, not calibrated centers. There is no demosaicing or physical subpixel registration; appended FalseRGB copies retained MSI indices 9, 6, and 2 exactly.
- **HOT-2024 RedNIR:** the reader returns 15 channels from its 4×4 PNG mosaic. Channel centers are estimated from the nominal 600–850 nm range; use release-specific calibration data for precise spectral analysis.
- **Hydrocarbon Spill:** checked ENVI headers consistently declare 20 channels in a non-monotonic order: channel 0 is 1000 nm, followed by 400–968.4 nm in channels 1–19. The loader preserves this order. FalseRGB selection can match measured header wavelengths to source indices; when automatic wavelength matching cannot select visible bands, it must not silently fall back to channel quantiles for this dataset.

### Datasets added in V2

| Dataset Name | `--dataset` Key | # Bands | Wavelengths | Expected Extension | File Loader Detail / Required Accompanying Files |
|---|---|---:|---|---|---|
| BeyondRGB | `beyondrgb` | 16 raw / 12 default (or 16) used | ≈400–710 nm for first 12; 400–710 nm estimated peaks for all 16 | `.h5` | HDF5 `MIS` 4×4 CFA; default output 484×646×12, or 484×646×15 with FalseRGB; optional 16-band output is 484×646×16 or ×19; FalseRGB uses Band 9/6/2 |
| Cabbage & Eggplant | `cabbage_eggplant` | 277 | 399.86–949.60 nm | `.hdr` | ENVI header and paired binary cube; Reflectance and Radiance are distinct products |
| CloudPatch-7 | `cloudpatch7` | 462 | 388.86–1004.46 nm | `.bip` | BIP patch requires its matching `.bip.hdr`; MAT inputs are also registered |
| Cocoa Beans | `cocoa_beans` | 300 (some products 301) | 400–1000 nm | `.mat` | MATLAB spectral cube |
| Clinical Applications | `clinical_applications` | 100 | 400–1000 nm | `.bin` | Float32 binary cube with dataset-defined dimensions |
| DeepHS Debris | `deephs_debris` | 249 | 400–1000 nm | `.hdr` / `.bin` | ENVI header and paired binary data |
| GANs / Pinot Noir | `gans_grape` | 186 | 406.8–995.8 nm | `.tif` | TIFF spectral cube |
| GHIFVD | `ghifvd` | 345 | 376.2–821.62 nm | `.mat` | Compressed PCA cube under `pca/`; uses accompanying `matlabHyper/hyperWavelengths.csv` |
| HASCID | `hascid` | 204 | 400–1000 nm | `.npy` | NumPy spectral cube |
| HFD100 | `hfd100` | 31 / 256 | 400–700 / 376.8–1075.8 nm | `.mat` | Wavelength configuration depends on truth or full-cube product |
| HSI-1 | `hsi1` | 96 | 400–1000 nm | `.tiff` | uint8 TIFF spectral cube |
| HSIDermoscopy | `hsidermoscopy` | 16 | 400–1000 nm | `.mat` | MATLAB spectral cube |
| HSIFoodIngr-64 | `hsifoodingr64` | 204 | 397.32–1003.58 nm | `.hdr` | ENVI header and paired binary; fixed display profile is limited to `REFLECTANCE_*` products |
| Hydrocarbon Spill | `hydrocarbon_spill` | 20 | Header-defined: checked headers list 1000 nm first, then 400–968.4 nm | `.dat` | ENVI binary and matching header; preserves header/cube channel order without sorting |
| Hyper-Skin | `hyper_skin` | 4 / 31 / 448 | 4-band MSI: 450/520/600/850 nm; 31-band VIS: 400–700 nm; 31-band NIR: 700–1000 nm; 448-band: 400–1000 nm | `.mat` | MSI and 31-band products supported; 448-band full cube not locally verified |
| HyperLeaf2024 | `hyperleaf2024` | 204 | 397.32–1003.58 nm | `.tiff` | uint16 TIFF; without RGB append preserves channel-first layout; append mode returns HWC |
| HyperPRI | `hyperpri` | 299 | 402–998 nm | `.dat` | ENVI binary uses dataset-root `hinalea_hsi.hdr` |
| UGV Terrains | `ugv_terrains` | 204 | 397.32–1003.58 nm | `.hdr` | ENVI header and paired binary data |
| MobiSpectral | `mobispectral` | 204 | 400–1000 nm | `.mat` | MATLAB reconstructed spectral cube |
| Near Infrared | `near_infrared` | 96 | 1293–2215 nm | `.png` | Grayscale wavelength slices; filenames contain wavelength values |
| OMHS | `omhs` | 29 | 420–700 nm | `.mat` | MATLAB spectral cube |
| Mineral Abundance | `mineral_abundance` | 288 | 960–2500.042 nm | `.hdr` | ENVI header and paired binary; reflectance scale is not inferred |
| TOHS | `tohs` | 164 | 350–1002 nm | `.tiff` | TIFF spectral cube |
| Tecnalia | `tecnalia` | 76 | 415.05–1008.10 nm | `.mat` | MAT product with wavelength metadata; no common value scale is inferred |
| TokyoTech | `tokyotech` | 59 | 400–980 nm | `.mat` | MATLAB visible/NIR spectral cube |
| UWA Face | `uwa_face` | 33 | 400–720 nm | `.png` | Grayscale wavelength-slice stack in the same directory |
| Almond/Pistachio/Garlic | `almond_pistachio_garlic` | 224 | 400.67–997.99 nm | `.hdr` | ENVI `data` cube; optional fixed calibration requires split-local `WHITEREF` and `DARKREF` pairs |

The support descriptions summarize the implemented formats and product limits. RGB display output is mapped to `[0,255]`; when a fixed scale is unavailable, selected RGB channels are normalized independently per image.

## Features

- Reads MATLAB `.mat`, HDF5 `.h5`, NumPy `.npy/.npz`, TIFF, ENVI header/raw files, wavelength-slice image stacks, and the project-specific `.hsd` format.
- Resolves wavelengths from file headers or dataset configurations, including multi-product datasets such as Hyper-Skin and HFD100.
- Returns HSI cubes in HWC layout and provides wavelengths for spectral channels.
- Appends three RGB channels after the HSI channels with `append_rgb=True`, producing `(H, W, C+3)`.
- Maps appended RGB values to `[0,255]` by default, using a fixed profile when available or per-image min-max scaling otherwise.

## Dataset Reading

Add the project directory to your Python module path, then use the common API:

```python
from hyperspectral_image_reader_v2 import load_hypervision_matrix

path = "path/to/your/TokyoTech_sample.mat"
matrix, wavelengths = load_hypervision_matrix(
    path,
    dataset_name="tokyotech",
    return_wavelengths=True,
)

print(matrix.shape)       # (H, W, C)
print(len(wavelengths))   # C
```

Use the identifier shown in the tables as `dataset_name`. For ENVI data, pass either the `.hdr` file or its paired binary file. Follow each dataset publisher’s file layout and companion-file requirements.

### Append FalseRGB Channels

For the spectral FalseRGB path used by the added datasets, RGB is derived from three HSI channels rather than paired RGB images or previews. Existing V1 dataset-specific RGB paths retain their established behavior. The spectral path first selects distinct bands nearest to 650/550/450 nm. If suitable visible bands are unavailable, it selects bands by position when spectral channel order is known.

```python
matrix, wavelengths, metadata = load_hypervision_matrix(
    path,
    dataset_name="tokyotech",
    append_rgb=True,
    return_metadata=True,
)

print(matrix.shape)  # (H, W, C + 3)
print(matrix[..., -3:].min(), matrix[..., -3:].max())  # RGB value range: [0,255]
print(metadata["appended_channel_descriptors"])
```

By default, `rgb_value_mode="display"` maps appended channels to `[0,255]`. A verified fixed profile is used when available; otherwise, each selected RGB channel is independently min-max normalized within the current image. This fallback is intended for visualization and does not make intensities comparable across images. Constant channels map to zero. Set `rgb_value_mode="copy_hsi"` to preserve source values; this mode does not guarantee the `[0,255]` range.

For non-visible bands, provide explicit target wavelengths in R, G, B order:

```python
matrix, wavelengths, metadata = load_hypervision_matrix(
    path,
    dataset_name="near_infrared",
    append_rgb=True,
    rgb_mode="non_visible",
    rgb_wavelengths=(2100, 1750, 1400),
    return_metadata=True,
)
```

Each target must match a distinct spectral channel within 30 nm. Appended RGB channels are not included in `wavelengths`; inspect `metadata` for channel selection and value mapping details.

### Command Line

```bash
python read_dataset_image.py \
  --path /path/to/sample.hdr \
  --dataset hsifoodingr64 \
  --append-rgb \
  --output ./sample.npy
```

Optional flags are `--rgb-mode auto|visible|non_visible`, `--rgb-value-mode display|copy_hsi`, and `--rgb-wavelengths R_NM G_NM B_NM`. The default value mode is `display`.

## Pipeline Integration

The project also provides `LoadHyperspectralImage`, a callable transform compatible with MMCV/MMDetection pipelines:

```python
from hyperspectral_image_reader_v2 import LoadHyperspectralImage

loader = LoadHyperspectralImage(dataset_type="tokyotech", append_rgb=True)
results = loader({"img_path": path})
image = results["img"]
wavelengths = results["wavelengths"]
```

## Citation

```bibtex
@misc{fu2026hypervisionarxiv,
  title         = {{HyperVision: A Channel-Adaptive Ground-Based Hyperspectral Vision Pre-trained Backbone}},
  author        = {Guanyiman Fu and Jingtao Li and Zihang Cheng and Zhuanfeng Li and Diqi Chen and Yan Xu and Xiangyu Liu and Fengchao Xiong and Jianfeng Lu and Chengrong Chen and Jun Zhou},
  year          = {2026},
  eprint        = {2605.17286},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.17286}
}
```
