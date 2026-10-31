"""Evidence-based false RGB rendering from hyperspectral cubes.

The profiles below use only fixed, dataset-level ranges. They intentionally do
not infer scaling from an individual cube. Add a profile only when the stored
values' scale is documented or follows directly from the encoding.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment


VISIBLE_RGB_WAVELENGTHS_NM = (650.0, 550.0, 450.0)
MAX_VISIBLE_WAVELENGTH_DEVIATION_NM = 30.0
MAX_RGB_WAVELENGTH_DEVIATION_NM = 30.0

# Fixed scale profiles. `kind` distinguishes physical reflectance from stored
# integer codes; mapping is always dataset/product-level and never per-image.
_PROFILE_SPECS = {
    "hsi1": (0.0, 255.0, "encoded_intensity", "uint8 TIFF samples, full 8-bit code range"),
    "near_infrared": (0.0, 255.0, "encoded_intensity", "8-bit grayscale wavelength-slice PNGs"),
    "uwa_face": (0.0, 255.0, "encoded_intensity", "8-bit grayscale wavelength-slice PNGs"),
    "hsifoodingr64": (0.0, 1.0, "reflectance", "REFLECTANCE ENVI product; unitless reflectance"),
    "cabbage_eggplant": (0.0, 1.0, "reflectance", "ENVI product explicitly named Reflectance_Data"),
    "hyper_skin": (0.0, 1.0, "calibrated_reflectance", "paper documents dark/white reference radiometric reflectance calibration; 31-band VIS/NIR products"),
    "almond_pistachio_garlic": (0.0, 1.0, "calibrated_reference_ratio", "per-split local DARKREF/WHITEREF ENVI calibration using (raw-dark)/(white-dark), documented by dataset paper"),
}


def resolve_normalization_profile(dataset_type, file_path=None, source_dtype=None, channel_count=None):
    """Return a verified fixed map, enforcing product/path and dtype evidence."""
    key = dataset_type.lower()
    if key not in _PROFILE_SPECS:
        raise ValueError(
            f"False RGB is unsupported for dataset '{key}': no verified fixed normalization profile is available."
        )
    low = str(file_path or "").replace("\\", "/").lower()
    if key == "hsifoodingr64" and file_path and "reflectance_" not in low:
        raise ValueError("HSIFoodIngr-64 fixed reflectance mapping only applies to REFLECTANCE_* cubes.")
    if key == "cabbage_eggplant":
        if not file_path or "reflectance_data" not in low:
            raise ValueError("Cabbage/Eggplant fixed mapping applies only to *Reflectance_Data ENVI products.")
    if key == "hyper_skin":
        if channel_count != 31 or not ("/vis/" in low or "/nir/" in low):
            raise ValueError("Hyper-Skin fixed reflectance mapping applies only to its 31-band VIS or NIR HSI products.")
    if key == "almond_pistachio_garlic":
        p = Path(file_path or "")
        if p.stem.lower() != "data" or p.parent.name.lower() not in {"train", "test"}:
            raise ValueError("Almond/Pistachio/Garlic calibration only applies to split-local data cubes.")
        for name in ("WHITEREF", "DARKREF"):
            if not (p.parent / f"{name}.hdr").is_file() or not (p.parent / f"{name}.raw").is_file():
                raise ValueError(f"Almond/Pistachio/Garlic requires a unique split-local {name}.hdr/.raw pair.")
        if len(list(p.parent.glob("WHITEREF*.hdr"))) != 1 or len(list(p.parent.glob("DARKREF*.hdr"))) != 1:
            raise ValueError("Almond/Pistachio/Garlic reference pairing is ambiguous.")
    if key in {"hsifoodingr64", "cabbage_eggplant"} and source_dtype is not None:
        if np.dtype(source_dtype).name != "float32":
            raise ValueError(f"False RGB profile '{key}' requires the documented float32 ENVI product.")
    if key == "almond_pistachio_garlic" and source_dtype is not None:
        if np.dtype(source_dtype).name != "uint16":
            raise ValueError("Almond/Pistachio/Garlic reference calibration requires uint16 ENVI samples.")
    expected_dtype = {"hsi1": "uint8", "near_infrared": "uint8", "uwa_face": "uint8"}.get(key)
    if source_dtype is not None and expected_dtype and np.dtype(source_dtype).name != expected_dtype:
        raise ValueError(
            f"False RGB profile '{key}' requires {expected_dtype} encoded samples; got {np.dtype(source_dtype).name}."
        )
    expected_channels = {"hsi1": 96, "near_infrared": 96, "uwa_face": 33, "hsifoodingr64": 204, "cabbage_eggplant": 277, "hyper_skin": 31, "almond_pistachio_garlic": 224}.get(key)
    if channel_count is not None and expected_channels is not None and channel_count != expected_channels:
        raise ValueError(
            f"False RGB profile '{key}' is verified for {expected_channels} spectral channels, got {channel_count}."
        )
    lo, hi, kind, evidence = _PROFILE_SPECS[key]
    return {"minimum": lo, "maximum": hi, "kind": kind, "evidence": evidence, "dataset_type": key}


def map_hsi_to_display(cube, profile):
    """Apply one fixed dataset profile to every HSI band, returning float32 [0,255]."""
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Fixed HSI mapping expects an H×W×C cube; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("HSI cube contains NaN or Inf; fixed mapping cannot proceed.")
    lo, hi = float(profile["minimum"]), float(profile["maximum"])
    mapped = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * np.float32(255.0)
    metadata = {
        "method": "fixed_linear",
        "profile_kind": profile["kind"],
        "source_range": [lo, hi],
        "output_range": [0.0, 255.0],
        "evidence": profile["evidence"],
        "observed_min": float(np.min(arr)),
        "observed_max": float(np.max(arr)),
        "below_range_sample_count": int(np.count_nonzero(arr < lo)),
        "above_range_sample_count": int(np.count_nonzero(arr > hi)),
        "clipping": "clip to the profile's fixed source range",
    }
    return mapped.astype(np.float32, copy=False), metadata


def _select_unique_bands(wavelengths, targets):
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size < 3:
        raise ValueError("False RGB requires at least three spectral wavelengths.")
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError("False RGB requires finite spectral wavelengths.")
    if not np.all(np.isfinite(targets)) or targets.shape != (3,):
        raise ValueError("False RGB requires exactly three finite target wavelengths.")

    # Solve the three-channel assignment together so two RGB primaries cannot
    # accidentally select the same spectral band.
    costs = np.abs(targets[:, None] - wavelengths[None, :])
    rows, columns = linear_sum_assignment(costs)
    indices = np.empty(3, dtype=np.int64)
    indices[rows] = columns
    errors = costs[np.arange(3), indices]
    return indices, wavelengths[indices], errors


def build_false_rgb(
    cube, wavelengths, dataset_type, mode="auto", rgb_wavelengths=None,
    *, profile=None, already_mapped=False, diagnostic_cube=None,
    value_mode="display", spectral_order_known=False, wavelength_source=None,
    wavelength_verified=None, allow_estimated_wavelengths=False, channel_indices=None,
):
    """Build three H×W FalseRGB channels from an H×W×C spectral cube.

    ``copy_hsi`` returns selected source samples verbatim. ``display`` uses a
    verified fixed dataset profile when available, otherwise independently
    min-max maps each selected RGB channel in this image to [0,255]. Neither
    uses paired RGB.
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"False RGB expects an H×W×C cube; got shape {cube.shape}.")
    if wavelengths is not None and len(wavelengths) != cube.shape[-1]:
        raise ValueError(
            "False RGB cannot use a cube that already contains appended channels: "
            f"cube has {cube.shape[-1]} channels but {len(wavelengths)} spectral wavelengths."
        )

    dataset_type = dataset_type.lower()
    mode = mode.lower()
    value_mode = value_mode.lower()
    if value_mode not in {"copy_hsi", "display"}:
        raise ValueError("rgb_value_mode must be 'copy_hsi' or 'display'.")
    if mode in {"visible", "non_visible"}:
        if wavelengths is None:
            raise ValueError(
                f"rgb_mode='{mode}' requires per-channel wavelengths; the available vector is unavailable or was withheld because it is estimated."
            )
        if wavelength_verified is False and not allow_estimated_wavelengths:
            raise ValueError(
                f"rgb_mode='{mode}' received estimated wavelengths; pass allow_estimated_wavelengths=True to opt in explicitly."
            )
    if not np.all(np.isfinite(cube)):
        raise ValueError("HSI source cube contains NaN or Inf values.")
    effective_order_known = bool(spectral_order_known)
    if wavelengths is not None:
        wavelength_array = np.asarray(wavelengths, dtype=np.float64)
        diffs = np.diff(wavelength_array)
        if diffs.size and (np.all(diffs > 0) or np.all(diffs < 0)):
            effective_order_known = True

    selection_method = None
    if channel_indices is not None:
        if mode != "auto":
            raise ValueError("Explicit source channel indices are supported only with rgb_mode='auto'.")
        raw_indices = np.asarray(channel_indices)
        if raw_indices.shape != (3,) or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("False RGB channel_indices must contain exactly three integer indices in R,G,B order.")
        indices = raw_indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= cube.shape[-1]) or np.unique(indices).size != 3:
            raise ValueError("False RGB channel_indices must be three distinct indices within the source cube.")
        selected_wavelengths = (
            np.asarray(wavelengths, dtype=np.float64)[indices]
            if wavelengths is not None else None
        )
        deviations = None
        targets = None
        selection_method = "explicit_channel_indices"
    elif mode == "auto" and wavelengths is not None and cube.shape[-1] >= 3:
        try:
            indices, selected_wavelengths, deviations = _select_unique_bands(wavelengths, VISIBLE_RGB_WAVELENGTHS_NM)
            if np.all(deviations <= MAX_VISIBLE_WAVELENGTH_DEVIATION_NM):
                targets = np.asarray(VISIBLE_RGB_WAVELENGTHS_NM, dtype=np.float64)
                selection_method = "visible_wavelengths"
            else:
                indices = None
        except ValueError:
            indices = None
        if indices is None:
            targets = None
    elif mode == "auto":
        targets = None
    elif mode == "visible":
        if rgb_wavelengths is not None:
            raise ValueError("Visible mode uses its fixed 650/550/450 nm targets; omit rgb_wavelengths.")
        targets = np.asarray(VISIBLE_RGB_WAVELENGTHS_NM, dtype=np.float64)
        indices, selected_wavelengths, deviations = _select_unique_bands(wavelengths, targets)
        if np.any(deviations > MAX_VISIBLE_WAVELENGTH_DEVIATION_NM):
            details = ", ".join(
                f"{target:g}→{actual:g} nm (off by {delta:g} nm)"
                for target, actual, delta in zip(targets, selected_wavelengths, deviations)
            )
            raise ValueError(
                f"False RGB visible targets exceed the {MAX_VISIBLE_WAVELENGTH_DEVIATION_NM:g} nm tolerance: {details}."
            )
        selection_method = "visible_wavelengths"
    elif mode == "non_visible":
        if rgb_wavelengths is None:
            raise ValueError("non_visible mode requires three explicit rgb_wavelengths in R,G,B order.")
        targets = np.asarray(rgb_wavelengths, dtype=np.float64)
        if targets.shape != (3,) or not np.all(np.isfinite(targets)):
            raise ValueError("non_visible mode requires exactly three finite rgb_wavelengths.")
        if np.any((targets >= 400.0) & (targets <= 700.0)):
            raise ValueError("non_visible targets must all be outside the visible 400–700 nm range.")
        indices, selected_wavelengths, deviations = _select_unique_bands(wavelengths, targets)
        if np.any((selected_wavelengths >= 400.0) & (selected_wavelengths <= 700.0)):
            raise ValueError(
                "non_visible targets must select three source bands outside the visible 400–700 nm range."
            )
        if np.any(deviations > MAX_RGB_WAVELENGTH_DEVIATION_NM):
            details = ", ".join(
                f"{target:g}→{actual:g} nm (off by {delta:g} nm)"
                for target, actual, delta in zip(targets, selected_wavelengths, deviations)
            )
            raise ValueError(
                f"False RGB targets exceed the {MAX_RGB_WAVELENGTH_DEVIATION_NM:g} nm tolerance: {details}."
            )
        selection_method = "explicit_wavelengths"
    elif mode != "auto":
        raise ValueError("rgb_mode must be 'auto', 'visible', or 'non_visible'.")

    degeneracy = None
    if mode == "auto" and selection_method is None:
        if not effective_order_known:
            raise ValueError("Auto FalseRGB needs verified wavelengths or a known ordered spectral axis.")
        count = cube.shape[-1]
        if count < 1:
            raise ValueError("FalseRGB requires at least one source channel.")
        quantiles = np.asarray([0.85, 0.50, 0.15])
        indices = np.rint(quantiles * max(0, count - 1)).astype(np.int64)
        # Resolve collisions while preserving R,G,B ordering where possible.
        if count >= 3:
            for role in range(3):
                if indices[role] in indices[:role]:
                    options = [i for i in range(count) if i not in indices[:role]]
                    indices[role] = min(options, key=lambda i: abs(i - quantiles[role] * (count - 1)))
        selected_wavelengths = None
        deviations = None
        targets = None
        selection_method = "ordered_channel_quantiles"
        if count < 3:
            degeneracy = {"source_channel_count": int(count), "duplicate_channel_indices": count == 1,
                          "composite_mean_channel": count == 2,
                          "policy": "single band repeats; two bands map long, mean(two), short"}
            if count == 2:
                indices = np.asarray([1, -1, 0], dtype=object)
                selection_method = "ordered_channel_quantiles_with_two_band_mean"
    profile_unavailable_reason = None
    if value_mode == "display" and profile is None:
        # Prefer a dataset/product-level fixed scale. When its scale is unknown,
        # normalize only the selected RGB channels independently for this cube.
        try:
            profile = resolve_normalization_profile(
                dataset_type, source_dtype=cube.dtype, channel_count=cube.shape[-1]
            )
        except ValueError as exc:
            profile_unavailable_reason = str(exc)
    source_cube = np.asarray(diagnostic_cube if diagnostic_cube is not None else cube, dtype=np.float32)
    if not np.all(np.isfinite(source_cube)):
        raise ValueError(
            f"False RGB source bands for '{dataset_type}' contain NaN or Inf values; "
            "repair or mask invalid samples before rendering."
        )
    channel_components = [[int(i)] if int(i) >= 0 else [0, 1] for i in indices]
    if len(indices) == 3 and cube.shape[-1] == 2 and int(indices[1]) < 0:
        selected = np.stack((cube[..., 1], np.mean(cube, axis=-1, dtype=np.float32), cube[..., 0]), axis=-1)
        diagnostic_channels = np.stack((source_cube[..., 1], np.mean(source_cube, axis=-1, dtype=np.float32), source_cube[..., 0]), axis=-1)
    else:
        selected = cube[..., [int(i) for i in indices]]
        diagnostic_channels = source_cube[..., [int(i) for i in indices]]
    if value_mode == "copy_hsi":
        normalization = {"method": "copy_source_channels", "output_dtype": str(selected.dtype),
                         "image_extrema_scaling": False, "gamma": 1.0}
    elif profile is None:
        selected = np.asarray(selected, dtype=np.float32)
        channel_min = np.min(selected, axis=(0, 1), keepdims=True)
        channel_max = np.max(selected, axis=(0, 1), keepdims=True)
        span = channel_max - channel_min
        normalized = np.zeros_like(selected, dtype=np.float32)
        np.divide(selected - channel_min, span, out=normalized, where=span > 0)
        selected = normalized * np.float32(255.0)
        normalization = {
            "method": "per_channel_image_minmax",
            "profile_kind": "unverified_source_scale",
            "source_range_per_rgb_channel": [
                [float(channel_min[0, 0, i]), float(channel_max[0, 0, i])]
                for i in range(3)
            ],
            "output_range": [0.0, 255.0],
            "constant_channel_policy": "map to zero",
            "scale_scope": "each selected RGB channel independently within this image",
            "profile_unavailable_reason": profile_unavailable_reason,
            "gamma": 1.0,
        }
    elif already_mapped:
        mapped_cube = np.asarray(cube, dtype=np.float32)
        if len(indices) == 3 and cube.shape[-1] == 2 and int(indices[1]) < 0:
            selected = np.stack((mapped_cube[..., 1], np.mean(mapped_cube, axis=-1), mapped_cube[..., 0]), axis=-1)
        else:
            selected = mapped_cube[..., [int(i) for i in indices]]
        normalization = {"method": "selected_from_fixed_mapped_hsi", "profile_kind": profile["kind"],
                         "source_range": [profile["minimum"], profile["maximum"]],
                         "output_range": [0.0, 255.0], "evidence": profile["evidence"], "gamma": 1.0,
                         "clipping": "clip to the fixed profile range; negative/over-range values clipped"}
    else:
        mapped_cube, _ = map_hsi_to_display(cube, profile)
        if len(indices) == 3 and cube.shape[-1] == 2 and int(indices[1]) < 0:
            selected = np.stack((mapped_cube[..., 1], np.mean(mapped_cube, axis=-1), mapped_cube[..., 0]), axis=-1)
        else:
            selected = mapped_cube[..., [int(i) for i in indices]]
        normalization = {"method": "selected_from_fixed_mapped_hsi", "profile_kind": profile["kind"],
                         "source_range": [profile["minimum"], profile["maximum"]],
                         "output_range": [0.0, 255.0], "evidence": profile["evidence"], "gamma": 1.0,
                         "clipping": "clip to the fixed profile range; negative/over-range values clipped"}
    # Fixed-range clipping is part of the declared display mapping. Report
    # observed band statistics for diagnosis only; they never affect scaling.
    source_band_diagnostics = []
    for role_idx, (role, index, band) in enumerate(zip(
        ("R", "G", "B"), indices, (diagnostic_channels[..., i] for i in range(3))
    )):
        target = None if targets is None else targets[role_idx]
        actual = None if selected_wavelengths is None else selected_wavelengths[role_idx]
        band_min = float(np.min(band))
        band_max = float(np.max(band))
        source_band_diagnostics.append({
            "role": role,
            "target_wavelength_nm": None if targets is None else float(target),
            "source_channel_index": None if int(index) < 0 else int(index),
            "source_channel_indices": channel_components[role_idx],
            "source_wavelength_nm": None if selected_wavelengths is None else float(actual),
            "observed_min": band_min,
            "observed_max": band_max,
            "negative_sample_count": int(np.count_nonzero(band < 0.0)),
            "constant": bool(band_min == band_max),
        })

    rgb = np.asarray(selected, dtype=np.float32 if value_mode == "display" else selected.dtype)

    if degeneracy is not None:
        false_rgb_kind = "degenerate"
    elif selection_method == "visible_wavelengths":
        false_rgb_kind = "visible_band_composite"
    else:
        false_rgb_kind = "spectral_false_color"
    metadata = {
        "source": "derived_from_hsi",
        "kind": false_rgb_kind,
        "mode": mode,
        "value_mode": value_mode,
        "rgb_value_mode": value_mode,
        "channel_order": "RGB",
        "spatial_alignment": "same_source_grid",
        "selection_method": selection_method,
        "spectral_order_known": effective_order_known,
        "wavelength_source": wavelength_source or ("provided_vector" if wavelengths is not None else "unavailable"),
        "wavelength_verified": wavelength_verified,
        "wavelength_status": (
            "verified" if wavelength_verified is True else
            ("estimated" if wavelength_verified is False else "unknown")
        ),
        "allow_estimated_wavelengths": bool(allow_estimated_wavelengths),
        "degeneracy": degeneracy,
        "target_wavelengths_nm": None if targets is None else targets.tolist(),
        "selected_indices": [None if int(index) < 0 else int(index) for index in indices],
        "source_channel_indices": [None if int(index) < 0 else int(index) for index in indices],
        "source_channel_components": channel_components,
        "source_wavelengths_nm": None if selected_wavelengths is None else selected_wavelengths.tolist(),
        "wavelength_deviation_nm": None if deviations is None else deviations.tolist(),
        "source_band_diagnostics": source_band_diagnostics,
        "normalization": normalization,
        "channel_descriptors": [
            {
                "name": f"FalseRGB_{role}",
                "type": "false_rgb",
                "role": role,
                "target_wavelength_nm": None if targets is None else float(targets[role_idx]),
                "source_channel_index": None if int(index) < 0 else int(index),
                "source_channel_indices": channel_components[role_idx],
                "source_wavelength_nm": None if selected_wavelengths is None else float(selected_wavelengths[role_idx]),
            }
            for role_idx, (role, index) in enumerate(zip(("R", "G", "B"), indices))
        ],
    }
    return rgb, metadata
