# Copyright (c) OpenMMLab. All rights reserved.
"""
Enhanced Hyperspectral Dataset Pipelines for MMDetection and Computer Vision models.
Supports legacy HyperVision datasets and all datasets in hsi_datasets/datasets_new.
"""

import os
import glob
import re
from pathlib import Path
import numpy as np
import scipy.io as sio
import h5py
import tifffile
import cv2
from PIL import Image

try:
    import spectral.io.envi as envi
except ImportError:
    envi = None

from .wavelength_configs import HYPERLEAF2024_WAVELENGTHS_NM, WAVELENGTH_CONFIGS, resolve_wavelengths
try:
    from .false_rgb import build_false_rgb, map_hsi_to_display, resolve_normalization_profile
except ImportError:
    from false_rgb import build_false_rgb, map_hsi_to_display, resolve_normalization_profile

try:
    from .utils import ds_load, get_image_loader
except ImportError:
    try:
        from hyperspectral_image_reader_v2.utils import ds_load, get_image_loader
    except ImportError:
        from utils import ds_load, get_image_loader

def read_hsd(filename):
    """Read .hsd format files."""
    data = np.fromfile(filename, dtype=np.int32)
    height = int(data[0])
    width = int(data[1])
    SR = int(data[2])
    D = int(data[3])

    data_f = np.fromfile(filename, dtype=np.float32)
    a = 7
    average = data_f[a:a + SR]
    a = a + SR
    coeff = data_f[a:a + D * SR].reshape((D, SR))
    a = a + D * SR
    scoredata = data_f[a:a + height * width * D].reshape((height * width, D))

    temp = np.dot(scoredata, coeff)
    img_data = (temp + average).reshape((height, width, SR))
    return img_data.astype(np.float32)

class LoadHyperspectralImage:
    """Load hyperspectral image from diverse formats.

    This class handles loading of hyperspectral images from various file formats
    (.mat, .h5, .npy, .tif, .tiff, .hdr/.bin/.raw/.dat/.bip/.img, .png, .hsd)
    and resolves wavelengths with explicit source and verification metadata.

    Args:
        dataset_type (str): Type of dataset (e.g. 'harvard', 'tokyotech', 'hascid', etc.)
        to_float32 (bool): Whether to convert to float32. Defaults to True.
        append_rgb (bool): Whether to append synthetic RGB channels. Defaults to False.
    """

    def __init__(
        self,
        dataset_type='harvard',
        to_float32=True,
        append_rgb=False,
        rgb_mode='auto',
        rgb_wavelengths=None,
        rgb_value_mode='display',
        allow_estimated_wavelengths=False,
        beyondrgb_band_count=12,
    ):
        self.dataset_type = dataset_type.lower()
        self.to_float32 = to_float32
        self.append_rgb = append_rgb
        self.rgb_mode = rgb_mode
        self.rgb_wavelengths = rgb_wavelengths
        self.rgb_value_mode = rgb_value_mode
        self.allow_estimated_wavelengths = bool(allow_estimated_wavelengths)
        if beyondrgb_band_count not in (12, 16):
            raise ValueError("beyondrgb_band_count must be 12 or 16.")
        self.beyondrgb_band_count = int(beyondrgb_band_count)
        self.wavelengths = None
        self.wavelength_source = 'unavailable'
        self.wavelength_verified = None
        self.spectral_axis = -1
        self.wavelength_metadata = {
            'wavelength_source': self.wavelength_source,
            'wavelength_verified': self.wavelength_verified,
            'wavelength_status': 'unknown',
            'spectral_axis': self.spectral_axis,
        }
        self.bands = None
        self.channel_metadata = None
        self.beyondrgb_metadata = None

    def _ensure_hwc(self, arr):
        """Ensure the numpy array is in (H, W, C) format."""
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3:
            # If channels are first (C, H, W) where C is smaller than H, W
            # Or if C is the smallest dimension but at index 0
            if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.shape[2] < arr.shape[0] and arr.shape[2] < arr.shape[1]:
                # already H, W, C
                pass
        return arr

    def _load_envi_file(self, file_path):
        """Helper to load ENVI header (.hdr) or its paired data binary file using memmap/asarray."""
        image_path = None
        if file_path.endswith('.hdr'):
            hdr_path = file_path
            base, _ = os.path.splitext(file_path)
            for cand_ext in ['.bin', '.raw', '.img', '.dat', '.bip', '']:
                if os.path.exists(base + cand_ext) and (base + cand_ext) != hdr_path:
                    image_path = base + cand_ext
                    break
        else:
            image_path = file_path
            base, _ = os.path.splitext(file_path)
            hdr_path = base + '.hdr'
            appended_header = file_path + '.hdr'
            if self.append_rgb and os.path.isfile(appended_header):
                hdr_path = appended_header
            if not os.path.exists(hdr_path):
                if self.dataset_type == 'hyperpri':
                    candidates = []
                    for ancestor in Path(file_path).parents:
                        if ancestor.name.lower() == 'hyperpri':
                            candidate = ancestor / 'hinalea_hsi.hdr'
                            if candidate.is_file():
                                candidates.append(str(candidate))
                            break
                    if len(candidates) != 1:
                        raise FileNotFoundError(
                            f"Expected one dataset-level HyperPRI hinalea_hsi.hdr; found {len(candidates)}."
                        )
                    hdr_path = candidates[0]
                else:
                    d = os.path.dirname(file_path)
                    cand = glob.glob(os.path.join(d, '*.hdr'))
                    if cand:
                        hdr_path = cand[0]
                    else:
                        # Check parent dir or dataset root (e.g. for HyperPRI)
                        parent_hdrs = glob.glob(os.path.join(os.path.dirname(d), '*.hdr')) + glob.glob(os.path.join(d, '..', '..', '*.hdr'))
                        hdr_path = parent_hdrs[0] if parent_hdrs else file_path + '.hdr'

        hdr = envi.read_envi_header(hdr_path)
        raw_wavelengths = hdr.get('wavelength', [])

        if image_path and os.path.exists(image_path):
            img_obj = envi.open(hdr_path, image=image_path)
        else:
            try:
                img_obj = envi.open(hdr_path)
            except Exception:
                d = os.path.dirname(hdr_path)
                data_cands = glob.glob(os.path.join(d, '*.bin')) + glob.glob(os.path.join(d, '*.raw')) + glob.glob(os.path.join(d, '*.dat'))
                if data_cands:
                    img_obj = envi.open(hdr_path, image=data_cands[0])
                else:
                    raise FileNotFoundError(f"Could not find companion data file for ENVI header: {hdr_path}")

        # spectral.io.envi returns the ENVI layout directly as
        # (lines, samples, bands). The generic axis heuristic can mistake
        # relatively short image dimensions for a channels-first cube.
        arr = np.asarray(img_obj.asarray())
        return arr, raw_wavelengths

    def _calibrate_split_reference_cube(self, cube, file_path):
        """Calibrate one local almond-family ENVI cube from its split references."""
        if envi is None:
            raise ImportError("The spectral package is required for ENVI reference calibration.")
        source_path = os.path.abspath(file_path)
        folder = os.path.dirname(source_path)
        stem, ext = os.path.splitext(source_path)
        data_hdr_path = source_path if ext.lower() == '.hdr' else stem + '.hdr'
        white_hdr = os.path.join(folder, 'WHITEREF.hdr')
        dark_hdr = os.path.join(folder, 'DARKREF.hdr')
        white_raw = os.path.join(folder, 'WHITEREF.raw')
        dark_raw = os.path.join(folder, 'DARKREF.raw')

        def header(path):
            return envi.read_envi_header(path)

        data_meta, white_meta, dark_meta = map(header, (data_hdr_path, white_hdr, dark_hdr))

        def int_field(meta, key):
            return int(meta[key])

        def wave_vector(meta):
            return np.asarray(meta.get('wavelength', []), dtype=np.float64)

        for name, ref_meta in (("WHITEREF", white_meta), ("DARKREF", dark_meta)):
            if int_field(ref_meta, 'samples') != int_field(data_meta, 'samples'):
                raise ValueError(f"{name} sample width does not match data cube.")
            if int_field(ref_meta, 'bands') != int_field(data_meta, 'bands'):
                raise ValueError(f"{name} band count does not match data cube.")
            if int_field(ref_meta, 'lines') != int_field(white_meta, 'lines'):
                raise ValueError("White and dark reference line counts do not match.")
            if int(ref_meta.get('data type', -1)) != int(data_meta.get('data type', -2)):
                raise ValueError(f"{name} ENVI sample type does not match data cube.")
            ref_waves = wave_vector(ref_meta)
            data_waves = wave_vector(data_meta)
            if ref_waves.shape != data_waves.shape or not np.allclose(ref_waves, data_waves, rtol=0.0, atol=1e-6):
                raise ValueError(f"{name} wavelength vector does not match data cube.")
            data_units = str(data_meta.get('wavelength units', ''))
            ref_units = str(ref_meta.get('wavelength units', ''))
            if data_units and ref_units and data_units.lower() != ref_units.lower():
                raise ValueError(f"{name} wavelength units do not match data cube.")
            data_exposure = float(data_meta.get('exposure time', 'nan'))
            ref_exposure = float(ref_meta.get('exposure time', 'nan'))
            if not np.isfinite(data_exposure) or not np.isfinite(ref_exposure) or abs(data_exposure - ref_exposure) > 1e-6:
                raise ValueError(f"{name} exposure time does not match data cube.")
            data_day = str(data_meta.get('start time', '')).split('_')[:3]
            ref_day = str(ref_meta.get('start time', '')).split('_')[:3]
            if not data_day or data_day != ref_day:
                raise ValueError(f"{name} reference was not acquired on the data cube's date.")

        expected_data_shape = (
            int_field(data_meta, 'lines'), int_field(data_meta, 'samples'), int_field(data_meta, 'bands')
        )
        if tuple(cube.shape) != expected_data_shape:
            raise ValueError(f"Data cube H×W×C shape {cube.shape} does not match ENVI header {expected_data_shape}.")

        def load_reference(hdr, raw, meta, name):
            ref_obj = envi.open(hdr, image=raw)
            arr = np.asarray(ref_obj.asarray())
            expected = (int_field(meta, 'lines'), int_field(meta, 'samples'), int_field(meta, 'bands'))
            if tuple(arr.shape) != expected:
                raise ValueError(f"{name} array shape {arr.shape} does not match ENVI header {expected}.")
            return arr

        white = load_reference(white_hdr, white_raw, white_meta, 'WHITEREF')
        dark = load_reference(dark_hdr, dark_raw, dark_meta, 'DARKREF')
        white_f = white.astype(np.float32)
        dark_f = dark.astype(np.float32)
        raw_f = np.asarray(cube, dtype=np.float32)
        nonfinite_refs = int(np.count_nonzero(~np.isfinite(white_f)) + np.count_nonzero(~np.isfinite(dark_f)))
        nonfinite_data = int(np.count_nonzero(~np.isfinite(raw_f)))
        if nonfinite_refs or nonfinite_data:
            raise ValueError(
                f"Reference calibration has nonfinite input samples (data={nonfinite_data}, refs={nonfinite_refs})."
            )
        # References and data share the same 512-sample axis. Average only
        # across calibration scan lines so column-dependent illumination is
        # preserved when applying the references to each spatial sample.
        white_mean = np.mean(white_f, axis=0, dtype=np.float64).astype(np.float32)
        dark_mean = np.mean(dark_f, axis=0, dtype=np.float64).astype(np.float32)
        denominator = white_mean - dark_mean
        invalid_denominators = np.argwhere(denominator <= 1.0)
        if invalid_denominators.size:
            raise ValueError(
                "White-minus-dark calibration denominator is zero/near-zero at "
                f"sample/band positions {invalid_denominators[:20].tolist()}."
            )
        calibrated = (raw_f - dark_mean[None, :, :]) / denominator[None, :, :]
        invalid_calibrated = int(np.count_nonzero(~np.isfinite(calibrated)))
        if invalid_calibrated:
            raise ValueError(f"Calibration produced {invalid_calibrated} nonfinite data samples.")
        units = data_meta.get('wavelength units') or 'nm (dataset specification; ENVI header omits units)'
        metadata = {
            'method': '(raw - mean_lines(DARKREF)) / (mean_lines(WHITEREF) - mean_lines(DARKREF))',
            'profile_kind': 'calibrated_reference_ratio',
            'data_header': data_hdr_path,
            'white_reference_header': white_hdr,
            'dark_reference_header': dark_hdr,
            'white_reference_lines': int_field(white_meta, 'lines'),
            'dark_reference_lines': int_field(dark_meta, 'lines'),
            'reference_alignment': 'preserve shared sample axis; average reference scan lines only',
            'data_exposure_time': float(data_meta['exposure time']),
            'reference_exposure_times': [float(white_meta['exposure time']), float(dark_meta['exposure time'])],
            'wavelength_units': units,
            'wavelength_match': True,
            'invalid_reference_sample_count': nonfinite_refs,
            'invalid_data_sample_count': nonfinite_data,
            'invalid_calibrated_sample_count': invalid_calibrated,
            'near_zero_denominator_sample_band_count': int(invalid_denominators.shape[0]),
        }
        return calibrated, metadata

    def _load_mat_file(self, file_path):
        """Robustly load .mat file via scipy.io or h5py with whosmat inspection."""
        raw_wavelengths = None
        normalized_path = str(file_path).replace('\\', '/').lower()
        if self.dataset_type == 'ghifvd' and '/pca/' in normalized_path:
            mat = sio.loadmat(file_path, variable_names=['wc', 'pcc', 'wid', 'hei'])
            required = {'wc', 'pcc', 'wid', 'hei'}
            if not required.issubset(mat):
                raise ValueError(f"GHIFVD compressed MAT is missing {sorted(required - set(mat))}.")
            wc = np.asarray(mat['wc'])
            pcc = np.asarray(mat['pcc'])

            def scalar_dimension(key):
                value = np.asarray(mat[key]).reshape(-1)
                if value.size != 1 or not np.isfinite(value[0]) or value[0] != int(value[0]):
                    raise ValueError(f"GHIFVD MAT '{key}' must be a finite integer scalar.")
                return int(value[0])

            height, width = scalar_dimension('hei'), scalar_dimension('wid')
            if wc.ndim != 2 or pcc.ndim != 2 or wc.shape[1] != pcc.shape[1]:
                raise ValueError(f"GHIFVD PCA shapes are incompatible: wc={wc.shape}, pcc={pcc.shape}.")
            if wc.shape[0] != height * width:
                raise ValueError(
                    f"GHIFVD spatial dimensions {height}×{width} do not match {wc.shape[0]} pixel coefficients."
                )
            dataset_root = Path(file_path).parent.parent
            wavelength_path = dataset_root / 'matlabHyper' / 'hyperWavelengths.csv'
            if not wavelength_path.is_file():
                raise FileNotFoundError(f"GHIFVD decoder wavelengths file is missing: {wavelength_path}")
            all_wavelengths = np.loadtxt(wavelength_path, delimiter=',', dtype=np.float64).reshape(-1)
            # Exact slice used by matlabHyper/readCompressedDAT.m: w(20:364).
            raw_wavelengths = all_wavelengths[19:364].tolist()
            if pcc.shape[0] != len(raw_wavelengths) or len(raw_wavelengths) != 345:
                raise ValueError(
                    f"GHIFVD PCA has {pcc.shape[0]} components but decoder wavelengths contain {len(raw_wavelengths)} bands."
                )
            # Equivalent to MATLAB: spectra=pcc*wc'; reshape(spectra',hei,wid,bands).
            spectra = pcc @ wc.T
            if spectra.shape != (len(raw_wavelengths), height * width) or not np.all(np.isfinite(spectra)):
                raise ValueError(f"GHIFVD decoded spectra have invalid shape or values: {spectra.shape}.")
            cube = spectra.T.reshape((height, width, len(raw_wavelengths)), order='F')
            return cube, raw_wavelengths
        try:
            vars_info = sio.whosmat(file_path)
            var_names = [v[0] for v in vars_info]

            # Prioritized key search
            target_var = None
            img_keys = ['rad', 'hyperimg', 'img', 'cube', 'DataCubeC', 'RefCube', 'truth',
                        'data', 'dataset', 'hyperfile', 'ref', 'hsi', 'HSI', 'spectral_data']
            for k in img_keys:
                if k in var_names:
                    target_var = k
                    break

            if target_var is None:
                valid_vars = [v for v in vars_info if len(v[1]) >= 2 and not v[0].startswith('__')]
                if valid_vars:
                    valid_vars.sort(key=lambda v: np.prod(v[1]), reverse=True)
                    target_var = valid_vars[0][0]
                elif 'cspaces' in var_names:
                    target_var = 'cspaces'

            # Check for wavelengths variable
            wv_keys = [v for v in var_names if v.lower() in ['wavelengths', 'wavelength', 'wlns', 'bands', 'wl']]
            load_vars = [target_var] + wv_keys if target_var else None

            mat = sio.loadmat(file_path, variable_names=load_vars)

            for wk in wv_keys:
                if wk in mat and isinstance(mat[wk], np.ndarray):
                    raw_wavelengths = mat[wk].flatten().tolist()
                    break

            data_arr = None
            if target_var == 'cspaces' and 'cspaces' in mat:
                cs = mat['cspaces']
                for field in ['XYZ', 'RGB', 'gRGB', 'xyY', 'LAB']:
                    if field in cs.dtype.names:
                        data_arr = cs[field][0, 0]
                        break
            elif target_var and target_var in mat:
                data_arr = mat[target_var]

            if data_arr is None:
                raise ValueError(f"No suitable image matrix found in MAT file: {file_path}")
            return data_arr, raw_wavelengths

        except Exception:
            # Fallback to h5py for MATLAB v7.3 files
            with h5py.File(file_path, 'r') as hf:
                if 'wavelengths' in hf:
                    raw_wavelengths = np.array(hf['wavelengths']).flatten().tolist()

                img_keys = ['img', 'cube', 'data', 'rad', 'RefCube', 'hyperfile', 'truth']
                data_arr = None
                for k in img_keys:
                    if k in hf and hasattr(hf[k], 'shape') and len(hf[k].shape) in [2, 3]:
                        data_arr = np.array(hf[k])
                        break
                if data_arr is None:
                    valid_k = [k for k in hf.keys() if hasattr(hf[k], 'shape') and len(hf[k].shape) in [2, 3]]
                    if valid_k:
                        valid_k.sort(key=lambda k: np.prod(hf[k].shape), reverse=True)
                        data_arr = np.array(hf[valid_k[0]])
                if data_arr is None:
                    raise ValueError(f"No suitable HDF5 matrix found in MAT file: {file_path}")
                return data_arr, raw_wavelengths

    def _load_h5_file(self, file_path):
        """Load .h5 file."""
        with h5py.File(file_path, 'r') as hf:
            keys = [k for k in hf.keys() if hasattr(hf[k], 'shape') and len(hf[k].shape) in [2, 3]]
            if not keys:
                raise ValueError(f"No 2D/3D datasets in H5 file: {file_path}")
            keys.sort(key=lambda k: np.prod(hf[k].shape), reverse=True)
            arr = np.array(hf[keys[0]])
            raw_wavelengths = None
            if 'wavelengths' in hf:
                raw_wavelengths = np.array(hf['wavelengths']).flatten().tolist()
            if self.dataset_type == 'beyondrgb':
                if 'MIS' not in hf:
                    raise ValueError("BeyondRGB HDF5 file has no documented 'MIS' CFA mosaic dataset.")
                mosaic = np.asarray(hf['MIS'])
                decoded_cube = self._unpack_beyondrgb_cfa(mosaic)
                if decoded_cube.shape[-1] != 16:
                    raise ValueError(
                        f"BeyondRGB CFA decoder returned {decoded_cube.shape[-1]} channels; expected 16."
                    )
                # Always unpack the complete CFA before optionally filtering bands.
                effective_count = self.beyondrgb_band_count
                arr = decoded_cube[..., :effective_count]
                self.beyondrgb_metadata = {
                    'original_cfa_shape': list(mosaic.shape),
                    'decoded_cube_shape': list(decoded_cube.shape),
                    'original_band_count': 16,
                    'effective_band_count': effective_count,
                    'retained_band_indices': list(range(effective_count)),
                    'removed_band_indices': list(range(effective_count, 16)),
                    'filter_operation': f'unpack full 4x4 CFA, then retain cube[..., :{effective_count}]',
                }
                raw_wavelengths = None
            return arr, raw_wavelengths

    @staticmethod
    def _unpack_beyondrgb_cfa(mosaic):
        """Unpack the documented 4×4 CFA into 16 phase-sampled channels.

        Channel positions are taken from BeyondRGB WACV 2024 Figure S1, read
        row-major as [[9,11,13,15],[8,10,12,14],[1,3,5,7],[0,2,4,6]].
        No interpolation or registration is performed.
        """
        mosaic = np.asarray(mosaic)
        if mosaic.ndim != 2 or mosaic.shape[0] % 4 or mosaic.shape[1] % 4:
            raise ValueError(
                "BeyondRGB CFA mosaic must be 2D with dimensions divisible by four; "
                f"got {mosaic.shape}."
            )
        pattern = np.asarray(
            [[9, 11, 13, 15], [8, 10, 12, 14], [1, 3, 5, 7], [0, 2, 4, 6]],
            dtype=np.intp,
        )
        tiles = mosaic.reshape(mosaic.shape[0] // 4, 4, mosaic.shape[1] // 4, 4).transpose(0, 2, 1, 3)
        cube = np.empty((mosaic.shape[0] // 4, mosaic.shape[1] // 4, 16), dtype=mosaic.dtype)
        for row in range(4):
            for col in range(4):
                cube[..., pattern[row, col]] = tiles[..., row, col]
        return cube

    @staticmethod
    def _load_ladybird_scanline(file_path):
        """Read one native Pika-XC2 scan as (1, spatial, spectral), no stacking."""
        normalized = Path(file_path).resolve()
        roots = [p for p in (normalized, *normalized.parents) if p.name == 'Ladybird_Cobbitty_2017_Brassica_Dataset']
        if len(roots) != 1:
            raise ValueError("Ladybird scanline must be located under exactly one Ladybird dataset root.")
        dataset_root = roots[0]
        wavelengths_path = dataset_root / 'calibrations' / 'pika-xc2' / 'spectra.txt'
        if not wavelengths_path.is_file():
            raise FileNotFoundError(f"Ladybird Pika-XC2 wavelength calibration is missing: {wavelengths_path}")
        raw = cv2.imread(str(normalized), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2:
            raise ValueError(f"Ladybird Pika-XC2 input must be a single-channel PNG; got {getattr(raw, 'shape', None)}.")
        wavelengths = np.loadtxt(wavelengths_path, dtype=np.float64).reshape(-1)
        # The dataset's own example notebook states each image row is one
        # wavelength and columns are spatial samples.
        if wavelengths.size != raw.shape[0] or not np.all(np.isfinite(wavelengths)):
            raise ValueError(
                f"Ladybird wavelength vector has {wavelengths.size} entries for {raw.shape[0]} spectral rows."
            )
        if wavelengths.size > 1 and not np.all(np.diff(wavelengths) > 0):
            raise ValueError("Ladybird spectra.txt must be strictly increasing in spectral row order.")
        return raw.T[np.newaxis, ...], wavelengths.tolist()

    def _load_tiff_file(self, file_path):
        """Load .tiff/.tif hyperspectral file."""
        arr = tifffile.imread(file_path)
        return arr, None

    def _load_npz_file(self, file_path):
        """Load one HSI cube and optional wavelengths from a NumPy archive."""
        with np.load(file_path, allow_pickle=False) as archive:
            keys = list(archive.files)
            wave_key = next(
                (key for key in keys if key.lower() in {'wavelengths', 'wavelength', 'wavelength_nm', 'wavelengths_nm'}),
                None,
            )
            raw_wavelengths = None if wave_key is None else np.asarray(archive[wave_key]).reshape(-1).tolist()
            preferred = ('cube', 'hsi_cube', 'hsi', 'data')
            cube_key = next((key for key in preferred if key in archive.files), None)
            if cube_key is None:
                candidates = [key for key in keys if key != wave_key and archive[key].ndim == 3]
                if len(candidates) != 1:
                    raise ValueError(
                        f"NumPy archive must contain one identifiable 3D HSI cube; found {candidates}."
                    )
                cube_key = candidates[0]
            cube = np.asarray(archive[cube_key])
            if cube.ndim != 3:
                raise ValueError(f"NumPy archive cube '{cube_key}' must be three-dimensional; got {cube.shape}.")
            if raw_wavelengths is not None and len(raw_wavelengths) != cube.shape[-1]:
                raw_wavelengths = None
            return cube, raw_wavelengths

    def _load_image_stack(self, file_path):
        """Load dataset composed of single wavelength PNG/JPG slices in a folder."""
        folder = os.path.dirname(file_path)
        png_files = sorted(glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg')))
        if self.append_rgb and self.dataset_type == 'near_infrared':
            wavelength_files = []
            for path in png_files:
                match = re.search(r'(\d+)nm', os.path.basename(path), flags=re.IGNORECASE)
                if match is None:
                    wavelength_files = []
                    break
                wavelength_files.append((float(match.group(1)), path))
            if wavelength_files:
                wavelength_files.sort(key=lambda item: (item[0], item[1]))
                png_files = [path for _, path in wavelength_files]
        if len(png_files) <= 1:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            return img, None

        imgs = []
        raw_wavelengths = []
        for p in png_files:
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            imgs.append(im)
            m = re.search(r'(\d+)nm', os.path.basename(p))
            if m:
                raw_wavelengths.append(float(m.group(1)))

        stack = np.stack(imgs, axis=-1)
        if len(raw_wavelengths) != stack.shape[-1]:
            raw_wavelengths = None
        return stack, raw_wavelengths

    def load(self, file_path):
        """Load image and return (img, wavelengths)."""
        # Metadata belongs to one load call; a failed or subsequent call must
        # not expose descriptors from a previous cube.
        self.channel_metadata = None
        self.beyondrgb_metadata = None
        self.wavelength_source = 'unavailable'
        self.wavelength_verified = None
        self.spectral_axis = -1
        self.wavelength_metadata = {
            'wavelength_source': self.wavelength_source,
            'wavelength_verified': self.wavelength_verified,
            'wavelength_status': 'unknown',
            'spectral_axis': self.spectral_axis,
        }
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        header_wavelengths = None
        ladybird_scanline = False
        envi_source = False

        if self.dataset_type == 'hyperblood':
            img, raw_wav = ds_load.get_data(file_path)
            if header_wavelengths is None:
                header_wavelengths = raw_wav
            if self.append_rgb:
                img_rgb = ds_load.get_rgb(img, raw_wav)
                if self.rgb_value_mode == 'display':
                    img_rgb = np.asarray(img_rgb, dtype=np.float32) * np.float32(255.0)
                img = np.concatenate((img, img_rgb), axis=2)
        elif self.dataset_type == 'hotvis':
            img = get_image_loader(img_file=file_path, cellSize=4)
            if self.append_rgb:
                rgb_path = file_path.replace('/HSI-VIS/', '/HSI-VIS-FalseColor/').replace(".png", ".jpg")
                if os.path.exists(rgb_path):
                    rgb_cube = get_image_loader(rgb_path, cellSize=-1)
                    img_rgb = np.array(rgb_cube).astype(np.float32)
                    img = np.concatenate((img, img_rgb), axis=2)
        elif self.dataset_type == 'hotrednir':
            # Preserve the existing HOT mosaic decode: the final 4x4 slot is
            # omitted, yielding 15 returned channels. Do not remove another
            # band without validating the exact source release and frame.
            img = get_image_loader(img_file=file_path, cellSize=4)[:, :, :-1]
            if self.append_rgb:
                rgb_path = file_path.replace('/HSI-RedNIR/', '/HSI-RedNIR-FalseColor/').replace(".png", ".jpg")
                if os.path.exists(rgb_path):
                    rgb_cube = get_image_loader(rgb_path, cellSize=-1)
                    img_rgb = np.array(rgb_cube).astype(np.float32)
                    img = np.concatenate((img, img_rgb), axis=2)
        elif self.dataset_type == 'hotnir':
            img = get_image_loader(img_file=file_path, cellSize=5)
            if self.append_rgb:
                rgb_path = file_path.replace('/HSI-NIR/', '/HSI-NIR-FalseColor/').replace(".png", ".jpg")
                if os.path.exists(rgb_path):
                    rgb_cube = get_image_loader(rgb_path, cellSize=-1)
                    img_rgb = np.array(rgb_cube).astype(np.float32)
                    img = np.concatenate((img, img_rgb), axis=2)
        elif ext == '.mat':
            img, header_wavelengths = self._load_mat_file(file_path)
        elif ext == '.h5':
            img, header_wavelengths = self._load_h5_file(file_path)
        elif ext in ['.tiff', '.tif']:
            img, header_wavelengths = self._load_tiff_file(file_path)
        elif ext == '.npz':
            img, header_wavelengths = self._load_npz_file(file_path)
        elif ext == '.npy':
            img = np.load(file_path)
        elif ext == '.hsd':
            img = read_hsd(file_path)
        elif ext == '.hdr':
            img, header_wavelengths = self._load_envi_file(file_path)
            envi_source = True
        elif ext in ['.bin', '.raw', '.dat', '.bip', '.img']:
            base, _ = os.path.splitext(file_path)
            hdr_path = base + '.hdr'
            d = os.path.dirname(file_path)
            local_hdrs = glob.glob(os.path.join(d, '*.hdr'))
            root_hdrs = glob.glob(os.path.join(d, '..', '*.hdr')) + glob.glob(os.path.join(d, '..', '..', '*.hdr'))

            if self.dataset_type == 'hyperpri' or os.path.exists(hdr_path) or local_hdrs or root_hdrs:
                img, header_wavelengths = self._load_envi_file(file_path)
                envi_source = True
            elif self.dataset_type == 'clinical_applications':
                raw_data = np.fromfile(file_path, dtype=np.float32)
                img = raw_data.reshape((100, 1000, 1000))
                img = np.transpose(img, (1, 2, 0)) # -> 1000, 1000, 100
            else:
                raise ValueError(f"Binary file without ENVI header: {file_path}")
        elif ext in ['.png', '.jpg', '.jpeg']:
            if self.dataset_type == 'ladybird_cobbitty':
                img, header_wavelengths = self._load_ladybird_scanline(file_path)
                ladybird_scanline = True
            elif self.dataset_type in ['near_infrared', 'uwa_face']:
                img, header_wavelengths = self._load_image_stack(file_path)
            else:
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # HyperLeaf TIFFs store (204 spectral, 48 rows, 352 columns). Keep the
        # legacy append=False orientation untouched; append mode makes HWC.
        ghifvd_hwc = (
            self.dataset_type == 'ghifvd'
            and '/pca/' in file_path.replace('\\', '/').lower()
            and img.ndim == 3
            and header_wavelengths is not None
            and len(header_wavelengths) == img.shape[-1]
        )
        hyperleaf_transposed = (
            self.append_rgb and self.dataset_type == 'hyperleaf2024'
            and img.ndim == 3 and img.shape[0] == 204 and img.shape[1:] == (48, 352)
        )
        if ghifvd_hwc:
            # Local MATLAB decoder already returns [height,width,bands]. The
            # generic axis heuristic would mistake short H/W dimensions for C,H,W.
            pass
        elif hyperleaf_transposed:
            img = np.transpose(img, (1, 2, 0))
        elif ladybird_scanline:
            pass
        elif envi_source:
            # ENVI readers already return (lines, samples, bands); preserve
            # that authoritative spectral axis instead of guessing from sizes.
            pass
        else:
            img = self._ensure_hwc(img)
        source_dtype = img.dtype
        calibration_metadata = None
        calibration_fallback_reason = None
        if self.append_rgb and self.rgb_value_mode == 'display' and self.dataset_type == 'almond_pistachio_garlic':
            try:
                img, calibration_metadata = self._calibrate_split_reference_cube(img, file_path)
            except (ValueError, FileNotFoundError) as exc:
                # If calibration evidence is incomplete for this sample, retain
                # the source cube and let display mode normalize only RGB bands.
                calibration_fallback_reason = str(exc)

        if self.to_float32 and img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        # Resolve wavelengths
        hyperleaf_legacy_chw = (
            self.dataset_type == 'hyperleaf2024'
            and not self.append_rgb
            and img.ndim == 3
            and img.shape[0] == 204
            and img.shape[1:] == (48, 352)
        )
        self.spectral_axis = 0 if hyperleaf_legacy_chw else -1
        num_channels = img.shape[self.spectral_axis]
        self.bands = num_channels
        if (
            self.append_rgb and self.dataset_type == 'hyper_skin' and num_channels == 31
            and not (header_wavelengths is not None and len(header_wavelengths) == num_channels)
        ):
            path_lower = file_path.replace('\\', '/').lower()
            if '/vis/' in path_lower:
                self.wavelengths = np.linspace(400.0, 700.0, 31).tolist()
            elif '/nir/' in path_lower:
                self.wavelengths = np.linspace(700.0, 1000.0, 31).tolist()
            else:
                self.wavelengths = resolve_wavelengths(
                    self.dataset_type, num_channels, file_path, header_wavelengths
                )
        elif self.dataset_type == 'hyperleaf2024' and num_channels == 204:
            self.wavelengths = list(HYPERLEAF2024_WAVELENGTHS_NM)
        else:
            self.wavelengths = resolve_wavelengths(
                dataset_type=self.dataset_type,
                num_channels=num_channels,
                img_path=file_path,
                header_wavelengths=header_wavelengths
            )
        # Keep the returned wavelength vector for display and compatibility,
        # but do not use range-interpolated estimates for spectral band matching.
        path_lower = file_path.replace('\\', '/').lower()
        header_vector_valid = (
            header_wavelengths is not None
            and len(header_wavelengths) == num_channels
            and np.all(np.isfinite(np.asarray(header_wavelengths, dtype=np.float64)))
        )
        if self.dataset_type == 'beyondrgb' and num_channels in (12, 16) and len(self.wavelengths) == num_channels:
            self.wavelength_source = 'figure_s1_approximate_peak_estimates'
            self.wavelength_verified = False
        elif not self.wavelengths:
            self.wavelength_source = 'unavailable'
            self.wavelength_verified = None
        elif self.dataset_type == 'ghifvd' and '/pca/' in path_lower:
            self.wavelength_source = 'ghifvd_matlabHyper_hyperWavelengths_csv'
            self.wavelength_verified = True
        elif self.dataset_type == 'hyperleaf2024' and num_channels == 204:
            self.wavelength_source = 'official_hyperleaf_centers'
            self.wavelength_verified = True
        elif header_vector_valid:
            if self.dataset_type == 'hyperblood':
                self.wavelength_source = 'dataset_spectral_calibration_vector'
            elif (
                os.path.splitext(file_path)[1].lower() == '.hdr'
                or self.dataset_type in {
                    'libhsi', 'virginia_tech_tree', 'vnihdhiatlimafb',
                    'deephsnir', 'deephsvis', 'deephsviscor', 'deephs_debris',
                    'cabbage_eggplant', 'cloudpatch7', 'hyperpri', 'ugv_terrains',
                    'hsifoodingr64', 'hydrocarbon_spill', 'mineral_abundance',
                    'almond_pistachio_garlic',
                }
            ):
                self.wavelength_source = 'file_header'
            elif os.path.splitext(file_path)[1].lower() == '.mat':
                self.wavelength_source = 'mat_wavelength_variable'
            elif ladybird_scanline:
                self.wavelength_source = 'dataset_spectra_txt'
            elif self.dataset_type in {'near_infrared', 'uwa_face'}:
                self.wavelength_source = 'wavelength_tagged_filenames'
            else:
                self.wavelength_source = 'file_calibration_vector'
            self.wavelength_verified = True
        elif self.dataset_type == 'fiftyoutdoor' and num_channels == 33:
            self.wavelength_source = 'official_regular_sampling_10nm'
            self.wavelength_verified = True
        elif self.dataset_type == 'hs_sod' and num_channels == 81:
            self.wavelength_source = 'official_regular_sampling_5nm'
            self.wavelength_verified = True
        else:
            self.wavelength_source = 'linear_range_fallback'
            self.wavelength_verified = False
        self.wavelength_metadata = {
            'wavelength_source': self.wavelength_source,
            'wavelength_verified': self.wavelength_verified,
            'wavelength_status': (
                'verified' if self.wavelength_verified is True else
                ('estimated' if self.wavelength_verified is False else 'unknown')
            ),
            'spectral_axis': self.spectral_axis,
        }

        # Keep the pre-existing legacy RGB branches above unchanged. For the
        # new cube datasets, append only spectra-derived FalseRGB channels.
        if self.append_rgb and self.dataset_type not in {
            'hyperblood', 'hotvis', 'hotrednir', 'hotnir'
        }:
            if self.dataset_type == 'ghifvd' and '/pca/' not in file_path.replace('\\', '/').lower():
                raise ValueError(
                    "GHIFVD FalseRGB supports only compressed PCA MAT files decoded with the dataset's local official recipe; colorspace/PCA display products are not HSI cubes."
                )
            profile = None
            if self.rgb_value_mode == 'display':
                try:
                    profile = resolve_normalization_profile(
                        self.dataset_type,
                        file_path=file_path,
                        source_dtype=source_dtype,
                        channel_count=img.shape[-1],
                    )
                except ValueError:
                    profile = None
                if profile is not None:
                    base_cube, hsi_mapping_metadata = map_hsi_to_display(img, profile)
                    if calibration_metadata is not None:
                        hsi_mapping_metadata['calibration'] = calibration_metadata
                else:
                    # Unknown dataset scales affect only the appended RGB channels;
                    # those are normalized per selected channel inside build_false_rgb.
                    base_cube = img
                    hsi_mapping_metadata = {
                        'method': 'identity_for_hsi_channels',
                        'output_dtype': str(img.dtype),
                        'source_range': None,
                        'image_extrema_scaling': False,
                    }
                    if calibration_fallback_reason is not None:
                        hsi_mapping_metadata['calibration_fallback_reason'] = calibration_fallback_reason
            elif self.rgb_value_mode == 'copy_hsi':
                base_cube = img
                hsi_mapping_metadata = {
                    'method': 'identity', 'output_dtype': str(img.dtype),
                    'source_range': None, 'image_extrema_scaling': False,
                }
            else:
                raise ValueError("rgb_value_mode must be 'copy_hsi' or 'display'.")
            # Only measured per-file vectors and explicitly verified center
            # lists may select by wavelength. Estimated ranges use channel-order
            # quantiles in auto mode and cannot satisfy visible/non-visible mode.
            known_vector = self.wavelength_verified is True
            wavelength_source = self.wavelength_source
            allow_estimated_match = (
                self.allow_estimated_wavelengths
                and self.wavelength_verified is False
                and self.rgb_mode in {'visible', 'non_visible'}
            )
            source_wavelengths = self.wavelengths if (known_vector or allow_estimated_match) else None
            rgb_channel_indices = None
            if self.dataset_type == 'beyondrgb' and self.rgb_mode == 'auto':
                # This is an explicit band-index profile, not wavelength
                # matching; the approximate centers remain marked estimated.
                source_wavelengths = self.wavelengths
                rgb_channel_indices = [9, 6, 2]
            if self.dataset_type == 'hyper_skin' and num_channels == 31 and ('/vis/' in path_lower or '/nir/' in path_lower):
                known_vector = False
                source_wavelengths = self.wavelengths if allow_estimated_match else None
                wavelength_source = 'product_range_fallback'
                self.wavelength_source = wavelength_source
                self.wavelength_metadata['wavelength_source'] = wavelength_source
            ordered_hsi_datasets = {
                'cabbage_eggplant', 'cloudpatch7', 'cocoa_beans', 'clinical_applications',
                'deephs_debris', 'gans_grape', 'hascid', 'hfd100', 'hsi1', 'hsidermoscopy',
                'hsifoodingr64', 'hyper_skin', 'hyperleaf2024',
                'ugv_terrains', 'mobispectral', 'near_infrared', 'omhs', 'mineral_abundance',
                'tohs', 'tecnalia', 'tokyotech', 'uwa_face', 'almond_pistachio_garlic', 'hyperpri', 'ghifvd',
                'beyondrgb', 'ladybird_cobbitty',
            }
            wavelength_order_known = False
            if known_vector and len(self.wavelengths) == img.shape[-1]:
                wavelength_diffs = np.diff(np.asarray(self.wavelengths, dtype=np.float64))
                wavelength_order_known = bool(
                    wavelength_diffs.size == 0
                    or np.all(wavelength_diffs > 0)
                    or np.all(wavelength_diffs < 0)
                )
            rgb, false_rgb_metadata = build_false_rgb(
                base_cube,
                source_wavelengths,
                self.dataset_type,
                mode=self.rgb_mode,
                rgb_wavelengths=self.rgb_wavelengths,
                profile=profile,
                already_mapped=self.rgb_value_mode == 'display',
                diagnostic_cube=img,
                value_mode=self.rgb_value_mode,
                spectral_order_known=(
                    wavelength_order_known
                    or self.dataset_type in ordered_hsi_datasets
                    or (
                        self.dataset_type in WAVELENGTH_CONFIGS
                        and self.dataset_type != 'hydrocarbon_spill'
                    )
                ),
                wavelength_source=wavelength_source,
                wavelength_verified=self.wavelength_verified,
                allow_estimated_wavelengths=self.allow_estimated_wavelengths,
                channel_indices=rgb_channel_indices,
            )
            spectral_channel_count = img.shape[-1]
            source_hsi_shape = list(img.shape)
            source_spatial_shape = list(img.shape[:2])
            source_hsi_dtype = str(img.dtype)
            img = np.concatenate((base_cube, rgb), axis=-1)
            cfa_phase_offsets = [
                [3, 0], [2, 0], [3, 1], [2, 1], [3, 2], [2, 2], [3, 3], [2, 3],
                [1, 0], [0, 0], [1, 1], [0, 1], [1, 2], [0, 2], [1, 3], [0, 3],
            ]
            self.channel_metadata = {
                'spectral_channel_count': spectral_channel_count,
                'spectral_wavelengths_nm': list(self.wavelengths) if len(self.wavelengths) == spectral_channel_count else None,
                **self.wavelength_metadata,
                'source': {
                    'dataset_type': self.dataset_type,
                    'file_path': os.path.abspath(file_path),
                    'hsi_shape_hwc': source_hsi_shape,
                    'dtype': source_hsi_dtype,
                    'wavelength_source': false_rgb_metadata['wavelength_source'],
                    'wavelength_verified': self.wavelength_verified,
                    **({
                        'spatial_type': 'single_scanline',
                        'reconstruction': 'none',
                        'scanline_count': 1,
                        'spectral_direction': 'ascending wavelength in channel order',
                        'wavelength_source': 'Ladybird dataset calibrations/pika-xc2/spectra.txt',
                    } if ladybird_scanline else {}),
                    **({
                        'spatial_type': 'phase_sampled_cfa_cube',
                        'reconstruction': '4x4 CFA unpack only; no demosaicing or registration',
                        'cfa_channel_phase_offsets_row_col': cfa_phase_offsets,
                    } if self.dataset_type == 'beyondrgb' else {}),
                },
                'hsi_mapping': hsi_mapping_metadata,
                'appended_channel_descriptors': false_rgb_metadata['channel_descriptors'],
                'false_rgb': false_rgb_metadata,
                'channel_order': 'HSI spectral channels followed by R,G,B FalseRGB copies',
                'spatial_alignment': {
                    'method': ('CFA_phase_subsampled_without_demosaic' if self.dataset_type == 'beyondrgb' else ('same_native_scanline_samples' if ladybird_scanline else 'pixelwise_same_grid')),
                    'source_spatial_shape': source_spatial_shape,
                    'output_spatial_shape': list(img.shape[:2]),
                    'appended_channels_share_source_pixel_grid': self.dataset_type != 'beyondrgb',
                    **({'cfa_channel_phase_offsets_row_col': cfa_phase_offsets} if self.dataset_type == 'beyondrgb' else {}),
                },
            }
            if ladybird_scanline:
                self.channel_metadata.update({
                    'spatial_type': 'single_scanline',
                    'reconstruction': 'none',
                    'scanline_count': 1,
                    'spectral_direction': 'ascending wavelength in channel order',
                })
            if self.dataset_type == 'beyondrgb':
                self.channel_metadata['false_rgb']['spatial_sampling'] = 'selected CFA phases; no demosaicing or registration'

        if self.dataset_type == 'beyondrgb':
            cfa_phase_offsets = [
                [3, 0], [2, 0], [3, 1], [2, 1], [3, 2], [2, 2], [3, 3], [2, 3],
                [1, 0], [0, 0], [1, 1], [0, 1], [1, 2], [0, 2], [1, 3], [0, 3],
            ]
            filter_metadata = dict(self.beyondrgb_metadata or {})
            filter_metadata.update({
                'original_band_count': 16,
                'effective_band_count': self.beyondrgb_band_count,
                'retained_band_indices': list(range(self.beyondrgb_band_count)),
                'removed_band_indices': list(range(self.beyondrgb_band_count, 16)),
                'wavelength_peaks_nm': list(self.wavelengths),
                'wavelength_status': 'estimated',
                'false_rgb_source_indices': [9, 6, 2] if self.append_rgb else None,
            })
            if self.channel_metadata is None:
                # Keep the CFA/filter provenance available in no-append mode
                # as well; shape and phase metadata describe the actual returned cube.
                self.channel_metadata = {
                    'spectral_channel_count': int(img.shape[-1]),
                    'spectral_wavelengths_nm': list(self.wavelengths),
                    **self.wavelength_metadata,
                    'source': {
                        'dataset_type': self.dataset_type,
                        'file_path': os.path.abspath(file_path),
                        'hsi_shape_hwc': list(img.shape),
                        'dtype': str(img.dtype),
                        'wavelength_source': self.wavelength_source,
                        'wavelength_verified': self.wavelength_verified,
                        'spatial_type': 'phase_sampled_cfa_cube',
                        'reconstruction': f"unpack full 4x4 CFA; retain bands 0-{self.beyondrgb_band_count - 1}; no demosaicing or registration",
                        'cfa_channel_phase_offsets_row_col': cfa_phase_offsets,
                    },
                    'appended_channel_descriptors': [],
                    'false_rgb': None,
                    'channel_order': f'{self.beyondrgb_band_count} retained MSI channels in original CFA band order',
                    'spatial_alignment': {
                        'method': 'CFA_phase_subsampled_without_demosaic',
                        'source_spatial_shape': list(img.shape[:2]),
                        'output_spatial_shape': list(img.shape[:2]),
                        'appended_channels_share_source_pixel_grid': False,
                        'cfa_channel_phase_offsets_row_col': cfa_phase_offsets,
                    },
                }
            self.channel_metadata.update(filter_metadata)
            self.channel_metadata['source'].update({
                key: value for key, value in filter_metadata.items()
                if key in {'original_cfa_shape', 'decoded_cube_shape', 'original_band_count',
                           'effective_band_count', 'retained_band_indices', 'removed_band_indices',
                           'filter_operation'}
            })
            self.channel_metadata['source']['wavelength_source'] = self.wavelength_source
            self.channel_metadata['source']['wavelength_verified'] = False
            self.channel_metadata['source']['wavelength_status'] = 'estimated'

        return img, self.wavelengths

    def __call__(self, results: dict) -> dict:
        """MMCV / MMDetection compatible callable interface."""
        img_path = results['img_path']
        img, wavelengths = self.load(img_path)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['wavelengths'] = wavelengths
        results['wavelength_metadata'] = dict(self.wavelength_metadata)
        if self.channel_metadata is not None:
            results['channel_metadata'] = self.channel_metadata
        return results
