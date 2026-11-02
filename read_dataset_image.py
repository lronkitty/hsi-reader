import os
import sys
import argparse
import numpy as np

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from hyperspectral_image_reader_v2.hyperspectral_pipelines import LoadHyperspectralImage
except ImportError:
    try:
        from hyperspectral_pipelines import LoadHyperspectralImage
    except ImportError as e:
        print(f"Error: Could not import LoadHyperspectralImage. Detail: {e}")
        sys.exit(1)

DATASET_EXTENSIONS = {
    # .mat formats
    'harvard': '.mat', 'umld2015': '.mat', 'umns2002': '.mat', 'umns2004': '.mat',
    'umos': '.mat', 'umri2015': '.mat', 'umemm': '.mat', 'hyperblood': ['.mat', '.tif', '.npz'],
    'arad_1k_31': '.mat', 'arad_1k_16': '.mat', 'cave': '.mat', 'fiftyoutdoor': '.mat',
    'hsodbitv2': '.mat', 'cloudpatch7': ['.mat', '.hdr', '.bip'], 'cocoa_beans': '.mat', 'hfd100': '.mat',
    'hsidermoscopy': '.mat', 'hyper_skin': '.mat', 'mobispectral': '.mat',
    'omhs': '.mat', 'tecnalia': '.mat', 'tokyotech': '.mat', 'ghifvd': '.mat',

    # .h5 formats
    'icvl': '.h5', 'hs_sod': ['.h5', '.mat'], 'beyondrgb': '.h5',

    # .npy / .npz formats
    'hsidrive20': '.npy', 'aphid': '.npy', 'hascid': '.npy',
    'hyperdrive': '.npz', 'hyperdrivevnir': '.npz', 'hyperdriveswir': '.npz',

    # ENVI formats (requires .hdr + raw/bin/img/bip file, pass the .hdr or data file path)
    'libhsi': '.hdr', 'virginia_tech_tree': '.hdr', 'vnihdhiatlimafb': '.hdr',
    'cabbage_eggplant': '.hdr', 'hyperpri': '.hdr', 'ugv_terrains': '.hdr',
    'mineral_abundance': '.hdr', 'almond_pistachio_garlic': '.hdr',
    'hsifoodingr64': '.hdr', 'hydrocarbon_spill': '.hdr',

    # ENVI / Binary formats (.bin, .raw, .img, .bip, .dat)
    'deephsnir': '.bin', 'deephsvis': '.bin', 'deephsviscor': '.bin', 'deephs_debris': '.bin',
    'clinical_applications': '.bin',

    # Image formats (.png, .jpg, .tif, .tiff)
    'hotvis': '.png', 'hotnir': '.png', 'hotrednir': '.png', 'near_infrared': '.png', 'uwa_face': '.png',
    'hsiroad': '.tif', 'gans_grape': '.tif', 'hsi1': '.tiff', 'hyperleaf2024': '.tiff', 'tohs': '.tiff',

    # Custom format
    'hyperspectralcityv2': '.hsd',
}

def load_hypervision_matrix(
    file_path,
    dataset_name,
    return_wavelengths=True,
    append_rgb=False,
    rgb_mode='auto',
    rgb_wavelengths=None,
    rgb_value_mode='display',
    allow_estimated_wavelengths=False,
    return_metadata=False,
    beyondrgb_band_count=12,
):
    """
    Loads a hyperspectral image and processes it into the exact matrix shape and scale
    expected by HyperVision / CV models, alongside its resolved wavelengths list.

    Args:
        file_path (str): Path to the image file.
        dataset_name (str): Name of the dataset (e.g., 'harvard', 'tokyotech', 'hascid', etc.).
        return_wavelengths (bool): Whether to return wavelengths. Defaults to True.
        append_rgb (bool): Append three spectra-derived FalseRGB channels. Defaults to False.
        rgb_mode (str): 'auto', strict 'visible', or explicit 'non_visible'.
        rgb_value_mode (str): 'display' maps appended RGB to [0,255] using a fixed profile or per-channel image min-max; 'copy_hsi' preserves source samples.
        rgb_wavelengths (sequence): Required R,G,B target wavelengths in non_visible mode.
        return_metadata (bool): Also return channel descriptors and normalization metadata.

    Returns:
        matrix (np.ndarray): (H, W, C) output. With append_rgb=False this keeps the
            historical per-image HSI min-max to [0,255]. With append_rgb=True for
            copy_hsi mode, the loaded HSI samples and appended channels share the
            source scale. Display mode maps appended RGB to [0,255], using a
            verified fixed profile or per-channel image min-max when no fixed
            profile is available.
        wavelengths (list[float]): Spectral wavelengths only; appended RGB descriptors are separate metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image path not found: {file_path}")

    # Validate file extension
    _, ext = os.path.splitext(file_path)
    expected_ext = DATASET_EXTENSIONS.get(dataset_name.lower())
    if expected_ext:
        if isinstance(expected_ext, (list, tuple)):
            if ext.lower() not in [e.lower() for e in expected_ext]:
                print(f"Warning: Expected file extension in {expected_ext} for dataset '{dataset_name}', but got '{ext}'.")
        elif ext.lower() != expected_ext.lower():
            print(f"Warning: Expected file extension '{expected_ext}' for dataset '{dataset_name}', but got '{ext}'.")

    # Initialize the dataset loader pipeline
    loader = LoadHyperspectralImage(
        dataset_type=dataset_name,
        to_float32=True,
        append_rgb=append_rgb,
        rgb_mode=rgb_mode,
        rgb_wavelengths=rgb_wavelengths,
        rgb_value_mode=rgb_value_mode,
        allow_estimated_wavelengths=allow_estimated_wavelengths,
        beyondrgb_band_count=beyondrgb_band_count,
    )
    img, wavelengths = loader.load(file_path)

    # append_rgb=False retains the historical per-sample min-max path exactly.
    # New append mode already fixed-maps every HSI band first and derives RGB
    # from that mapped HSI; do not apply any second scale to the concatenation.
    spectral_channels = (
        int(loader.channel_metadata['spectral_channel_count'])
        if loader.channel_metadata is not None
        else (len(wavelengths) if wavelengths is not None else img.shape[-1])
    )
    if append_rgb and loader.channel_metadata is not None:
        if img.shape[-1] != spectral_channels + 3:
            raise ValueError(
                "False RGB append did not produce exactly three extra channels; "
                f"got {img.shape[-1]} channels for {spectral_channels} wavelengths."
            )
        processed_matrix = img
    elif append_rgb:
        # Preserve the legacy dataset append behavior, including its existing
        # HSI-only scaling treatment.
        hsi = img[..., :spectral_channels]
        false_rgb = img[..., spectral_channels:]
    else:
        hsi = img

    if not (append_rgb and loader.channel_metadata is not None):
        hsi_min = hsi.min()
        hsi_max = hsi.max()
        if hsi_max > hsi_min:
            processed_hsi = 255.0 * (hsi - hsi_min) / (hsi_max - hsi_min)
        else:
            processed_hsi = np.zeros_like(hsi)

        if append_rgb:
            processed_matrix = np.concatenate((processed_hsi, false_rgb.astype(np.float32, copy=False)), axis=-1)
        else:
            processed_matrix = processed_hsi

    metadata = loader.channel_metadata if loader.channel_metadata is not None else loader.wavelength_metadata

    if return_metadata:
        if return_wavelengths:
            return processed_matrix, wavelengths, metadata
        return processed_matrix, metadata
    if return_wavelengths:
        return processed_matrix, wavelengths
    return processed_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read HSI dataset image and output the matrix and wavelengths.")
    parser.add_argument('--path', type=str, required=True, help="Path to the HSI image file.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g. harvard, tokyotech, hascid, etc.).")
    parser.add_argument('--output', type=str, default=None, help="Optional path to save the output matrix as a .npy file.")
    parser.add_argument('--append-rgb', action='store_true', help="Append three spectra-derived FalseRGB channels.")
    parser.add_argument('--rgb-mode', choices=['auto', 'visible', 'non_visible'], default='auto', help="FalseRGB spectral selection mode.")
    parser.add_argument('--rgb-value-mode', choices=['copy_hsi', 'display'], default='display', help="Map appended RGB to [0,255]; copy_hsi preserves source values.")
    parser.add_argument('--rgb-wavelengths', type=float, nargs=3, metavar=('R_NM', 'G_NM', 'B_NM'), help="Explicit R,G,B target wavelengths; required for non_visible mode.")
    parser.add_argument('--allow-estimated-wavelengths', action='store_true', help="Allow visible/non-visible wavelength matching against a range-interpolated wavelength estimate.")
    parser.add_argument('--beyondrgb-band-count', type=int, choices=[12, 16], default=12, help="BeyondRGB effective MSI channels (12 by default; 16 keeps all CFA channels).")

    args = parser.parse_args()

    try:
        result = load_hypervision_matrix(
            args.path,
            args.dataset,
            return_wavelengths=True,
            append_rgb=args.append_rgb,
            rgb_mode=args.rgb_mode,
            rgb_wavelengths=args.rgb_wavelengths,
            rgb_value_mode=args.rgb_value_mode,
            allow_estimated_wavelengths=args.allow_estimated_wavelengths,
            return_metadata=args.append_rgb,
            beyondrgb_band_count=args.beyondrgb_band_count,
        )
        if args.append_rgb:
            matrix, wavelengths, metadata = result
        else:
            matrix, wavelengths = result
        print("\nSuccessfully loaded and processed HSI image.")
        print(f"Matrix shape (H, W, C): {matrix.shape}")
        print(f"Value range: [{matrix.min():.2f}, {matrix.max():.2f}]")
        print(f"Data type: {matrix.dtype}")
        print(f"Total resolved wavelengths: {len(wavelengths)}")
        if args.append_rgb:
            print(f"Appended channels: {metadata['appended_channel_descriptors']}")
            print(f"FalseRGB normalization: {metadata['false_rgb']['normalization']}")
        if len(wavelengths) > 0:
            print(f"Wavelengths preview: {wavelengths[:3]} ... {wavelengths[-3:]}")
        else:
            print("Wavelengths: None")

        if args.output:
            np.save(args.output, matrix)
            print(f"Saved processed matrix to: {args.output}")

    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)
