"""
Wavelength definitions and dynamic dispatch helper for Hyperspectral Datasets.
Supports both legacy HyperVision datasets and newly added datasets in hsi_datasets/datasets_new.
"""

import numpy as np

# Official HyperLeaf2024 image channel centers (Kaggle competition dataset
# metadata). Kept separate from the legacy fallback configuration so callers
# with append_rgb=False retain their historical returned wavelength values.
HYPERLEAF2024_WAVELENGTHS_NM = [
    397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08,
    431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87,
    466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80,
    501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89,
    536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
    572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51,
    607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04,
    643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73,
    678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56,
    714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
    750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67,
    786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95,
    822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37,
    859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95,
    896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
    932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55,
    969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58,
]

WAVELENGTH_CONFIGS = {
    # --- Legacy HyperVision datasets ---
    'harvard': [420.0 + 10.0 * i for i in range(31)],  # 420-720nm, 31 bands
    'umld2015': [400.0 + 10.0 * i for i in range(33)], # 400-720nm, 33 bands
    'umns2002': [410.0 + 10.0 * i for i in range(31)], # 410-710nm, 31 bands
    'umns2004': [400.0 + 10.0 * i for i in range(33)], # 400-720nm, 33 bands
    'umos': [400.0 + 10.0 * i for i in range(33)],     # 400-720nm, 33 bands
    'umri2015': [400.0 + 10.0 * i for i in range(33)], # 400-720nm, 33 bands
    'umemm': [400.0 + 10.0 * i for i in range(33)],    # 400-720nm, 33 bands
    'hyperblood': [
        401.5636, 406.528, 411.4977, 416.4725, 421.4525, 426.4379, 431.4284, 436.4241,
        441.4251, 446.4313, 451.4427, 456.4594, 461.4813, 466.5084, 471.5408, 476.5783,
        481.6211, 486.6691, 491.7224, 496.7808, 501.8445, 506.9134, 511.9876, 517.067,
        522.1515, 527.2413, 532.3364, 537.4367, 542.5422, 547.6529, 552.7689, 557.8901,
        563.0164, 568.1481, 573.2849, 578.427, 583.5743, 588.7269, 593.8846, 599.0476,
        604.2158, 609.3892, 614.5679, 635.3348, 640.5396, 645.7496, 650.9649, 656.1853,
        661.4111, 666.642, 671.8782, 677.1195, 682.3661, 687.6179, 692.8751, 698.1374,
        703.4049, 708.6775, 713.9553, 719.238
    ],
    'hsidrive20': np.linspace(600.0, 975.0, 25).tolist(), # 25 bands, sensor range 600-975nm
    'hotrednir': np.linspace(600.0, 850.0, 15).tolist(),  # Estimated centers for the 15 channels returned by the legacy 4x4 mosaic decoder
    'arad_1k_31': [400.0 + 10.0 * i for i in range(31)], # 400-700nm, 31 bands
    'arad_1k_16': np.linspace(400.0, 1000.0, 16).tolist(), # 16-band product, nominal 400-1000nm range
    'cave': [400.0 + 10.0 * i for i in range(31)],       # 400-700nm, 31 bands
    'icvl': [400.0 + 10.0 * i for i in range(31)],       # 400-700nm, 31 bands
    'hs_sod': np.linspace(380.0, 780.0, 81).tolist(),     # 81 bands, 380-780nm
    'vnihdhiatlimafb': np.linspace(397.32, 1003.58, 204).tolist(), # fallback from checked Specim IQ header; per-file header wins
    'virginia_tech_tree': np.linspace(394.7, 1005.86, 420).tolist(), # fallback from checked VNIR header; per-file header wins
    'deephsnir': np.linspace(900.0, 1700.0, 224).tolist(),
    'deephsvis': np.linspace(400.0, 1000.0, 204).tolist(),
    'deephsviscor': np.linspace(400.0, 1000.0, 204).tolist(),
    'hsodbitv2': np.linspace(400.0, 1000.0, 200).tolist(),
    'hotvis': np.linspace(460.0, 600.0, 16).tolist(),    # 16 decoded bands; exact centers require release metadata
    'hotnir': np.linspace(665.0, 960.0, 25).tolist(),    # 25 decoded bands; exact centers require release metadata
    'libhsi': np.linspace(400.0, 1000.0, 204).tolist(),
    'fiftyoutdoor': [400.0 + 10.0 * i for i in range(33)], # 33 bands, 400-720nm
    'aphid': np.linspace(436.0, 965.0, 237).tolist(),       # Reported 237-band range; prefer dataset meta.yaml when present
    'hyperspectralcityv2': np.linspace(450.0, 950.0, 128).tolist(),
    'hykov2nir': np.linspace(600.0, 975.0, 25).tolist(),
    'hykov2vis': np.linspace(470.0, 630.0, 15).tolist(),
    'hsiroad': np.linspace(600.0, 960.0, 25).tolist(),
    'hyperdrivevnir': np.linspace(660.0, 900.0, 24).tolist(),
    'hyperdriveswir': np.linspace(1000.0, 1700.0, 25).tolist(),
    'hyperdrive': np.linspace(400.0, 1000.0, 25).tolist(),

    # --- New Datasets from datasets_new ---
    'cabbage_eggplant': np.linspace(399.86, 949.60, 277).tolist(), # 277 bands (VNIR)
    'cloudpatch7': np.linspace(388.86, 1004.46, 462).tolist(),     # 462 bands (VNIR)
    'cocoa_beans': np.linspace(400.0, 1000.0, 300).tolist(),       # 300 bands (or 301 bands)
    'clinical_applications': np.linspace(400.0, 1000.0, 100).tolist(), # 100 bands (Helicoid VNIR)
    'deephs_debris': np.linspace(400.0, 1000.0, 249).tolist(),     # Corning microHSI 249 bands
    'gans_grape': np.linspace(406.8, 995.8, 186).tolist(),         # HySpex VNIR-1800 186 bands
    'hascid': np.linspace(400.0, 1000.0, 204).tolist(),            # Specim IQ 204 bands
    'hfd100': {
        31: [400.0 + 10.0 * i for i in range(31)],                 # 31 bands (400-700nm)
        256: np.linspace(376.8, 1075.8, 256).tolist(),             # 256 bands (Full VNIR)
    },
    'hsi1': np.linspace(400.0, 1000.0, 96).tolist(),               # 96 bands (400-1000nm)
    'hsidermoscopy': np.linspace(400.0, 1000.0, 16).tolist(),       # 16 bands MSI/HSI dermoscopy
    'hsifoodingr64': np.linspace(397.32, 1003.58, 204).tolist(),    # Specim IQ 204 bands
    # Checked HSC-2 headers list channel 0 at 1000 nm, then channels 1-19
    # from 400 to 968.4 nm. Preserve this sensor order; never sort the cube.
    'hydrocarbon_spill': [1000.0, 400.0, 431.6, 463.2, 494.7, 526.3, 557.9, 589.5, 621.1, 652.6, 684.2, 715.8, 747.4, 778.9, 810.5, 842.1, 873.7, 905.3, 936.8, 968.4], # Senop HSC-2 header order
    'hyper_skin': {
        4: [450.0, 520.0, 600.0, 850.0],                          # MSI 4 bands
        31: np.linspace(400.0, 1000.0, 31).tolist(),              # NIR 31 bands
        448: np.linspace(400.0, 1000.0, 448).tolist(),            # Full HSI 448 bands
    },
    'hyperleaf2024': HYPERLEAF2024_WAVELENGTHS_NM,                 # Official channel centers
    'hyperpri': np.linspace(402.0, 998.0, 299).tolist(),           # Hinalea 299 bands
    'ugv_terrains': np.linspace(397.32, 1003.58, 204).tolist(),    # Specim IQ 204 bands
    'mobispectral': np.linspace(400.0, 1000.0, 204).tolist(),      # Specim IQ reconstructed 204 bands
    'near_infrared': [
        1293, 1302, 1312, 1322, 1332, 1341, 1351, 1361, 1370, 1380, 1390, 1400, 1409, 1419,
        1429, 1438, 1448, 1458, 1467, 1477, 1487, 1497, 1506, 1516, 1526, 1535, 1545, 1555,
        1564, 1574, 1584, 1594, 1603, 1613, 1623, 1632, 1642, 1652, 1662, 1671, 1681, 1691,
        1700, 1710, 1720, 1729, 1739, 1749, 1759, 1768, 1778, 1788, 1797, 1807, 1817, 1827,
        1836, 1846, 1856, 1865, 1875, 1885, 1894, 1904, 1914, 1924, 1933, 1943, 1953, 1962,
        1972, 1982, 1991, 2001, 2011, 2021, 2030, 2040, 2050, 2059, 2069, 2079, 2089, 2098,
        2108, 2118, 2127, 2137, 2147, 2156, 2166, 2176, 2186, 2195, 2205, 2215
    ],
    'omhs': [420.0 + 10.0 * i for i in range(29)],                 # 420-700nm, 29 bands
    'mineral_abundance': np.linspace(960.0, 2500.042, 288).tolist(), # SWIR HySpex 288 bands
    'tohs': np.linspace(350.0, 1002.0, 164).tolist(),              # 164 bands
    'tecnalia': [415.05 + i * (1008.10 - 415.05) / 75.0 for i in range(76)], # 76 bands
    'tokyotech': [400.0 + 10.0 * i for i in range(59)],            # 59 bands (400-980nm)
    'uwa_face': [400.0 + 10.0 * i for i in range(33)],             # 33 bands (400-720nm)
    'almond_pistachio_garlic': np.linspace(400.67, 997.99, 224).tolist(), # Headwall Photonics 224 bands
}

def resolve_wavelengths(dataset_type, num_channels=None, img_path=None, header_wavelengths=None):
    """
    Dynamically resolves the wavelengths list for a given dataset, handling multi-wavelength
    cases based on:
    1. Direct header wavelengths (if provided, e.g. from ENVI .hdr)
    2. Number of spectral channels (e.g. Hyper-Skin 4 vs 31 vs 448 bands, HFD100 31 vs 256 bands)
    3. File path subdirectories/keywords (e.g. 'MSI' vs 'NIR', 'VNIR' vs 'SWIR')
    4. Dataset-specific fallback dictionary.
    """
    # 1. If header already contains exact wavelengths
    if header_wavelengths is not None and len(header_wavelengths) > 0:
        try:
            parsed_header = [float(w) for w in header_wavelengths]
            if (
                (num_channels is None or len(parsed_header) == num_channels)
                and np.all(np.isfinite(np.asarray(parsed_header, dtype=np.float64)))
            ):
                return parsed_header
        except Exception:
            pass

    key = dataset_type.lower()
    cfg = WAVELENGTH_CONFIGS.get(key)

    if cfg is None:
        # Fallback linear interpolation between 400 and 1000nm if num_channels is known
        if num_channels is not None and num_channels > 1:
            return np.linspace(400.0, 1000.0, num_channels).tolist()
        elif num_channels == 1:
            return [550.0]
        return []

    # If cfg is a multi-wavelength dictionary
    if isinstance(cfg, dict):
        # 1. Check channel count exact match
        if num_channels is not None and num_channels in cfg:
            return cfg[num_channels]

        # 2. Check path keywords
        if img_path is not None:
            path_lower = img_path.lower()
            if 'msi' in path_lower and 4 in cfg:
                return cfg[4]
            if 'nir' in path_lower and 31 in cfg:
                return cfg[31]
            if 'test_spec' in path_lower or 'train_spec' in path_lower:
                if 204 in cfg: return cfg[204]
                if 448 in cfg: return cfg[448]

        # 3. Default to closest band match or largest
        if num_channels is not None:
            closest_bands = min(cfg.keys(), key=lambda b: abs(b - num_channels))
            return cfg[closest_bands]
        # Return first item
        return next(iter(cfg.values()))

    # If cfg is a list
    if isinstance(cfg, list):
        if num_channels is not None and len(cfg) != num_channels and num_channels > 0:
            # If band count differs, interpolate
            return np.linspace(cfg[0], cfg[-1], num_channels).tolist()
        return cfg

    return []
