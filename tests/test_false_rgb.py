import tempfile
import unittest
import gc
import os
from pathlib import Path
import re

import h5py
import numpy as np
from scipy.io import savemat

from hyperspectral_image_reader_v2.false_rgb import build_false_rgb, resolve_normalization_profile
from hyperspectral_image_reader_v2.hyperspectral_pipelines import LoadHyperspectralImage
from hyperspectral_image_reader_v2.read_dataset_image import DATASET_EXTENSIONS, load_hypervision_matrix
from hyperspectral_image_reader_v2.wavelength_configs import WAVELENGTH_CONFIGS


DATASET_ROOT = Path(os.environ.get("HSI_DATASET_ROOT", "/nonexistent/hsi_dataset"))
ARCHIVE_TEST_ROOT = Path(os.environ.get("HYPERVISION_TEST_DATA_ROOT", "/nonexistent/hypervision_test_data"))


class FalseRgbTests(unittest.TestCase):
    def test_real_supported_new_dataset_samples_through_full_api(self):
        root = DATASET_ROOT
        samples = {
            "cabbage_eggplant": "Cabbage_Eggplant_combined_Hyperspectral_dataset/Cabbage_Eggplant_Crops/CE_Reflectance_Data.hdr",
            "cloudpatch7": "CloudPatch-7_Hyperspectral_Dataset/DATAPORT/patches_from_parentimages/Scan_07-20-2022_1029_AZ315_EL25_L_D/Scan_07-20-2022_1029_AZ315_EL25_L_D_c04_x0333y2099w50h50.bip",
            "cocoa_beans": "Cocoa_beans_spectral_image_with_three_fermentation_levels/data_for_classification.mat",
            "clinical_applications": "Dataset_for_Hyperspectral_Clinical_Applications/PD1C1/Image.bin",
            "deephs_debris": "DeepHS_Debris/CORNING_HSI/concrete/concrete_106_front.hdr",
            "gans_grape": "GANs/dataset/data/h1_0301_19_6.tif",
            "ghifvd": "GHIFVD/pca/internals_mushroom_SegmentedCroppedCompressed.mat",
            "hascid": "HASCID-Dataset/gt_files/gtRef_1402.npy",
            "hfd100": "HFD100/MatScenes60/Train/P001/2019-04-17-1618_I25.5.mat",
            "hsi1": "HSI-1/VOC2007/JPEGImages/pad_rgb_0050.tiff",
            "hsidermoscopy": "HSIDermoscopy/HSI_datacube_after_correction/DNCube/154.mat",
            "hsifoodingr64": "HSIFoodIngr-64/HSIFoodIngr-64_data_6/REFLECTANCE_362.hdr",
            "hydrocarbon_spill": "Hydrocarbon_Spill_Hyperspectral_Dataset/Fotos17042024/Fotos17042024/a0029/HSI_snapshot__20241704124719.dat",
            "hyper_skin": "Hyper-Skin/Hyper-Skin(MSI, NIR)/test/NIR/p002_neutral_front.mat",
            "hyperleaf2024": "HyperLeaf2024/images/00000.tiff",
            "hyperpri": "HyperPRI/box_material/lexan/20240306_lexan_dry_ref.dat",
            "ugv_terrains": "Hyperspectral_image_dataset_of_unstructured_terrains_for_UGV_perception/HSI_Raw_Dataset/2019-11-18_002/results/REFLECTANCE_2019-11-18_002.hdr",
            "mobispectral": "MobiSpectral/dataset_tomato/reconstruction/Train_Spec/292.mat",
            "near_infrared": "Near_Infrared_Hyperspectral_Image_Dataset/data/coffee/coffee_1293nm.png",
            "omhs": "OMHS-The_Objects_Mosaic_Hyperspectral_Database/SHSGAN_database/LocX_0_LocY_0_RefCube_1.mat",
            "mineral_abundance": "Proximal_Multi-Scenario_SWIR_Hyperspectral_Mineral_Abundance_Dataset/4-In_situ_outcrop(part2)/1-In_situ_outcrop_HSI/OC0007R2.hdr",
            "tohs": "TOHS_Dataset/acornstatue0_export/acornstatue_000_002_snapshot_RAW.tiff",
            "tecnalia": "Tecnalia_Hyperspectral_Dataset/data/061025_Steel_Cu_Al_Steel_Brass_25ms_20fps_288_720_01.mat",
            "tokyotech": "TokyoTech_59-band_Visible-NIR_Hyperspectral_Image_Dataset/Butterfly2.mat",
            "uwa_face": "UWA_Hyperspectral_Face_Database/UWA HSFD V1.1/HyperSpec_Face_Session1/0001/F_01.png",
            "almond_pistachio_garlic": "hyperspectral_image_datasets-almond_pistachio_and_garlic_stems/Almond/Train/data.hdr",
        }
        if not root.is_dir():
            self.skipTest("datasets_new is not mounted")
        for dataset, relative_path in samples.items():
            path = root / relative_path
            with self.subTest(dataset=dataset, sample=relative_path):
                self.assertTrue(path.is_file(), f"Missing audit sample: {path}")
                matrix, wavelengths, metadata = load_hypervision_matrix(
                    str(path), dataset, append_rgb=True, rgb_value_mode="copy_hsi", return_metadata=True
                )
                self.assertEqual(matrix.shape[-1], len(wavelengths) + 3)
                self.assertEqual(matrix.dtype, np.float32)
                self.assertEqual(metadata["false_rgb"]["rgb_value_mode"], "copy_hsi")
                self.assertEqual(metadata["false_rgb"]["source"], "derived_from_hsi")
                self.assertEqual(metadata["false_rgb"]["channel_order"], "RGB")
                self.assertEqual(metadata["false_rgb"]["spatial_alignment"], "same_source_grid")
                self.assertEqual(metadata["source"]["dataset_type"], dataset)
                self.assertEqual(metadata["source"]["file_path"], str(path.absolute()))
                self.assertEqual(metadata["source"]["dtype"], "float32")
                self.assertEqual(metadata["source"]["hsi_shape_hwc"], list(matrix.shape[:2]) + [len(wavelengths)])
                self.assertEqual(
                    metadata["channel_order"],
                    "HSI spectral channels followed by R,G,B FalseRGB copies",
                )
                alignment = metadata["spatial_alignment"]
                self.assertEqual(alignment["method"], "pixelwise_same_grid")
                self.assertEqual(alignment["source_spatial_shape"], list(matrix.shape[:2]))
                self.assertEqual(alignment["output_spatial_shape"], list(matrix.shape[:2]))
                self.assertTrue(alignment["appended_channels_share_source_pixel_grid"])
                if dataset == "ghifvd":
                    self.assertEqual(metadata["false_rgb"]["wavelength_source"], "ghifvd_matlabHyper_hyperWavelengths_csv")
                indices = metadata["false_rgb"]["source_channel_indices"]
                self.assertTrue(all(index is not None for index in indices))
                # Compare every pixel without allocating one full-cube boolean array.
                for row_start in range(0, matrix.shape[0], 8):
                    row_end = min(row_start + 8, matrix.shape[0])
                    self.assertTrue(np.isfinite(matrix[row_start:row_end]).all())
                    np.testing.assert_array_equal(
                        matrix[row_start:row_end, :, -3:],
                        matrix[row_start:row_end, :, indices],
                    )
                del matrix, wavelengths, metadata
                gc.collect()

    def test_real_unsupported_new_formats_fail_with_cube_specific_reason(self):
        root = DATASET_ROOT
        if not root.is_dir():
            self.skipTest("datasets_new is not mounted")
        samples = {
            "ghifvd": ("GHIFVD/cspaces/internals_eggplant_SegmentedCroppedCompressed_cspaces.mat", "compressed PCA MAT"),
        }
        for dataset, (relative_path, message) in samples.items():
            path = root / relative_path
            with self.subTest(dataset=dataset, sample=relative_path):
                self.assertTrue(path.is_file(), f"Missing unsupported-format sample: {path}")
                with self.assertRaisesRegex((ValueError, ImportError, NotImplementedError), message):
                    load_hypervision_matrix(str(path), dataset, append_rgb=True)

    def test_hyperspectral_fruit_is_not_registered_as_a_supported_dataset(self):
        self.assertNotIn("hyperspectral_fruit", DATASET_EXTENSIONS)
        self.assertNotIn("hyperspectral_fruit", WAVELENGTH_CONFIGS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fruit.lo"
            path.touch()
            with self.assertRaisesRegex(ValueError, r"Unsupported file format: \.lo"):
                LoadHyperspectralImage("hyperspectral_fruit", append_rgb=True).load(str(path))

    def test_beyondrgb_cfa_unpack_is_reversible_and_channel_ordered(self):
        pattern = np.array([[9, 11, 13, 15], [8, 10, 12, 14], [1, 3, 5, 7], [0, 2, 4, 6]])
        cube = np.broadcast_to(np.arange(16, dtype=np.uint16), (2, 3, 16)).copy()
        mosaic = np.empty((8, 12), dtype=np.uint16)
        for row in range(4):
            for col in range(4):
                mosaic[row::4, col::4] = cube[..., pattern[row, col]]
        actual = LoadHyperspectralImage._unpack_beyondrgb_cfa(mosaic)
        np.testing.assert_array_equal(actual, cube)
        rebuilt = np.empty_like(mosaic)
        for row in range(4):
            for col in range(4):
                rebuilt[row::4, col::4] = actual[..., pattern[row, col]]
        np.testing.assert_array_equal(rebuilt, mosaic)
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            LoadHyperspectralImage._unpack_beyondrgb_cfa(np.zeros((5, 8)))

    def test_beyondrgb_explicit_false_rgb_indices_keep_estimates_unverified(self):
        wavelengths = [400, 420, 450, 470, 510, 530, 550, 590, 640, 660, 680, 710]
        cube = np.arange(2 * 3 * 12, dtype=np.float32).reshape(2, 3, 12)
        rgb, metadata = build_false_rgb(
            cube,
            wavelengths,
            "beyondrgb",
            mode="auto",
            channel_indices=[9, 6, 2],
            wavelength_source="figure_s1_approximate_peak_estimates",
            wavelength_verified=False,
            value_mode="copy_hsi",
        )
        np.testing.assert_array_equal(rgb, cube[..., [9, 6, 2]])
        self.assertEqual(metadata["selection_method"], "explicit_channel_indices")
        self.assertEqual(metadata["source_channel_indices"], [9, 6, 2])
        self.assertEqual(metadata["source_wavelengths_nm"], [660.0, 550.0, 450.0])
        self.assertFalse(metadata["wavelength_verified"])

    def test_real_beyondrgb_nt_wt_support_12_or_16_bands_and_aligned_false_rgb(self):
        self.assertEqual(LoadHyperspectralImage("beyondrgb").beyondrgb_band_count, 12)
        root = DATASET_ROOT / "BeyondRGB/clb/BLUE_blue/10_light7"
        paths = [root / "NT/MIS.h5", root / "WT/MIS.h5"]
        if not all(path.is_file() for path in paths):
            self.skipTest("real BeyondRGB NT/WT CFA samples are not mounted")
        pattern = np.array([[9, 11, 13, 15], [8, 10, 12, 14], [1, 3, 5, 7], [0, 2, 4, 6]])
        wavelength_profile = [400, 420, 450, 470, 510, 530, 550, 590, 640, 660, 680, 710, 680, 520, 630, 530]
        for path in paths:
            with self.subTest(sample=path.parent.name):
                with h5py.File(path, "r") as handle:
                    mosaic = np.asarray(handle["MIS"])
                self.assertEqual(mosaic.shape, (1936, 2584))
                decoded16 = LoadHyperspectralImage._unpack_beyondrgb_cfa(mosaic)
                self.assertEqual(decoded16.shape, (484, 646, 16))
                # Reinterleave all 16 phase channels before any band filtering.
                reinterleaved = np.empty_like(mosaic)
                for row in range(4):
                    for col in range(4):
                        reinterleaved[row::4, col::4] = decoded16[..., pattern[row, col]]
                np.testing.assert_array_equal(reinterleaved, mosaic)
                for band_count in (12, 16):
                    expected_cube = decoded16[..., :band_count].astype(np.float32)
                    expected_wavelengths = wavelength_profile[:band_count]
                    expected_removed = list(range(band_count, 16))
                    for append_rgb in (False, True):
                        loader = LoadHyperspectralImage(
                            "beyondrgb", append_rgb=append_rgb, rgb_value_mode="copy_hsi",
                            to_float32=True, beyondrgb_band_count=band_count,
                        )
                        output, wavelengths = loader.load(str(path))
                        self.assertEqual(output.shape, (484, 646, band_count + (3 if append_rgb else 0)))
                        self.assertEqual(output.dtype, np.float32)
                        self.assertTrue(np.isfinite(output).all())
                        np.testing.assert_array_equal(output[..., :band_count], expected_cube)
                        self.assertEqual(wavelengths, expected_wavelengths)
                        self.assertFalse(loader.wavelength_verified)
                        self.assertEqual(loader.wavelength_metadata["wavelength_status"], "estimated")
                        metadata = loader.channel_metadata
                        self.assertEqual(metadata["original_band_count"], 16)
                        self.assertEqual(metadata["effective_band_count"], band_count)
                        self.assertEqual(metadata["removed_band_indices"], expected_removed)
                        self.assertEqual(metadata["wavelength_peaks_nm"], expected_wavelengths)
                        self.assertEqual(metadata["wavelength_status"], "estimated")
                        self.assertEqual(metadata["source"]["decoded_cube_shape"], [484, 646, 16])
                        self.assertEqual(metadata["spatial_alignment"]["method"], "CFA_phase_subsampled_without_demosaic")
                        if append_rgb:
                            np.testing.assert_array_equal(output[..., -3:], output[..., [9, 6, 2]])
                            self.assertEqual(metadata["false_rgb_source_indices"], [9, 6, 2])
                            self.assertEqual(metadata["false_rgb"]["selection_method"], "explicit_channel_indices")
                            self.assertEqual(metadata["false_rgb"]["source_wavelengths_nm"], [660.0, 550.0, 450.0])
                            self.assertEqual(metadata["false_rgb"]["wavelength_status"], "estimated")
                    wrapper, wrapper_wavelengths, wrapper_metadata = load_hypervision_matrix(
                        str(path), "beyondrgb", append_rgb=False, return_metadata=True,
                        beyondrgb_band_count=band_count,
                    )
                    self.assertEqual(wrapper.shape, (484, 646, band_count))
                    self.assertEqual(wrapper.dtype, np.float32)
                    self.assertEqual(wrapper_wavelengths, wavelength_profile[:band_count])
                    self.assertEqual(wrapper_metadata["effective_band_count"], band_count)
                    self.assertTrue(np.isfinite(wrapper).all())
                del mosaic, decoded16, reinterleaved

    def test_real_ladybird_png_is_one_native_scanline_with_local_wavelengths(self):
        path = DATASET_ROOT / "Ladybird_Cobbitty_2017_Brassica_Dataset/example_code/autonomous/20171101_week_04/bed_01/pika-xc2/20171101T004558.420196.png"
        if not path.is_file():
            self.skipTest("real Ladybird Pika-XC2 sample is not mounted")
        import cv2
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        calibration = path.parents[5] / "calibrations" / "pika-xc2" / "spectra.txt"
        wavelengths = np.loadtxt(calibration)
        matrix, resolved, metadata = load_hypervision_matrix(
            str(path), "ladybird_cobbitty", append_rgb=True, rgb_value_mode="copy_hsi", return_metadata=True
        )
        self.assertEqual(raw.shape, (462, 1600))
        self.assertEqual(matrix.shape, (1, 1600, 465))
        self.assertEqual(resolved, wavelengths.tolist())
        self.assertEqual(matrix.dtype, np.float32)
        np.testing.assert_array_equal(matrix[0, :, :462], raw.T.astype(np.float32))
        self.assertEqual(metadata["source"]["spatial_type"], "single_scanline")
        self.assertEqual(metadata["source"]["reconstruction"], "none")
        self.assertEqual(metadata["source"]["scanline_count"], 1)
        self.assertIn("spectra.txt", metadata["source"]["wavelength_source"])
        self.assertEqual(metadata["spatial_alignment"]["method"], "same_native_scanline_samples")
        self.assertEqual(metadata["false_rgb"]["selection_method"], "visible_wavelengths")
        indices = metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(matrix[..., -3:], matrix[..., indices])
        self.assertTrue(np.isfinite(matrix).all())

    def test_real_ghifvd_pca_mat_uses_local_official_decoder(self):
        path = DATASET_ROOT / "GHIFVD/pca/internals_mushroom_SegmentedCroppedCompressed.mat"
        if not path.exists():
            self.skipTest("real GHIFVD PCA MAT sample is not mounted")
        loader = LoadHyperspectralImage("ghifvd", append_rgb=True, rgb_value_mode="copy_hsi")
        img, wavelengths = loader.load(str(path))
        self.assertEqual(img.shape, (176, 208, 348))
        self.assertEqual(len(wavelengths), 345)
        self.assertEqual(loader.channel_metadata["false_rgb"]["wavelength_source"], "ghifvd_matlabHyper_hyperWavelengths_csv")
        self.assertEqual(loader.channel_metadata["false_rgb"]["selection_method"], "visible_wavelengths")
        indices = loader.channel_metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(img[..., -3:], img[..., indices])

    def test_auto_copy_hsi_uses_wavelengths_or_quantiles_and_marks_degeneracy(self):
        visible = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
        rgb, meta = build_false_rgb(visible, np.linspace(400, 700, 12), "fixture", value_mode="copy_hsi")
        expected = [int(np.argmin(np.abs(np.linspace(400, 700, 12) - w))) for w in (650, 550, 450)]
        np.testing.assert_array_equal(rgb, visible[..., expected])
        self.assertEqual(meta["selection_method"], "visible_wavelengths")

        nir = np.arange(10, dtype=np.float32).reshape(1, 1, 10)
        rgb, meta = build_false_rgb(nir, np.linspace(800, 1200, 10), "fixture", value_mode="copy_hsi")
        self.assertEqual(meta["selection_method"], "ordered_channel_quantiles")
        np.testing.assert_array_equal(rgb, nir[..., meta["source_channel_indices"]])

        for count, expected_indices in ((1, [0, 0, 0]), (2, [1, None, 0])):
            cube = np.arange(count, dtype=np.float32).reshape(1, 1, count)
            rgb, meta = build_false_rgb(cube, None, "fixture", spectral_order_known=True, value_mode="copy_hsi")
            if count == 1:
                np.testing.assert_array_equal(rgb, cube[..., [0, 0, 0]])
                self.assertTrue(meta["degeneracy"]["duplicate_channel_indices"])
            else:
                np.testing.assert_array_equal(rgb, np.asarray([[[1.0, 0.5, 0.0]]], dtype=np.float32))
                self.assertEqual(meta["source_channel_indices"], expected_indices)
                self.assertTrue(meta["degeneracy"]["composite_mean_channel"])
            self.assertEqual(meta["rgb_value_mode"], "copy_hsi")
        with self.assertRaisesRegex(ValueError, "known ordered spectral axis"):
            build_false_rgb(np.ones((1, 1, 2)), None, "fixture", spectral_order_known=False)

    def test_estimated_wavelengths_do_not_drive_false_rgb_band_matching(self):
        cube = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
        rgb, metadata = build_false_rgb(
            cube,
            None,
            "aphid",
            mode="auto",
            spectral_order_known=True,
            wavelength_source="linear_range_fallback",
            wavelength_verified=False,
        )
        self.assertEqual(metadata["selection_method"], "ordered_channel_quantiles")
        self.assertEqual(metadata["wavelength_source"], "linear_range_fallback")
        self.assertFalse(metadata["wavelength_verified"])
        self.assertIsNone(metadata["source_wavelengths_nm"])
        with self.assertRaisesRegex(ValueError, "requires per-channel wavelengths"):
            build_false_rgb(cube, None, "aphid", mode="visible")
        estimated = [750.0, 850.0, 950.0]
        with self.assertRaisesRegex(ValueError, "pass allow_estimated_wavelengths=True"):
            build_false_rgb(cube[..., :3], estimated, "aphid", mode="non_visible",
                            rgb_wavelengths=estimated[::-1], wavelength_verified=False)
        _, opted_in = build_false_rgb(
            cube[..., :3], estimated, "aphid", mode="non_visible", rgb_wavelengths=estimated[::-1],
            wavelength_verified=False, allow_estimated_wavelengths=True, value_mode="copy_hsi",
        )
        self.assertTrue(opted_in["allow_estimated_wavelengths"])

    def test_nonmonotonic_verified_wavelengths_do_not_imply_ordered_channels(self):
        cube = np.zeros((2, 2, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "known ordered spectral axis"):
            build_false_rgb(
                cube,
                [1000.0, 400.0, 800.0, 900.0],
                "hydrocarbon_spill",
                mode="auto",
                wavelength_verified=True,
                spectral_order_known=False,
            )

    def test_hydrocarbon_spill_rgb_selection_preserves_header_band_indices(self):
        root = DATASET_ROOT / "Hydrocarbon_Spill_Hyperspectral_Dataset"
        if not root.is_dir():
            self.skipTest("Hydrocarbon Spill samples are not mounted")
        headers = list(root.rglob("*.hdr"))
        if not headers:
            self.skipTest("Hydrocarbon Spill ENVI headers are not mounted")
        loader = LoadHyperspectralImage("hydrocarbon_spill", append_rgb=True, rgb_value_mode="copy_hsi")
        cube, wavelengths = loader.load(str(headers[0]))
        self.assertEqual(cube.shape[-1], 23)
        self.assertEqual(len(wavelengths), 20)
        self.assertEqual(wavelengths[:3], [1000.0, 400.0, 431.6])
        self.assertTrue(loader.wavelength_verified)
        false_rgb = loader.channel_metadata["false_rgb"]
        self.assertEqual(false_rgb["selection_method"], "visible_wavelengths")
        expected_indices = [int(np.argmin(np.abs(np.asarray(wavelengths) - target)))
                            for target in (650.0, 550.0, 450.0)]
        self.assertEqual(false_rgb["source_channel_indices"], expected_indices)
        np.testing.assert_array_equal(cube[..., -3:], cube[..., expected_indices])

    def test_visible_bands_are_nearest_unique_and_fixed_scaled(self):
        wavelengths = [448.0, 451.0, 549.0, 552.0, 649.0, 651.0]
        cube = np.zeros((2, 2, len(wavelengths)), dtype=np.float32)
        cube[..., 1] = 0.25
        cube[..., 2] = 0.50
        cube[..., 4] = 1.25

        profile = resolve_normalization_profile(
            "hsifoodingr64", file_path="REFLECTANCE_fixture.hdr", source_dtype=np.float32, channel_count=204
        )
        rgb, metadata = build_false_rgb(
            cube, wavelengths, "hsifoodingr64", profile=profile, value_mode="display"
        )

        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.float32)
        self.assertEqual(metadata["source_channel_indices"], [4, 2, 1])
        np.testing.assert_allclose(rgb[0, 0], [255.0, 127.5, 63.75])
        self.assertEqual(metadata["normalization"]["method"], "selected_from_fixed_mapped_hsi")
        self.assertEqual(metadata["normalization"]["profile_kind"], "reflectance")
        self.assertEqual(metadata["normalization"]["gamma"], 1.0)
        self.assertTrue(metadata["source_band_diagnostics"][0]["constant"])
        self.assertEqual(metadata["source_band_diagnostics"][0]["observed_min"], 1.25)

    def test_non_visible_requires_explicit_targets(self):
        wavelengths = [800.0, 900.0, 1000.0]
        cube = np.zeros((1, 1, 3), dtype=np.uint8)
        cube[0, 0] = [0, 128, 255]

        with self.assertRaisesRegex(ValueError, "requires three explicit"):
            build_false_rgb(cube, wavelengths, "near_infrared", mode="non_visible")

        rgb, metadata = build_false_rgb(
            cube,
            wavelengths,
            "near_infrared",
            mode="non_visible",
            rgb_wavelengths=(1000.0, 900.0, 800.0),
            value_mode="copy_hsi",
        )
        np.testing.assert_allclose(rgb[0, 0], [255.0, 128.0, 0.0])
        self.assertEqual(metadata["source_channel_indices"], [2, 1, 0])

        with self.assertRaisesRegex(ValueError, "30 nm tolerance"):
            build_false_rgb(
                cube,
                wavelengths,
                "near_infrared",
                mode="non_visible",
                rgb_wavelengths=(1200.0, 900.0, 800.0),
            )

    def test_hsifood_non_visible_accepts_explicit_targets(self):
        wavelengths = [750.54, 850.29, 951.17]
        cube = np.zeros((1, 1, 3), dtype=np.float32)
        cube[0, 0] = [0.25, 0.5, 0.75]
        profile = resolve_normalization_profile(
            "hsifoodingr64", file_path="REFLECTANCE_fixture.hdr", source_dtype=np.float32, channel_count=204
        )
        rgb, metadata = build_false_rgb(
            cube,
            wavelengths,
            "hsifoodingr64",
            mode="non_visible",
            rgb_wavelengths=(750.0, 850.0, 950.0),
            profile=profile,
            value_mode="display",
        )
        np.testing.assert_allclose(rgb[0, 0], [63.75, 127.5, 191.25])
        self.assertEqual(metadata["source_channel_indices"], [0, 1, 2])

    def test_visible_tolerance_and_unverified_profiles_fail_closed(self):
        cube = np.ones((1, 1, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            build_false_rgb(cube, [500.0, 550.0, 600.0], "hsifoodingr64", mode="visible")
        _, metadata = build_false_rgb(cube, [450.0, 550.0, 650.0], "cocoa_beans", value_mode="display")
        self.assertEqual(metadata["normalization"]["method"], "per_channel_image_minmax")

    def test_non_finite_rgb_source_band_fails(self):
        cube = np.ones((1, 1, 3), dtype=np.float32)
        cube[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            build_false_rgb(cube, [450.0, 550.0, 650.0], "hsifoodingr64")

    def test_negative_values_are_clipped_and_reported_without_rescaling(self):
        cube = np.zeros((2, 2, 3), dtype=np.float32)
        cube[..., 0] = -0.25
        cube[..., 1] = 0.5
        cube[..., 2] = 1.25
        profile = resolve_normalization_profile(
            "hsifoodingr64", file_path="REFLECTANCE_fixture.hdr", source_dtype=np.float32, channel_count=204
        )
        rgb, metadata = build_false_rgb(
            cube, [650.0, 550.0, 450.0], "hsifoodingr64", profile=profile, value_mode="display"
        )
        np.testing.assert_allclose(rgb[0, 0], [0.0, 127.5, 255.0])
        diagnostics = metadata["source_band_diagnostics"]
        self.assertEqual(diagnostics[0]["negative_sample_count"], 4)
        self.assertEqual(diagnostics[0]["observed_min"], -0.25)
        self.assertTrue(all(item["constant"] for item in diagnostics))

    def test_swir_mineral_profile_fails_closed_without_cube_calibration(self):
        wavelengths = [997.562, 1501.966, 2097.592]
        cube = np.zeros((1, 1, 3), dtype=np.float32)
        cube[0, 0] = [0.25, 0.50, 0.75]
        rgb, metadata = build_false_rgb(
                cube,
                wavelengths,
                "mineral_abundance",
                mode="non_visible",
                rgb_wavelengths=(2100.0, 1500.0, 1000.0),
                value_mode="copy_hsi",
        )
        np.testing.assert_array_equal(rgb, cube[..., [2, 1, 0]])
        self.assertEqual(metadata["rgb_value_mode"], "copy_hsi")
        rgb_display, display_metadata = build_false_rgb(
            cube, wavelengths, "mineral_abundance", mode="non_visible",
            rgb_wavelengths=(2100.0, 1500.0, 1000.0), value_mode="display",
        )
        self.assertEqual(display_metadata["normalization"]["method"], "per_channel_image_minmax")
        self.assertIn(
            "no verified fixed normalization",
            display_metadata["normalization"]["profile_unavailable_reason"],
        )
        self.assertTrue(np.isfinite(rgb_display).all())

    def test_real_mineral_sample_copy_hsi_succeeds(self):
        root = DATASET_ROOT
        hdr_path = next(root.glob("Proximal_Multi-Scenario_SWIR_Hyperspectral_Mineral_Abundance_Dataset/**/OC0007R2.hdr"), None)
        if hdr_path is None:
            self.skipTest("real Mineral Abundance sample is not mounted")
        loader = LoadHyperspectralImage("mineral_abundance", append_rgb=True, rgb_value_mode="copy_hsi")
        img, wavelengths = loader.load(str(hdr_path))
        self.assertEqual(img.shape[-1], len(wavelengths) + 3)
        selected = loader.channel_metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(img[..., -3:], img[..., selected])

    def test_real_hyperpri_dat_uses_unique_ancestor_header(self):
        path = DATASET_ROOT / "HyperPRI/box_material/lexan/20240306_lexan_dry_ref.dat"
        if not path.exists():
            self.skipTest("real HyperPRI .dat sample is not mounted")
        loader = LoadHyperspectralImage("hyperpri", append_rgb=True, rgb_value_mode="copy_hsi")
        img, wavelengths = loader.load(str(path))
        self.assertEqual(img.shape, (608, 968, 302))
        self.assertEqual(len(wavelengths), 299)
        self.assertEqual(loader.channel_metadata["false_rgb"]["wavelength_source"], "file_header")
        selected = loader.channel_metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(img[..., -3:], img[..., selected])

    def test_rgb_wavelengths_must_match_spectral_cube_and_be_distinct(self):
        cube = np.ones((1, 1, 6), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "already contains appended channels"):
            build_false_rgb(cube, [450.0, 550.0, 650.0], "hsifoodingr64")

    def test_wrapper_appends_once_and_keeps_spectral_wavelengths_separate(self):
        wavelengths = np.linspace(400.0, 1000.0, 96)
        cube = np.broadcast_to(np.linspace(40, 220, 96), (100, 101, 96)).astype(np.uint8).copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            cube_path = Path(temp_dir) / "encoded_cube.npy"
            np.save(cube_path, cube)

            legacy_matrix, legacy_wavelengths = load_hypervision_matrix(
                str(cube_path), "hsi1"
            )
            matrix, spectral_wavelengths, metadata = load_hypervision_matrix(
                str(cube_path),
                "hsi1",
                append_rgb=True,
                rgb_value_mode="copy_hsi",
                return_metadata=True,
            )

            loader = LoadHyperspectralImage("hsi1", append_rgb=True, rgb_value_mode="copy_hsi")
            loader.load(str(cube_path))
            self.assertIsNotNone(loader.channel_metadata)
            loader.dataset_type = "cocoa_beans"
            loader.load(str(cube_path))
            self.assertEqual(loader.channel_metadata["false_rgb"]["rgb_value_mode"], "copy_hsi")

        self.assertEqual(legacy_matrix.shape, (100, 101, 96))
        self.assertEqual(len(legacy_wavelengths), 96)
        self.assertEqual(matrix.shape, (100, 101, 99))
        self.assertEqual(len(spectral_wavelengths), 96)
        self.assertEqual(metadata["spectral_channel_count"], 96)
        self.assertEqual(len(metadata["appended_channel_descriptors"]), 3)
        self.assertFalse(metadata["false_rgb"]["normalization"]["image_extrema_scaling"])
        np.testing.assert_array_equal(matrix[..., :96], cube.astype(np.float32))
        expected_legacy = 255.0 * (cube.astype(np.float32) - cube.min()) / (cube.max() - cube.min())
        np.testing.assert_array_equal(legacy_matrix, expected_legacy)
        self.assertFalse(np.array_equal(matrix[..., :96], legacy_matrix))
        selected = metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(matrix[..., 96:], matrix[..., selected])
        self.assertEqual(metadata["hsi_mapping"]["method"], "identity")

    def test_hdf5_and_mat_format_fixtures_append_exact_source_channels(self):
        # Synthetic format fixtures only; these do not represent real HSI-1 data.
        cube = np.broadcast_to(np.linspace(0, 255, 96, dtype=np.uint8), (100, 101, 96)).copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = Path(temp_dir) / "cube.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("cube", data=cube)
            mat_path = Path(temp_dir) / "cube.mat"
            savemat(mat_path, {"cube": cube, "wavelengths": np.linspace(400, 1000, 96)})

            for path in (h5_path, mat_path):
                loader = LoadHyperspectralImage("hsi1", append_rgb=True)
                loaded, wavelengths = loader.load(str(path))
                self.assertEqual(loaded.shape, (100, 101, 99))
                self.assertEqual(len(wavelengths), 96)
                self.assertTrue(np.allclose(loaded[..., :96], cube))
                selected = loader.channel_metadata["false_rgb"]["source_channel_indices"]
                np.testing.assert_array_equal(loaded[..., 96:], loaded[..., selected])

    def test_real_hyperleaf_uint16_tiff_copy_hsi_succeeds_without_scale_factor(self):
        path = DATASET_ROOT / "HyperLeaf2024/images/00000.tiff"
        if not path.exists():
            self.skipTest("real HyperLeaf2024 sample is not mounted")
        legacy_loader = LoadHyperspectralImage("hyperleaf2024", append_rgb=False)
        legacy_img, legacy_wavelengths = legacy_loader.load(str(path))
        self.assertEqual(legacy_img.shape, (204, 48, 352))
        self.assertEqual(len(legacy_wavelengths), 204)
        self.assertEqual(legacy_loader.wavelength_metadata["spectral_axis"], 0)
        self.assertTrue(legacy_loader.wavelength_verified)

        loader = LoadHyperspectralImage("hyperleaf2024", append_rgb=True, rgb_value_mode="copy_hsi")
        img, wavelengths = loader.load(str(path))
        self.assertEqual(img.shape[-1], 207)
        self.assertEqual(len(wavelengths), 204)
        self.assertEqual(wavelengths, legacy_wavelengths)
        self.assertEqual(loader.channel_metadata["spectral_axis"], -1)
        selected = loader.channel_metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(img[..., -3:], img[..., selected])

    def test_v1_envi_real_samples_keep_header_spectral_axis(self):
        samples = [
            (
                "vnihdhiatlimafb",
                ARCHIVE_TEST_ROOT / "Visible Near infrared hyperspectral dataset of healthy and infected apple tree leaves images for the monitoring of apple fire blight/data_July2022/day_8/plant_3/REFLECTANCE_2082.hdr",
                (512, 512, 204),
                (397.32, 1003.58),
            ),
            (
                "virginia_tech_tree",
                ARCHIVE_TEST_ROOT / "Virginia_Tech_Trees/CamperDownElmLeafSny_exp_VNIR_hdr_raw/CamperDownElmLeafSny_exp20000_VNIR.hdr",
                (401, 800, 420),
                (394.7, 1005.86),
            ),
        ]
        if not any(path.is_file() for _, path, _, _ in samples):
            self.skipTest("V1 Apple Fire Blight / Virginia Tech ENVI fixtures are not mounted")
        for dataset, path, expected_shape, expected_range in samples:
            if not path.is_file():
                continue
            with self.subTest(dataset=dataset):
                loader = LoadHyperspectralImage(dataset)
                cube, wavelengths = loader.load(str(path))
                self.assertEqual(cube.shape, expected_shape)
                self.assertEqual(len(wavelengths), expected_shape[-1])
                self.assertAlmostEqual(min(wavelengths), expected_range[0], places=2)
                self.assertAlmostEqual(max(wavelengths), expected_range[1], places=2)
                self.assertTrue(loader.wavelength_verified)
                self.assertEqual(loader.wavelength_source, "file_header")
                self.assertTrue(np.isfinite(cube).all())

    def test_real_hyperdrive_vnir_npz_cube_loads_and_reports_estimated_centers(self):
        path = ARCHIVE_TEST_ROOT / "hyperdrive/working/HSI_REGISTERED_VNIR/olin_collection_05_16_23_parcel_b_v2_16.npz"
        if not path.is_file():
            self.skipTest("HyperDrive VNIR NPZ sample is not mounted")
        loader = LoadHyperspectralImage("hyperdrivevnir")
        cube, wavelengths = loader.load(str(path))
        self.assertEqual(cube.shape, (1016, 1671, 33))
        self.assertEqual(len(wavelengths), cube.shape[-1])
        self.assertEqual((min(wavelengths), max(wavelengths)), (660.0, 900.0))
        self.assertFalse(loader.wavelength_verified)
        self.assertEqual(loader.wavelength_source, "linear_range_fallback")
        self.assertTrue(np.isfinite(cube).all())

    def test_hyperblood_wavelengths_follow_the_same_band_filter_as_cube(self):
        path = ARCHIVE_TEST_ROOT / "HyperBlood/data/E_21.tif"
        if not path.is_file():
            self.skipTest("HyperBlood sample is not mounted")
        from hyperspectral_image_reader_v2.utils import ds_load
        reference_cube, reference_wavelengths = ds_load.get_data(str(path))
        loader = LoadHyperspectralImage("hyperblood")
        cube, wavelengths = loader.load(str(path))
        self.assertEqual(cube.shape, (519, 696, 113))
        self.assertEqual(len(wavelengths), cube.shape[-1])
        np.testing.assert_array_equal(cube, reference_cube)
        np.testing.assert_array_equal(wavelengths, reference_wavelengths)
        self.assertEqual(loader.wavelength_source, "dataset_spectral_calibration_vector")
        self.assertTrue(loader.wavelength_verified)

    def test_real_hsi_road_tiff_uses_estimated_wavelengths_and_quantile_rgb(self):
        path = ARCHIVE_TEST_ROOT / "hsi_road/images/003720_nir.tif"
        if not path.is_file():
            self.skipTest("HSI Road TIFF sample is not mounted")
        loader = LoadHyperspectralImage("hsiroad", append_rgb=True, rgb_value_mode="copy_hsi")
        cube, wavelengths = loader.load(str(path))
        self.assertEqual(cube.shape, (192, 384, 28))
        self.assertEqual(len(wavelengths), 25)
        self.assertFalse(loader.wavelength_verified)
        self.assertEqual(loader.channel_metadata["false_rgb"]["selection_method"], "ordered_channel_quantiles")
        selected = loader.channel_metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(cube[..., -3:], cube[..., selected])
        self.assertTrue(np.isfinite(cube).all())

    def test_real_hyperskin_nir_mat_is_calibrated_and_appendable(self):
        path = DATASET_ROOT / "Hyper-Skin/Hyper-Skin(MSI, NIR)/test/NIR/p002_neutral_front.mat"
        if not path.exists():
            self.skipTest("real Hyper-Skin NIR sample is not mounted")
        img, wavelengths = LoadHyperspectralImage(
            "hyper_skin", append_rgb=True, rgb_mode="non_visible", rgb_wavelengths=(900, 850, 750),
            allow_estimated_wavelengths=True,
        ).load(str(path))
        self.assertEqual(img.shape, (1024, 1024, 34))
        self.assertEqual(len(wavelengths), 31)
        self.assertGreaterEqual(min(wavelengths), 700.0)
        self.assertEqual(img.dtype, np.float32)

    def test_real_hyperskin_vis_mat_is_calibrated_and_appendable(self):
        path = DATASET_ROOT / "Hyper-Skin/Hyper-Skin(RGB, VIS)/test/VIS/p002_neutral_front.mat"
        if not path.exists():
            self.skipTest("real Hyper-Skin VIS sample is not mounted")
        img, wavelengths = LoadHyperspectralImage("hyper_skin", append_rgb=True).load(str(path))
        self.assertEqual(img.shape, (1024, 1024, 34))
        self.assertEqual(len(wavelengths), 31)
        self.assertGreaterEqual(min(wavelengths), 400.0)
        self.assertLessEqual(max(wavelengths), 700.0)

    def test_real_hyperskin_msi_uses_documented_four_channel_order(self):
        path = DATASET_ROOT / "Hyper-Skin/Hyper-Skin(MSI, NIR)/test/MSI/p002_neutral_front.mat"
        if not path.exists():
            self.skipTest("real Hyper-Skin MSI sample is not mounted")
        loader = LoadHyperspectralImage("hyper_skin", append_rgb=True, rgb_value_mode="copy_hsi")
        img, wavelengths = loader.load(str(path))
        self.assertEqual(len(wavelengths), 4)
        self.assertEqual(wavelengths, [450.0, 520.0, 600.0, 850.0])
        self.assertEqual(img.shape[-1], 7)
        indices = loader.channel_metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(img[..., -3:], img[..., indices])
        self.assertEqual(loader.channel_metadata["false_rgb"]["selection_method"], "ordered_channel_quantiles")

    def test_real_near_infrared_stack_is_sorted_by_numeric_wavelength(self):
        path = DATASET_ROOT / "Near_Infrared_Hyperspectral_Image_Dataset/data/coffee/coffee_1293nm.png"
        if not path.exists():
            self.skipTest("real Near Infrared image stack is not mounted")
        loader = LoadHyperspectralImage("near_infrared", append_rgb=True)
        img, wavelengths = loader.load(str(path))
        self.assertEqual(img.shape, (256, 192, 99))
        self.assertTrue(np.all(np.diff(wavelengths) > 0))
        selected = loader.channel_metadata["false_rgb"]["source_channel_indices"]
        np.testing.assert_array_equal(img[..., -3:], img[..., selected])

    def test_hyperskin_448_full_cube_is_not_enabled_without_local_sample(self):
        with self.assertRaisesRegex(ValueError, "31-band VIS or NIR"):
            resolve_normalization_profile(
                "hyper_skin", "/fixture-data/Hyper-Skin/full_cube.mat", np.float32, channel_count=448
            )

    def test_cabbage_reflectance_profile_is_product_path_scoped(self):
        reflectance_path = str(DATASET_ROOT / "Cabbage_Eggplant_combined_Hyperspectral_dataset/Cabbage_Crop/Cabbage_Reflectance_Data.hdr")
        radiance_path = reflectance_path.replace("Reflectance_Data", "Radiance_Data")
        if not Path(reflectance_path).exists():
            self.skipTest("real Cabbage Reflectance_Data header is not mounted")
        header = Path(reflectance_path).read_text(errors="replace").lower()
        self.assertRegex(header, r"bands\s*=\s*277")
        self.assertIn("wavelength units = nanometers", header)
        profile = resolve_normalization_profile(
            "cabbage_eggplant", reflectance_path, np.float32, channel_count=277
        )
        self.assertEqual(profile["kind"], "reflectance")
        # Synthetic cube fixture with the real product-specific profile.
        fixture_cube = np.broadcast_to(np.linspace(0.0, 1.0, 277), (1, 1, 277)).astype(np.float32)
        fixture_wavelengths = np.linspace(399.86, 949.60, 277)
        rgb, metadata = build_false_rgb(
            fixture_cube, fixture_wavelengths, "cabbage_eggplant", profile=profile, value_mode="display"
        )
        self.assertEqual(rgb.shape, (1, 1, 3))
        self.assertEqual(metadata["normalization"]["profile_kind"], "reflectance")
        with self.assertRaisesRegex(ValueError, "applies only"):
            resolve_normalization_profile("cabbage_eggplant", radiance_path, np.float32, channel_count=277)

    def test_tecnalia_profile_remains_unsupported_without_fixed_mat_scale(self):
        cube = np.ones((1, 1, 76), dtype=np.float32)
        _, metadata = build_false_rgb(cube, np.linspace(415.05, 1008.1, 76), "tecnalia", value_mode="display")
        self.assertEqual(metadata["normalization"]["method"], "per_channel_image_minmax")

    def test_almond_split_sample_calibrates_from_matched_references(self):
        path = DATASET_ROOT / "hyperspectral_image_datasets-almond_pistachio_and_garlic_stems/Almond/Train/data.hdr"
        if not path.exists():
            self.skipTest("real almond data/reference triplet is not mounted")
        loader = LoadHyperspectralImage("almond_pistachio_garlic", append_rgb=True, rgb_value_mode="display")
        img, wavelengths = loader.load(str(path))
        calibration = loader.channel_metadata["hsi_mapping"]["calibration"]
        self.assertEqual(img.shape, (400, 512, 227))
        self.assertEqual(len(wavelengths), 224)
        self.assertEqual(calibration["profile_kind"], "calibrated_reference_ratio")
        self.assertEqual(calibration["wavelength_match"], True)
        self.assertEqual(calibration["data_exposure_time"], calibration["reference_exposure_times"][0])
        self.assertEqual(calibration["invalid_data_sample_count"], 0)
        self.assertEqual(calibration["invalid_reference_sample_count"], 0)
        self.assertEqual(calibration["near_zero_denominator_sample_band_count"], 0)

    def test_almond_calibration_fails_when_reference_pair_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "Train"
            split.mkdir()
            data_path = split / "data.hdr"
            data_path.touch()
            with self.assertRaisesRegex(ValueError, "requires a unique split-local WHITEREF"):
                resolve_normalization_profile(
                    "almond_pistachio_garlic", str(data_path), np.uint16, channel_count=224
                )


if __name__ == "__main__":
    unittest.main()
