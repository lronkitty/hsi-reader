"""
Hyperspectral Image Reader V2
Extended dataset reader supporting comprehensive HSI datasets with dynamic multi-wavelength dispatch.
"""

from .wavelength_configs import WAVELENGTH_CONFIGS, resolve_wavelengths
from .hyperspectral_pipelines import LoadHyperspectralImage
from .read_dataset_image import load_hypervision_matrix, DATASET_EXTENSIONS
from .utils import ds_load, get_image_loader

__all__ = [
    'WAVELENGTH_CONFIGS',
    'resolve_wavelengths',
    'LoadHyperspectralImage',
    'load_hypervision_matrix',
    'DATASET_EXTENSIONS',
    'ds_load',
    'get_image_loader',
]
