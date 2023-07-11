"""Top-level package for rioxarray."""
__author__ = """rioxarray Contributors"""
import importlib.metadata

import rioxarray.raster_array  # noqa
import rioxarray.raster_dataset  # noqa
from rioxarray._io import open_rasterio
from rioxarray._options import set_options
from rioxarray._show_versions import show_versions

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "open_rasterio",
    "set_options",
    "show_versions",
    "__author__",
    "__version__",
]
