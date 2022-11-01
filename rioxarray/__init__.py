"""Top-level package for rioxarray."""
__author__ = """rioxarray Contributors"""
import importlib.metadata

import rioxarray.raster_array  # noqa
import rioxarray.raster_dataset  # noqa
from rioxarray._io import open_rasterio  # noqa
from rioxarray._options import set_options  # noqa
from rioxarray._show_versions import show_versions  # noqa

__version__ = importlib.metadata.version(__package__)
