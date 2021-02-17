"""Top-level package for rioxarray."""

__author__ = """rioxarray Contributors"""
__email__ = "alansnow21@gmail.com"

import rioxarray.raster_array  # noqa
import rioxarray.raster_dataset  # noqa
from rioxarray._options import set_options  # noqa
from rioxarray._show_versions import show_versions  # noqa
from rioxarray._version import __version__  # noqa

try:
    # This requires xarray >= 0.12.3
    from rioxarray._io import open_rasterio  # noqa
except ImportError:
    from xarray import open_rasterio  # noqa
