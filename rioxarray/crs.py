# -- coding: utf-8 --
"""
This module contains helper methods for rasterio.crs.CRS.
"""
from distutils.version import LooseVersion

import rasterio
from pyproj import CRS


def crs_to_wkt(crs_input):
    """
    This is to deal with change in rasterio.crs.CRS
    as well as to assist in the transition between GDAL 2/3.

    Parameters
    ----------
    crs_input: Any
        Input to create a CRS.

    Returns
    -------
    str: WKT string.

    """
    try:
        # rasterio.crs.CRS <1.0.14 and
        # old versions of opendatacube CRS
        crs_input = crs_input.wkt
    except AttributeError:
        pass
    crs = CRS.from_user_input(crs_input)
    if LooseVersion(rasterio.__gdal_version__) > LooseVersion("3.0.0"):
        return crs.to_wkt()
    return crs.to_wkt("WKT1_GDAL")
