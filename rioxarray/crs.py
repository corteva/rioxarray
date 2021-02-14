# -- coding: utf-8 --
"""
This module contains helper methods for rasterio.crs.CRS.
"""
from distutils.version import LooseVersion

import rasterio
import rasterio.crs
from pyproj import CRS
from rasterio.errors import CRSError


def crs_from_user_input(crs_input):
    """
    Return a rasterio.crs.CRS from user input.

    This is to deal with change in rasterio.crs.CRS
    as well as to assist in the transition between GDAL 2/3.

    Parameters
    ----------
    crs_input: Any
        Input to create a CRS.

    Returns
    -------
    rasterio.crs.CRS

    """
    if isinstance(crs_input, rasterio.crs.CRS):
        return crs_input
    try:
        # old versions of opendatacube CRS
        crs_input = crs_input.wkt
    except AttributeError:
        pass
    try:
        return rasterio.crs.CRS.from_user_input(crs_input)
    except CRSError:
        pass
    # use pyproj for edge cases
    crs = CRS.from_user_input(crs_input)
    if LooseVersion(rasterio.__gdal_version__) > LooseVersion("3.0.0"):
        return rasterio.crs.CRS.from_wkt(crs.to_wkt())
    return rasterio.crs.CRS.from_wkt(crs.to_wkt("WKT1_GDAL"))
