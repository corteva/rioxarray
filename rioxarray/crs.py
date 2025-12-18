"""
This module contains helper methods for rasterio.crs.CRS.
"""
from typing import Any

import rasterio
import rasterio.crs
from pyproj import CRS
from rasterio.errors import CRSError


def crs_from_user_input(crs_input: Any) -> rasterio.crs.CRS:
    """
    Return a rasterio.crs.CRS from user input.

    This is to deal with change in rasterio.crs.CRS
    as well as to assist in the transition between GDAL 2/3.

    Parameters
    ----------
    crs_input: Any
        Input to create a CRS. Can be:
        - rasterio.crs.CRS object
        - WKT string
        - PROJ string
        - EPSG code (int or string)
        - PROJJSON dict (Zarr proj:projjson format)

    Returns
    -------
    rasterio.crs.CRS

    """
    if isinstance(crs_input, rasterio.crs.CRS):
        return crs_input

    # Handle PROJJSON dict (Zarr proj:projjson convention)
    if isinstance(crs_input, dict):
        crs_input = CRS.from_json_dict(crs_input)

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
    return rasterio.crs.CRS.from_wkt(crs.to_wkt())
