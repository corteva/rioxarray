# -- coding: utf-8 --
"""
This module contains helper methods for rasterio.crs.CRS.
"""


def crs_to_wkt(rasterio_crs):
    """
    This is to deal with change in rasterio.crs.CRS.

    Parameters
    ----------
    rasterio_crs: :obj:`rasterio.crs.CRS`
        Rasterio object.

    Returns
    -------
    str: WKT string.

    """
    try:
        # rasterio>=1.0.14
        return rasterio_crs.to_wkt()
    except AttributeError:
        return rasterio_crs.wkt
