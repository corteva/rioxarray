"""
Utilities for reading and writing Zarr spatial and proj conventions.

This module provides functions for parsing and formatting metadata according to:
- Zarr spatial convention: https://github.com/zarr-conventions/spatial
- Zarr geo-proj convention: https://github.com/zarr-experimental/geo-proj
"""

import json
from typing import Optional, Tuple, Union

import rasterio.crs
from affine import Affine
from pyproj import CRS


def parse_spatial_transform(spatial_transform: Union[list, tuple]) -> Optional[Affine]:
    """
    Convert spatial:transform array to Affine object.

    Parameters
    ----------
    spatial_transform : list or tuple
        Affine transformation coefficients [a, b, c, d, e, f]

    Returns
    -------
    affine.Affine or None
        Affine transformation object, or None if invalid

    Examples
    --------
    >>> parse_spatial_transform([1.0, 0.0, 0.0, 0.0, -1.0, 1024.0])
    Affine(1.0, 0.0, 0.0,
           0.0, -1.0, 1024.0)
    """
    if not isinstance(spatial_transform, (list, tuple)):
        return None
    if len(spatial_transform) != 6:
        return None
    try:
        # spatial:transform format is [a, b, c, d, e, f]
        # which maps directly to Affine(a, b, c, d, e, f)
        return Affine(*spatial_transform)
    except (TypeError, ValueError):
        return None


def format_spatial_transform(affine: Affine) -> list:
    """
    Convert Affine object to spatial:transform array format.

    Parameters
    ----------
    affine : affine.Affine
        Affine transformation object

    Returns
    -------
    list
        Affine transformation coefficients [a, b, c, d, e, f]

    Examples
    --------
    >>> from affine import Affine
    >>> affine = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1024.0)
    >>> format_spatial_transform(affine)
    [1.0, 0.0, 0.0, 0.0, -1.0, 1024.0]
    """
    # Convert Affine to list [a, b, c, d, e, f]
    return list(affine)[:6]


def parse_proj_code(proj_code: str) -> Optional[rasterio.crs.CRS]:
    """
    Parse proj:code (e.g., 'EPSG:4326') to CRS.

    Parameters
    ----------
    proj_code : str
        Authority:code identifier (e.g., "EPSG:4326", "IAU_2015:30100")

    Returns
    -------
    rasterio.crs.CRS or None
        CRS object, or None if invalid

    Examples
    --------
    >>> parse_proj_code("EPSG:4326")
    CRS.from_epsg(4326)
    """
    if not isinstance(proj_code, str):
        return None
    try:
        return rasterio.crs.CRS.from_string(proj_code)
    except Exception:
        return None


def format_proj_code(crs: rasterio.crs.CRS) -> Optional[str]:
    """
    Format CRS as proj:code if it has an authority code.

    Parameters
    ----------
    crs : rasterio.crs.CRS
        CRS object

    Returns
    -------
    str or None
        Authority:code string (e.g., "EPSG:4326"), or None if no authority

    Examples
    --------
    >>> crs = rasterio.crs.CRS.from_epsg(4326)
    >>> format_proj_code(crs)
    'EPSG:4326'
    """
    try:
        # Try to get the authority and code
        auth_code = crs.to_authority()
        if auth_code:
            authority, code = auth_code
            return f"{authority}:{code}"
    except Exception:
        pass
    return None


def parse_proj_wkt2(proj_wkt2: str) -> Optional[rasterio.crs.CRS]:
    """
    Parse proj:wkt2 to CRS.

    Parameters
    ----------
    proj_wkt2 : str
        WKT2 (ISO 19162) CRS representation

    Returns
    -------
    rasterio.crs.CRS or None
        CRS object, or None if invalid

    Examples
    --------
    >>> wkt2 = 'GEOGCS["WGS 84",DATUM["WGS_1984",...'
    >>> parse_proj_wkt2(wkt2)
    CRS.from_wkt(wkt2)
    """
    if not isinstance(proj_wkt2, str):
        return None
    try:
        return rasterio.crs.CRS.from_wkt(proj_wkt2)
    except Exception:
        return None


def format_proj_wkt2(crs: rasterio.crs.CRS) -> str:
    """
    Format CRS as proj:wkt2 (WKT2 string).

    Parameters
    ----------
    crs : rasterio.crs.CRS
        CRS object

    Returns
    -------
    str
        WKT2 string representation

    Examples
    --------
    >>> crs = rasterio.crs.CRS.from_epsg(4326)
    >>> wkt2 = format_proj_wkt2(crs)
    >>> 'GEOGCS' in wkt2 or 'GEOGCRS' in wkt2
    True
    """
    return crs.to_wkt()


def parse_proj_projjson(proj_projjson: Union[dict, str]) -> Optional[rasterio.crs.CRS]:
    """
    Parse proj:projjson to CRS.

    Parameters
    ----------
    proj_projjson : dict or str
        PROJJSON CRS representation (dict or JSON string)

    Returns
    -------
    rasterio.crs.CRS or None
        CRS object, or None if invalid

    Examples
    --------
    >>> projjson = {"type": "GeographicCRS", ...}
    >>> parse_proj_projjson(projjson)
    CRS.from_json(projjson)
    """
    if isinstance(proj_projjson, str):
        try:
            proj_projjson = json.loads(proj_projjson)
        except json.JSONDecodeError:
            return None

    if not isinstance(proj_projjson, dict):
        return None

    try:
        # pyproj CRS can parse PROJJSON
        pyproj_crs = CRS.from_json_dict(proj_projjson)
        # Convert to rasterio CRS
        return rasterio.crs.CRS.from_wkt(pyproj_crs.to_wkt())
    except Exception:
        return None


def format_proj_projjson(crs: rasterio.crs.CRS) -> dict:
    """
    Format CRS as proj:projjson (PROJJSON dict).

    Parameters
    ----------
    crs : rasterio.crs.CRS
        CRS object

    Returns
    -------
    dict
        PROJJSON representation

    Examples
    --------
    >>> crs = rasterio.crs.CRS.from_epsg(4326)
    >>> projjson = format_proj_projjson(crs)
    >>> projjson["type"]
    'GeographicCRS'
    """
    # Convert to pyproj CRS to get PROJJSON
    pyproj_crs = CRS.from_wkt(crs.to_wkt())
    projjson_str = pyproj_crs.to_json()
    return json.loads(projjson_str)


def calculate_spatial_bbox(
    transform: Affine, shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Calculate spatial:bbox [xmin, ymin, xmax, ymax] from transform and shape.

    Parameters
    ----------
    transform : affine.Affine
        Affine transformation
    shape : tuple of int
        Shape as (height, width)

    Returns
    -------
    tuple of float
        Bounding box as (xmin, ymin, xmax, ymax)

    Examples
    --------
    >>> from affine import Affine
    >>> transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 1024.0)
    >>> shape = (1024, 1024)
    >>> calculate_spatial_bbox(transform, shape)
    (0.0, 0.0, 1024.0, 1024.0)
    """
    height, width = shape

    # Calculate corners in pixel coordinates
    corners_px = [
        (0, 0),  # top-left
        (width, 0),  # top-right
        (width, height),  # bottom-right
        (0, height),  # bottom-left
    ]

    # Transform to spatial coordinates
    corners_spatial = [transform * corner for corner in corners_px]

    # Extract x and y coordinates
    xs = [x for x, y in corners_spatial]
    ys = [y for x, y in corners_spatial]

    # Return bounding box
    return (min(xs), min(ys), max(xs), max(ys))


def validate_spatial_registration(registration: str) -> None:
    """
    Validate spatial:registration value ('pixel' or 'node').

    Parameters
    ----------
    registration : str
        Registration type to validate

    Raises
    ------
    ValueError
        If registration is not 'pixel' or 'node'

    Examples
    --------
    >>> validate_spatial_registration("pixel")
    >>> validate_spatial_registration("node")
    >>> validate_spatial_registration("invalid")
    Traceback (most recent call last):
        ...
    ValueError: spatial:registration must be 'pixel' or 'node', got 'invalid'
    """
    valid_values = {"pixel", "node"}
    if registration not in valid_values:
        raise ValueError(
            f"spatial:registration must be 'pixel' or 'node', got '{registration}'"
        )
