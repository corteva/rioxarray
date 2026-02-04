"""
Core convention methods for rioxarray.
"""

from typing import Optional, Union

import rasterio.crs
import xarray
from affine import Affine

from rioxarray._convention._base import ConventionProtocol
from rioxarray._convention.cf import CFConvention
from rioxarray._convention.zarr import ZarrConvention
from rioxarray._options import CONVENTION, get_option
from rioxarray.crs import crs_from_user_input
from rioxarray.enum import Convention

# Convention classes mapped by Convention enum
_CONVENTION_MODULES: dict[Convention, ConventionProtocol] = {
    Convention.CF: CFConvention,  # type: ignore[dict-item]
    Convention.Zarr: ZarrConvention,  # type: ignore[dict-item]
}


def _get_convention(convention: Convention | None) -> ConventionProtocol:
    """
    Get the convention module for writing.

    Parameters
    ----------
    convention : Convention enum value or None
        The convention to use. If None, uses the global default.

    Returns
    -------
    ConventionProtocol
        The module implementing the convention
    """
    if convention is None:
        convention = get_option(CONVENTION) or Convention.CF
    convention = Convention(convention)
    return _CONVENTION_MODULES[convention]


def read_crs_auto(
    obj: Union[xarray.Dataset, xarray.DataArray],
    **kwargs,
) -> Optional[rasterio.crs.CRS]:
    """
    Auto-detect and read CRS by trying convention readers.

    If a convention is set globally via set_options(), that convention
    is tried first for better performance. Then other conventions are
    tried as fallback.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read CRS from
    **kwargs
        Convention-specific parameters (e.g., grid_mapping for CF)

    Returns
    -------
    rasterio.crs.CRS or None
        CRS object, or None if not found in any convention
    """
    # Try the configured convention first (if set)
    configured_convention = get_option(CONVENTION)
    if configured_convention is not None:
        result = _CONVENTION_MODULES[configured_convention].read_crs(obj, **kwargs)
        if result is not None:
            return result

    # Try all other conventions
    for conv_enum, convention in _CONVENTION_MODULES.items():
        if conv_enum == configured_convention:
            continue  # Already tried this one
        result = convention.read_crs(obj, **kwargs)
        if result is not None:
            return result

    # Legacy fallback: look in attrs for 'crs' (not part of any convention)
    try:
        return crs_from_user_input(obj.attrs["crs"])
    except KeyError:
        pass

    return None


def read_transform_auto(
    obj: Union[xarray.Dataset, xarray.DataArray],
    **kwargs,
) -> Optional[Affine]:
    """
    Auto-detect and read transform by trying convention readers.

    If a convention is set globally via set_options(), that convention
    is tried first for better performance. Then other conventions are
    tried as fallback.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read transform from
    **kwargs
        Convention-specific parameters (e.g., grid_mapping for CF)

    Returns
    -------
    affine.Affine or None
        Transform object, or None if not found in any convention
    """
    # Try the configured convention first (if set)
    configured_convention = get_option(CONVENTION)
    if configured_convention is not None:
        result = _CONVENTION_MODULES[configured_convention].read_transform(
            obj, **kwargs
        )
        if result is not None:
            return result

    # Try all other conventions
    for conv_enum, convention in _CONVENTION_MODULES.items():
        if conv_enum == configured_convention:
            continue  # Already tried this one
        result = convention.read_transform(obj, **kwargs)
        if result is not None:
            return result

    # Legacy fallback: look in attrs for 'transform' (not part of any convention)
    try:
        return Affine(*obj.attrs["transform"][:6])
    except KeyError:
        pass

    return None


def read_spatial_dimensions_auto(
    obj: Union[xarray.Dataset, xarray.DataArray],
) -> Optional[tuple[str, str]]:
    """
    Auto-detect and read spatial dimensions by trying convention readers.

    If a convention is set globally via set_options(), that convention
    is tried first for better performance. Then other conventions are
    tried as fallback.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read spatial dimensions from

    Returns
    -------
    tuple of (y_dim, x_dim) or None
        Tuple of dimension names, or None if not found in any convention
    """
    # Try the configured convention first (if set)
    configured_convention = get_option(CONVENTION)
    if configured_convention is not None:
        result = _CONVENTION_MODULES[configured_convention].read_spatial_dimensions(obj)
        if result is not None:
            return result

    # Try all other conventions
    for conv_enum, convention in _CONVENTION_MODULES.items():
        if conv_enum == configured_convention:
            continue  # Already tried this one
        result = convention.read_spatial_dimensions(obj)
        if result is not None:
            return result

    return None
