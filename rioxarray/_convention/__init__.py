"""Convention modules for rioxarray.

This module defines the common interface for convention implementations
and provides helpers for selecting conventions.
"""

from typing import Dict, Optional, Protocol, Tuple, Union

import rasterio.crs
import xarray
from affine import Affine

from rioxarray._convention import cf
from rioxarray._options import CONVENTION, get_option
from rioxarray.crs import crs_from_user_input
from rioxarray.enum import Convention


class ConventionProtocol(Protocol):
    """Protocol defining the interface for convention modules."""

    @staticmethod
    def read_crs(
        obj: Union[xarray.Dataset, xarray.DataArray], **kwargs
    ) -> Optional[rasterio.crs.CRS]:
        """Read CRS from the object using this convention."""
        ...

    @staticmethod
    def read_transform(
        obj: Union[xarray.Dataset, xarray.DataArray], **kwargs
    ) -> Optional[Affine]:
        """Read transform from the object using this convention."""
        ...

    @staticmethod
    def read_spatial_dimensions(
        obj: Union[xarray.Dataset, xarray.DataArray],
    ) -> Optional[Tuple[str, str]]:
        """Read spatial dimensions (y_dim, x_dim) from the object using this convention."""
        ...

    @staticmethod
    def write_crs(
        obj: Union[xarray.Dataset, xarray.DataArray],
        crs: rasterio.crs.CRS,
        inplace: bool = True,
        **kwargs,
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """Write CRS to the object using this convention."""
        ...

    @staticmethod
    def write_transform(
        obj: Union[xarray.Dataset, xarray.DataArray],
        transform: Affine,
        inplace: bool = True,
        **kwargs,
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """Write transform to the object using this convention."""
        ...


# Convention modules mapped by Convention enum
_CONVENTION_MODULES: Dict[Convention, ConventionProtocol] = {Convention.CF: cf}


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
) -> Optional[Tuple[str, str]]:
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
