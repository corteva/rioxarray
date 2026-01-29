"""Convention modules for rioxarray.

This module defines the common interface for convention implementations
and provides helpers for selecting conventions.
"""

from typing import List, Optional, Protocol, Tuple, Union

import rasterio.crs
import xarray
from affine import Affine

from rioxarray._convention import cf


class ConventionReader(Protocol):
    """Protocol defining the interface for reading geospatial metadata from conventions."""

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


# List of convention modules in order of priority for reading
# CF convention is tried first
_READER_MODULES: List[ConventionReader] = [cf]


def read_crs_auto(
    obj: Union[xarray.Dataset, xarray.DataArray],
    grid_mapping: Optional[str] = None,
) -> Optional[rasterio.crs.CRS]:
    """
    Auto-detect and read CRS by trying all convention readers.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read CRS from
    grid_mapping : str, optional
        Name of the grid_mapping coordinate (passed to CF reader)

    Returns
    -------
    rasterio.crs.CRS or None
        CRS object, or None if not found in any convention
    """
    for reader in _READER_MODULES:
        result = reader.read_crs(obj, grid_mapping=grid_mapping)
        if result is not None:
            return result
    return None


def read_transform_auto(
    obj: Union[xarray.Dataset, xarray.DataArray],
    grid_mapping: Optional[str] = None,
) -> Optional[Affine]:
    """
    Auto-detect and read transform by trying all convention readers.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read transform from
    grid_mapping : str, optional
        Name of the grid_mapping coordinate (passed to CF reader)

    Returns
    -------
    affine.Affine or None
        Transform object, or None if not found in any convention
    """
    for reader in _READER_MODULES:
        result = reader.read_transform(obj, grid_mapping=grid_mapping)
        if result is not None:
            return result
    return None


def read_spatial_dimensions_auto(
    obj: Union[xarray.Dataset, xarray.DataArray],
) -> Optional[Tuple[str, str]]:
    """
    Auto-detect and read spatial dimensions by trying all convention readers.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read spatial dimensions from

    Returns
    -------
    tuple of (y_dim, x_dim) or None
        Tuple of dimension names, or None if not found in any convention
    """
    for reader in _READER_MODULES:
        result = reader.read_spatial_dimensions(obj)
        if result is not None:
            return result
    return None
