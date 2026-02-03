"""This module defines the common interface for convention implementations
and provides helpers for selecting conventions.
"""

from typing import Optional, Protocol, Union

import rasterio.crs
import xarray
from affine import Affine


class ConventionProtocol(Protocol):
    """Protocol defining the interface for convention modules."""

    @classmethod
    def read_crs(
        cls, obj: Union[xarray.Dataset, xarray.DataArray], **kwargs
    ) -> Optional[rasterio.crs.CRS]:
        """Read CRS from the object using this convention."""

    @classmethod
    def read_transform(
        cls, obj: Union[xarray.Dataset, xarray.DataArray], **kwargs
    ) -> Optional[Affine]:
        """Read transform from the object using this convention."""

    @classmethod
    def read_spatial_dimensions(
        cls,
        obj: Union[xarray.Dataset, xarray.DataArray],
    ) -> Optional[tuple[str, str]]:
        """Read spatial dimensions (y_dim, x_dim) from the object using this convention."""

    @classmethod
    def write_crs(
        cls,
        obj: Union[xarray.Dataset, xarray.DataArray],
        *,
        crs: rasterio.crs.CRS,
        **kwargs,
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """Write CRS to the object using this convention."""

    @classmethod
    def write_transform(
        cls,
        obj: Union[xarray.Dataset, xarray.DataArray],
        *,
        transform: Affine,
        **kwargs,
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """Write transform to the object using this convention."""
