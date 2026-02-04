"""
Zarr spatial and proj convention support for rioxarray.

This module provides functions for reading geospatial metadata according to:
- Zarr spatial convention: https://github.com/zarr-conventions/spatial
- Zarr geo-proj convention: https://github.com/zarr-experimental/geo-proj
"""
import json
from typing import Optional, Union

import rasterio.crs
import xarray
from affine import Affine

from rioxarray.crs import crs_from_user_input

# Convention identifiers
PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj:",
    "description": "Coordinate reference system information for geospatial data",
}

SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial:",
    "description": "Spatial coordinate information",
}


def has_convention_declared(attrs: dict, convention_name: str) -> bool:
    """
    Check if a specific convention is declared in zarr_conventions.

    Parameters
    ----------
    attrs : dict
        Attributes dictionary to check
    convention_name : str
        Name of convention to check for (e.g., "proj:" or "spatial:")

    Returns
    -------
    bool
        True if convention is declared
    """
    zarr_conventions = attrs.get("zarr_conventions", [])
    if not isinstance(zarr_conventions, list):
        return False

    for convention in zarr_conventions:
        if isinstance(convention, dict) and convention.get("name") == convention_name:
            return True
    return False


def get_declared_conventions(attrs: dict) -> set:
    """
    Get set of declared convention names from attrs.

    Parameters
    ----------
    attrs : dict
        Attributes dictionary to check

    Returns
    -------
    set
        Set of declared convention names (e.g., {"proj:", "spatial:"})
    """
    zarr_conventions = attrs.get("zarr_conventions", [])
    if not isinstance(zarr_conventions, list):
        return set()

    declared = set()
    for convention in zarr_conventions:
        if isinstance(convention, dict) and "name" in convention:
            declared.add(convention["name"])

    return declared


# ============================================================================
# Parsing utilities
# ============================================================================


def parse_spatial_transform(
    spatial_transform: Union[list, tuple],
) -> Optional[Affine]:
    """
    Convert spatial:transform array to Affine object.

    Parameters
    ----------
    spatial_transform : list or tuple
        Transform as [a, b, c, d, e, f] array

    Returns
    -------
    affine.Affine or None
        Affine transform object, or None if invalid
    """
    if not isinstance(spatial_transform, (list, tuple)):
        return None
    if len(spatial_transform) != 6:
        return None
    try:
        return Affine(*spatial_transform)
    except (TypeError, ValueError):
        return None


# ============================================================================
# Internal parsing helpers
# ============================================================================


def _parse_crs_from_attrs(
    attrs: dict, convention_check: bool = True
) -> Optional[rasterio.crs.CRS]:
    """
    Parse CRS from proj: attributes with fallback priority.

    Parameters
    ----------
    attrs : dict
        Attributes dictionary to parse from
    convention_check : bool, default True
        Whether to check for convention declaration

    Returns
    -------
    rasterio.crs.CRS or None
        Parsed CRS object, or None if not found
    """
    if convention_check and not has_convention_declared(attrs, "proj:"):
        return None

    # Try proj attributes in priority order: wkt2, code, projjson
    for proj_attr in ("proj:wkt2", "proj:code", "proj:projjson"):
        try:
            proj_value = attrs.get(proj_attr)
            if proj_value is not None:
                # Handle projjson dict by converting to JSON string
                if isinstance(proj_value, dict):
                    proj_value = json.dumps(proj_value)
                parsed_crs = crs_from_user_input(proj_value)
                if parsed_crs is not None:
                    return parsed_crs
        except (KeyError, TypeError, ValueError):
            pass
    return None


def _parse_transform_from_attrs(
    attrs: dict, convention_check: bool = True
) -> Optional[Affine]:
    """
    Parse transform from spatial: attributes.

    Parameters
    ----------
    attrs : dict
        Attributes dictionary to parse from
    convention_check : bool, default True
        Whether to check for convention declaration

    Returns
    -------
    affine.Affine or None
        Parsed transform object, or None if not found
    """
    if convention_check and not has_convention_declared(attrs, "spatial:"):
        return None

    try:
        spatial_transform = attrs.get("spatial:transform")
        if spatial_transform is not None:
            return parse_spatial_transform(spatial_transform)
    except (KeyError, TypeError, ValueError):
        pass
    return None


# ============================================================================
# ZarrConvention class implementing ConventionProtocol
# ============================================================================


class ZarrConvention:
    """Zarr convention class implementing ConventionProtocol."""

    @classmethod
    def read_crs(
        cls, obj: Union[xarray.Dataset, xarray.DataArray]
    ) -> Optional[rasterio.crs.CRS]:
        """
        Read CRS from Zarr proj: convention.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to read CRS from

        Returns
        -------
        rasterio.crs.CRS or None
            CRS object, or None if not found
        """
        return _parse_crs_from_attrs(obj.attrs)

    @classmethod
    def read_transform(
        cls, obj: Union[xarray.Dataset, xarray.DataArray]
    ) -> Optional[Affine]:
        """
        Read transform from Zarr spatial: convention.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to read transform from

        Returns
        -------
        affine.Affine or None
            Transform object, or None if not found
        """
        return _parse_transform_from_attrs(obj.attrs)

    @classmethod
    def read_spatial_dimensions(
        cls, obj: Union[xarray.Dataset, xarray.DataArray]
    ) -> Optional[tuple[str, str]]:
        """
        Read spatial dimensions from Zarr spatial: convention.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to read spatial dimensions from

        Returns
        -------
        tuple of (y_dim, x_dim) or None
            Tuple of dimension names, or None if not found
        """
        # Only interpret spatial:* attributes if convention is declared
        if not has_convention_declared(obj.attrs, "spatial:"):
            return None

        try:
            spatial_dims = obj.attrs.get("spatial:dimensions")
            if spatial_dims is not None and len(spatial_dims) >= 2:
                # spatial:dimensions format is ["y", "x"] or similar
                y_dim_name, x_dim_name = spatial_dims[-2:]  # Take last two
                if y_dim_name in obj.dims and x_dim_name in obj.dims:
                    return y_dim_name, x_dim_name
        except (KeyError, TypeError, ValueError):
            pass

        return None

    @classmethod
    def write_crs(
        cls,
        obj: Union[xarray.Dataset, xarray.DataArray],
        crs: rasterio.crs.CRS,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Write CRS using Zarr conventions.

        Note: Writing support will be implemented in a future PR.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to write CRS to
        crs : rasterio.crs.CRS
            CRS to write
        **kwargs
            Additional convention-specific parameters

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Object with CRS written

        Raises
        ------
        NotImplementedError
            Zarr write support is not yet implemented
        """
        raise NotImplementedError(
            "Zarr CRS writing is not yet implemented. "
            "Use Convention.CF for writing or wait for a future release."
        )

    @classmethod
    def write_transform(
        cls,
        obj: Union[xarray.Dataset, xarray.DataArray],
        *,
        transform: Affine,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Write transform using Zarr conventions.

        Note: Writing support will be implemented in a future PR.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to write transform to
        transform : affine.Affine
            Transform to write
        **kwargs
            Additional convention-specific parameters

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Object with transform written

        Raises
        ------
        NotImplementedError
            Zarr write support is not yet implemented
        """
        raise NotImplementedError(
            "Zarr transform writing is not yet implemented. "
            "Use Convention.CF for writing or wait for a future release."
        )
