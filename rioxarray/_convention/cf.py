"""
CF (Climate and Forecasts) convention support for rioxarray.

This module provides functions for reading and writing geospatial metadata according to
the CF conventions: https://github.com/cf-convention/cf-conventions
"""
from typing import Optional, Tuple, Union

import numpy
import pyproj
import rasterio.crs
import xarray
from affine import Affine

from rioxarray._options import EXPORT_GRID_MAPPING, get_option
from rioxarray._spatial_utils import _get_spatial_dims, _has_spatial_dims
from rioxarray.crs import crs_from_user_input
from rioxarray.exceptions import MissingSpatialDimensionError


def _find_grid_mapping(
    obj: Union[xarray.Dataset, xarray.DataArray],
    grid_mapping: Optional[str] = None,
) -> Optional[str]:
    """
    Find the grid_mapping coordinate name.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to search for grid_mapping
    grid_mapping : str, optional
        Explicit grid_mapping name to use

    Returns
    -------
    str or None
        The grid_mapping name, or None if not found
    """
    if grid_mapping is not None:
        return grid_mapping

    # Try to find grid_mapping attribute on data variables
    if hasattr(obj, "data_vars"):
        for data_var in obj.data_vars.values():
            if "grid_mapping" in data_var.attrs:
                return data_var.attrs["grid_mapping"]
            if "grid_mapping" in data_var.encoding:
                return data_var.encoding["grid_mapping"]

    if hasattr(obj, "attrs") and "grid_mapping" in obj.attrs:
        return obj.attrs["grid_mapping"]

    if hasattr(obj, "encoding") and "grid_mapping" in obj.encoding:
        return obj.encoding["grid_mapping"]

    return None


def read_crs(
    obj: Union[xarray.Dataset, xarray.DataArray], grid_mapping: Optional[str] = None
) -> Optional[rasterio.crs.CRS]:
    """
    Read CRS from CF conventions.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read CRS from
    grid_mapping : str, optional
        Name of the grid_mapping coordinate variable

    Returns
    -------
    rasterio.crs.CRS or None
        CRS object, or None if not found
    """
    grid_mapping = _find_grid_mapping(obj, grid_mapping)

    if grid_mapping is not None:
        try:
            grid_mapping_coord = obj.coords[grid_mapping]

            # Look in wkt attributes first for performance
            for crs_attr in ("spatial_ref", "crs_wkt"):
                try:
                    return crs_from_user_input(grid_mapping_coord.attrs[crs_attr])
                except KeyError:
                    pass

            # Look in grid_mapping CF attributes
            try:
                return pyproj.CRS.from_cf(grid_mapping_coord.attrs)
            except (KeyError, pyproj.exceptions.CRSError):
                pass
        except KeyError:
            # grid_mapping coordinate doesn't exist
            pass

    return None


def read_transform(
    obj: Union[xarray.Dataset, xarray.DataArray], grid_mapping: Optional[str] = None
) -> Optional[Affine]:
    """
    Read transform from CF conventions (GeoTransform attribute).

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read transform from
    grid_mapping : str, optional
        Name of the grid_mapping coordinate variable

    Returns
    -------
    affine.Affine or None
        Transform object, or None if not found
    """
    grid_mapping = _find_grid_mapping(obj, grid_mapping)

    if grid_mapping is not None:
        try:
            transform = numpy.fromstring(
                obj.coords[grid_mapping].attrs["GeoTransform"], sep=" "
            )
            # Calling .tolist() to assure the arguments are Python float and JSON serializable
            return Affine.from_gdal(*transform.tolist())
        except KeyError:
            pass

    return None


def read_spatial_dimensions(
    obj: Union[xarray.Dataset, xarray.DataArray],
) -> Optional[Tuple[str, str]]:
    """
    Read spatial dimensions from CF conventions.

    This function detects spatial dimensions based on:
    1. Standard dimension names ('x'/'y', 'longitude'/'latitude')
    2. CF coordinate attributes ('axis', 'standard_name')

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read spatial dimensions from

    Returns
    -------
    tuple of (y_dim, x_dim) or None
        Tuple of dimension names, or None if not found
    """
    x_dim = None
    y_dim = None

    # Check standard dimension names
    if "x" in obj.dims and "y" in obj.dims:
        return "y", "x"
    elif "longitude" in obj.dims and "latitude" in obj.dims:
        return "latitude", "longitude"

    # Look for coordinates with CF attributes
    for coord in obj.coords:
        # Make sure to only look in 1D coordinates
        # that has the same dimension name as the coordinate
        if obj.coords[coord].dims != (coord,):
            continue
        if (obj.coords[coord].attrs.get("axis", "").upper() == "X") or (
            obj.coords[coord].attrs.get("standard_name", "").lower()
            in ("longitude", "projection_x_coordinate")
        ):
            x_dim = coord
        elif (obj.coords[coord].attrs.get("axis", "").upper() == "Y") or (
            obj.coords[coord].attrs.get("standard_name", "").lower()
            in ("latitude", "projection_y_coordinate")
        ):
            y_dim = coord

    if x_dim is not None and y_dim is not None:
        return y_dim, x_dim

    return None


def write_crs(
    obj: Union[xarray.Dataset, xarray.DataArray],
    crs: rasterio.crs.CRS,
    inplace: bool = True,
    **kwargs,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write CRS using CF conventions.

    This also writes the grid_mapping attribute to encoding for CF compliance.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write CRS to
    crs : rasterio.crs.CRS
        CRS to write
    inplace : bool, default True
        If True, modify object in place
    **kwargs
        grid_mapping_name : str
            Name of the grid_mapping coordinate (required for CF)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with CRS written
    """
    grid_mapping_name = kwargs.get("grid_mapping_name")
    if grid_mapping_name is None:
        raise ValueError("grid_mapping_name is required for CF convention")

    obj_out = obj if inplace else obj.copy(deep=True)

    # Get original transform before modifying
    transform = read_transform(obj)

    # Remove old grid mapping coordinate if exists
    try:
        del obj_out.coords[grid_mapping_name]
    except KeyError:
        pass

    # Add grid mapping coordinate
    obj_out.coords[grid_mapping_name] = xarray.Variable((), 0)
    grid_map_attrs = {}
    if get_option(EXPORT_GRID_MAPPING):
        try:
            grid_map_attrs = pyproj.CRS.from_user_input(crs).to_cf()
        except KeyError:
            pass

    # spatial_ref is for compatibility with GDAL
    crs_wkt = crs.to_wkt()
    grid_map_attrs["spatial_ref"] = crs_wkt
    grid_map_attrs["crs_wkt"] = crs_wkt
    if transform is not None:
        grid_map_attrs["GeoTransform"] = " ".join(
            [str(item) for item in transform.to_gdal()]
        )
    obj_out.coords[grid_mapping_name].attrs = grid_map_attrs

    # Write grid_mapping to encoding (CF specific)
    obj_out = _write_grid_mapping(obj_out, grid_mapping_name)

    return obj_out


def _write_grid_mapping(
    obj: Union[xarray.Dataset, xarray.DataArray],
    grid_mapping_name: str,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write the CF grid_mapping attribute to the encoding.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write grid_mapping to
    grid_mapping_name : str
        Name of the grid_mapping coordinate

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with grid_mapping written to encoding
    """
    if hasattr(obj, "data_vars"):
        for var in obj.data_vars:
            if not _has_spatial_dims(obj, var=var):
                continue
            try:
                x_dim, y_dim = _get_spatial_dims(obj, var=var)
            except MissingSpatialDimensionError:
                continue
            # remove grid_mapping from attributes if it exists
            # and update the grid_mapping in encoding
            new_attrs = dict(obj[var].attrs)
            new_attrs.pop("grid_mapping", None)
            obj[var].attrs = new_attrs
            obj[var].encoding["grid_mapping"] = grid_mapping_name
            obj[var].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)

    # remove grid_mapping from attributes if it exists
    # and update the grid_mapping in encoding
    new_attrs = dict(obj.attrs)
    new_attrs.pop("grid_mapping", None)
    obj.attrs = new_attrs
    obj.encoding["grid_mapping"] = grid_mapping_name

    return obj


def write_transform(
    obj: Union[xarray.Dataset, xarray.DataArray],
    transform: Affine,
    inplace: bool = True,
    **kwargs,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write transform using CF conventions (GeoTransform attribute).

    This also writes the grid_mapping attribute to encoding for CF compliance.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write transform to
    transform : affine.Affine
        Transform to write
    inplace : bool, default True
        If True, modify object in place
    **kwargs
        grid_mapping_name : str
            Name of the grid_mapping coordinate (required for CF)

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with transform written
    """
    grid_mapping_name = kwargs.get("grid_mapping_name")
    if grid_mapping_name is None:
        raise ValueError("grid_mapping_name is required for CF convention")

    obj_out = obj if inplace else obj.copy(deep=True)

    try:
        grid_map_attrs = obj_out.coords[grid_mapping_name].attrs.copy()
    except KeyError:
        obj_out.coords[grid_mapping_name] = xarray.Variable((), 0)
        grid_map_attrs = obj_out.coords[grid_mapping_name].attrs.copy()

    grid_map_attrs["GeoTransform"] = " ".join(
        [str(item) for item in transform.to_gdal()]
    )
    obj_out.coords[grid_mapping_name].attrs = grid_map_attrs

    # Write grid_mapping to encoding (CF specific)
    obj_out = _write_grid_mapping(obj_out, grid_mapping_name)

    return obj_out
