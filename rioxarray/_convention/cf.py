"""
CF (Climate and Forecasts) convention support for rioxarray.

This module provides functions for reading and writing geospatial metadata according to
the CF conventions: https://github.com/cf-convention/cf-conventions
"""
from typing import Optional, Union

import pyproj
import rasterio.crs
import xarray
from affine import Affine

from rioxarray.crs import crs_from_user_input


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
    if grid_mapping is None:
        # Try to find grid_mapping attribute on data variables
        if hasattr(obj, "data_vars"):
            for data_var in obj.data_vars.values():
                if "grid_mapping" in data_var.attrs:
                    grid_mapping = data_var.attrs["grid_mapping"]
                    break
        elif hasattr(obj, "attrs") and "grid_mapping" in obj.attrs:
            grid_mapping = obj.attrs["grid_mapping"]

    if grid_mapping is None:
        # look in attrs for 'crs'
        try:
            return crs_from_user_input(obj.attrs["crs"])
        except KeyError:
            return None

    try:
        grid_mapping_coord = obj.coords[grid_mapping]
    except KeyError:
        return None

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

    return None


def read_transform(
    obj: Union[xarray.Dataset, xarray.DataArray], grid_mapping: Optional[str] = None
) -> Optional[Affine]:
    """
    Read transform from CF conventions.

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
    if grid_mapping is None:
        # Try to find grid_mapping attribute on data variables
        if hasattr(obj, "data_vars"):
            for data_var in obj.data_vars.values():
                if "grid_mapping" in data_var.attrs:
                    grid_mapping = data_var.attrs["grid_mapping"]
                    break
        elif hasattr(obj, "attrs") and "grid_mapping" in obj.attrs:
            grid_mapping = obj.attrs["grid_mapping"]

    if grid_mapping is not None:
        try:
            grid_mapping_coord = obj.coords[grid_mapping]
            geotransform = grid_mapping_coord.attrs.get("GeoTransform")
            if geotransform is not None:
                return _parse_geotransform(geotransform)
        except KeyError:
            pass

    # Look in dataset attributes for transform
    try:
        transform = obj.attrs["transform"]
        if hasattr(transform, "__iter__") and len(transform) == 6:
            return Affine(*transform)
        return transform
    except KeyError:
        pass

    return None


def write_crs(
    obj: Union[xarray.Dataset, xarray.DataArray],
    input_crs: Optional[object] = None,
    grid_mapping_name: str = "spatial_ref",
    inplace: bool = True,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write CRS using CF conventions.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write CRS to
    input_crs : object, optional
        CRS to write. Can be anything accepted by rasterio.crs.CRS.from_user_input
    grid_mapping_name : str, default "spatial_ref"
        Name for the grid_mapping coordinate
    inplace : bool, default True
        If True, modify object in place

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with CRS written
    """
    from rioxarray._options import EXPORT_GRID_MAPPING, get_option

    if input_crs is None:
        return obj

    crs = crs_from_user_input(input_crs)
    if crs is None:
        return obj

    obj_out = obj if inplace else obj.copy(deep=True)

    # Create grid_mapping coordinate if it doesn't exist
    if grid_mapping_name not in obj_out.coords:
        obj_out = obj_out.assign_coords({grid_mapping_name: xarray.DataArray(0)})

    # Write WKT for compatibility
    obj_out.coords[grid_mapping_name].attrs["spatial_ref"] = crs.to_wkt()
    obj_out.coords[grid_mapping_name].attrs["crs_wkt"] = crs.to_wkt()

    # Write CF attributes if enabled
    if get_option(EXPORT_GRID_MAPPING):
        try:
            # Convert to pyproj.CRS for CF support
            pyproj_crs = pyproj.CRS.from_user_input(crs)
            cf_dict = pyproj_crs.to_cf()
            obj_out.coords[grid_mapping_name].attrs.update(cf_dict)
        except (pyproj.exceptions.CRSError, AttributeError):
            pass

    # Set grid_mapping attribute on data variables
    if hasattr(obj_out, "data_vars"):
        for data_var_name in obj_out.data_vars:
            obj_out[data_var_name].attrs["grid_mapping"] = grid_mapping_name
    else:
        obj_out.attrs["grid_mapping"] = grid_mapping_name

    return obj_out


def write_transform(
    obj: Union[xarray.Dataset, xarray.DataArray],
    transform: Optional[Affine] = None,
    grid_mapping_name: str = "spatial_ref",
    inplace: bool = True,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write transform using CF conventions.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write transform to
    transform : affine.Affine, optional
        Transform to write
    grid_mapping_name : str, default "spatial_ref"
        Name for the grid_mapping coordinate
    inplace : bool, default True
        If True, modify object in place

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with transform written
    """
    if transform is None:
        return obj

    obj_out = obj if inplace else obj.copy(deep=True)

    # Create grid_mapping coordinate if it doesn't exist
    if grid_mapping_name not in obj_out.coords:
        obj_out = obj_out.assign_coords({grid_mapping_name: xarray.DataArray(0)})

    # Write GeoTransform as GDAL format string
    geotransform_str = f"{transform.a} {transform.b} {transform.c} {transform.d} {transform.e} {transform.f}"
    obj_out.coords[grid_mapping_name].attrs["GeoTransform"] = geotransform_str

    # Also write as dataset attribute for backward compatibility
    obj_out.attrs["transform"] = tuple(transform)

    return obj_out


def _parse_geotransform(geotransform: Union[str, list, tuple]) -> Optional[Affine]:
    """Parse GeoTransform from CF conventions."""
    if isinstance(geotransform, str):
        try:
            vals = [float(val) for val in geotransform.split()]
            if len(vals) == 6:
                return Affine(*vals)
        except (ValueError, TypeError):
            pass
    elif hasattr(geotransform, "__iter__") and len(geotransform) == 6:
        try:
            return Affine(*geotransform)
        except (ValueError, TypeError):
            pass
    return None
