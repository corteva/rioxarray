"""
Helper functions for determining spatial attributes.
"""
import copy
import math
import warnings
from collections.abc import Hashable
from typing import Any, Iterable, Optional, Union

import numpy
import rasterio.mask
import rasterio.warp
import rasterio.windows
import xarray
from affine import Affine
from rasterio.control import GroundControlPoint
from rasterio.features import geometry_mask

from rioxarray.exceptions import MissingSpatialDimensionError

FILL_VALUE_NAMES = ("_FillValue", "missing_value", "fill_value", "nodata")
UNWANTED_RIO_ATTRS = ("nodatavals", "is_tiled", "res")
DEFAULT_GRID_MAP = "spatial_ref"

# DTYPE TO NODATA MAP
# Based on: https://github.com/OSGeo/gdal/blob/v3.12.1/swig/python/gdal-utils/osgeo_utils/gdal_calc.py#L49-L66
# And: https://github.com/rasterio/rasterio/blob/1.5.0/rasterio/dtypes.py#L91-L103
_NODATA_DTYPE_MAP = {
    1: 255,  # GDT_Byte
    2: 65535,  # GDT_UInt16
    3: -32768,  # GDT_Int16
    4: 4294967295,  # GDT_UInt32
    5: -2147483648,  # GDT_Int32
    6: numpy.nan,  # GDT_Float32
    7: numpy.nan,  # GDT_Float64
    8: None,  # GDT_CInt16
    9: None,  # GDT_CInt32
    10: numpy.nan,  # GDT_CFloat32
    11: numpy.nan,  # GDT_CFloat64
    12: 18446744073709551615,  # GDT_UInt64
    13: -9223372036854775808,  # GDT_Int64
    14: -128,  # GDT_Int8
    15: numpy.nan,  # GDT_Float16
    16: numpy.nan,  # GDT_CFloat16
}


def _affine_has_rotation(affine: Affine) -> bool:
    """
    Determine if the affine has rotation.

    Parameters
    ----------
    affine: :obj:`affine.Affine`
        The affine of the grid.

    Returns
    -------
    bool
    """
    return affine.b == affine.d != 0


def _resolution(affine: Affine) -> tuple[float, float]:
    """
    Determine if the resolution of the affine.
    If it has rotation, the sign of the resolution is lost.

    Based on: https://github.com/mapbox/rasterio/blob/6185a4e4ad72b5669066d2d5004bf46d94a6d298/rasterio/_base.pyx#L943-L951

    Parameters
    ----------
    affine: :obj:`affine.Affine`
        The affine of the grid.


    Returns
    --------
    x_resolution: float
        The X resolution of the affine.
    y_resolution: float
        The Y resolution of the affine.
    """
    if not _affine_has_rotation(affine):
        return affine.a, affine.e
    return (
        math.sqrt(affine.a**2 + affine.d**2),
        math.sqrt(affine.b**2 + affine.e**2),
    )


def affine_to_coords(
    affine: Affine, width: int, height: int, *, x_dim: str = "x", y_dim: str = "y"
) -> dict[str, numpy.ndarray]:
    """Generate 1d pixel centered coordinates from affine.

    Based on code from the xarray rasterio backend.

    Parameters
    ----------
    affine: :obj:`affine.Affine`
        The affine of the grid.
    width: int
        The width of the grid.
    height: int
        The height of the grid.
    x_dim: str, optional
        The name of the X dimension. Default is 'x'.
    y_dim: str, optional
        The name of the Y dimension. Default is 'y'.

    Returns
    -------
    dict: x and y coordinate arrays.

    """
    transform = affine * affine.translation(0.5, 0.5)
    if affine.is_rectilinear and not _affine_has_rotation(affine):
        x_coords, _ = transform * (numpy.arange(width), numpy.zeros(width))
        _, y_coords = transform * (numpy.zeros(height), numpy.arange(height))
    else:
        x_coords, y_coords = transform * numpy.meshgrid(
            numpy.arange(width),
            numpy.arange(height),
        )
    return {y_dim: y_coords, x_dim: x_coords}


def _generate_spatial_coords(
    *, affine: Affine, width: int, height: int
) -> dict[Hashable, Any]:
    """get spatial coords in new transform"""
    new_spatial_coords = affine_to_coords(affine, width, height)
    if new_spatial_coords["x"].ndim == 1:
        return {
            "x": xarray.IndexVariable("x", new_spatial_coords["x"]),
            "y": xarray.IndexVariable("y", new_spatial_coords["y"]),
        }
    return {
        "xc": (("y", "x"), new_spatial_coords["x"]),
        "yc": (("y", "x"), new_spatial_coords["y"]),
    }


def _get_nonspatial_coords(
    src_data_array: Union[xarray.DataArray, xarray.Dataset],
) -> dict[Hashable, Union[xarray.Variable, xarray.IndexVariable]]:
    coords: dict[Hashable, Union[xarray.Variable, xarray.IndexVariable]] = {}
    for coord in set(src_data_array.coords) - {
        src_data_array.rio.x_dim,
        src_data_array.rio.y_dim,
        DEFAULT_GRID_MAP,
    }:
        # skip 2D spatial coords
        if (
            src_data_array.rio.x_dim in src_data_array[coord].dims
            and src_data_array.rio.y_dim in src_data_array[coord].dims
        ):
            continue
        if src_data_array[coord].ndim == 1:
            coords[coord] = xarray.IndexVariable(
                src_data_array[coord].dims,
                src_data_array[coord].values,
                src_data_array[coord].attrs,
            )
        else:
            coords[coord] = xarray.Variable(
                src_data_array[coord].dims,
                src_data_array[coord].values,
                src_data_array[coord].attrs,
            )
    return coords


def _make_coords(
    *,
    src_data_array: Union[xarray.DataArray, xarray.Dataset],
    dst_affine: Affine,
    dst_width: int,
    dst_height: int,
    force_generate: bool = False,
) -> dict[Hashable, Any]:
    """Generate the coordinates of the new projected `xarray.DataArray`"""
    coords = _get_nonspatial_coords(src_data_array)
    if (
        force_generate
        or (
            src_data_array.rio.x_dim in src_data_array.coords
            and src_data_array.rio.y_dim in src_data_array.coords
        )
        or ("xc" in src_data_array.coords and "yc" in src_data_array.coords)
    ):
        new_coords = _generate_spatial_coords(
            affine=dst_affine, width=dst_width, height=dst_height
        )
        new_coords.update(coords)
        return new_coords
    return coords


def _get_data_var_message(obj: Union[xarray.DataArray, xarray.Dataset]) -> str:
    """
    Get message for named data variables.
    """
    try:
        return f" Data variable: {obj.name}" if obj.name else ""
    except AttributeError:
        return ""


def _get_spatial_dims(
    obj: Union[xarray.Dataset, xarray.DataArray], *, var: Union[Any, Hashable]
) -> tuple[str, str]:
    """
    Retrieve the spatial dimensions of the dataset
    """
    try:
        return obj[var].rio.x_dim, obj[var].rio.y_dim
    except MissingSpatialDimensionError as err:
        try:
            obj[var].rio.set_spatial_dims(
                x_dim=obj.rio.x_dim, y_dim=obj.rio.y_dim, inplace=True
            )
            return obj.rio.x_dim, obj.rio.y_dim
        except MissingSpatialDimensionError:
            raise err from None


def _has_spatial_dims(
    obj: Union[xarray.Dataset, xarray.DataArray], *, var: Union[Any, Hashable]
) -> bool:
    """
    Check to see if the variable in the Dataset has spatial dimensions
    """
    try:
        # pylint: disable=pointless-statement
        _get_spatial_dims(obj, var=var)
    except MissingSpatialDimensionError:
        return False
    return True


def _order_bounds(
    *,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    resolution_x: float,
    resolution_y: float,
) -> tuple[float, float, float, float]:
    """
    Make sure that the bounds are in the correct order
    """
    if resolution_y < 0:
        top = maxy
        bottom = miny
    else:
        top = miny
        bottom = maxy
    if resolution_x < 0:
        left = maxx
        right = minx
    else:
        left = minx
        right = maxx

    return left, bottom, right, top


def _convert_gcps_to_geojson(
    gcps: Iterable[GroundControlPoint],
) -> dict:
    """
    Convert GCPs to geojson.

    Parameters
    ----------
    gcps: The list of GroundControlPoint instances.

    Returns
    -------
    A FeatureCollection dict.
    """

    def _gcp_coordinates(gcp):
        if gcp.z is None:
            return [gcp.x, gcp.y]
        return [gcp.x, gcp.y, gcp.z]

    features = [
        {
            "type": "Feature",
            "properties": {
                "id": gcp.id,
                "info": gcp.info,
                "row": gcp.row,
                "col": gcp.col,
            },
            "geometry": {"type": "Point", "coordinates": _gcp_coordinates(gcp)},
        }
        for gcp in gcps
    ]
    return {"type": "FeatureCollection", "features": features}


def _convert_str_to_resampling(name: str) -> rasterio.warp.Resampling:
    """
    Convert from string to rasterio.warp.Resampling enum, raises ValueError on bad input.

    Parameters
    ----------
    name: str
        The string to convert.

    Returns
    -------
    :obj:`rasterio.warp.Resampling`
    """
    try:
        return getattr(rasterio.warp.Resampling, name.lower())
    except AttributeError:
        raise ValueError(f"Bad resampling parameter: {name}") from None


def _generate_attrs(
    *, src_data_array: xarray.DataArray, dst_nodata: Optional[float]
) -> dict[str, Any]:
    # add original attributes
    new_attrs = copy.deepcopy(src_data_array.attrs)
    # remove all nodata information
    for unwanted_attr in FILL_VALUE_NAMES + UNWANTED_RIO_ATTRS:
        new_attrs.pop(unwanted_attr, None)

    # add nodata information
    fill_value = (
        src_data_array.rio.nodata
        if src_data_array.rio.nodata is not None
        else dst_nodata
    )
    if src_data_array.rio.encoded_nodata is None and fill_value is not None:
        new_attrs["_FillValue"] = fill_value

    return new_attrs


def _add_attrs_proj(
    *, new_data_array: xarray.DataArray, src_data_array: xarray.DataArray
) -> xarray.DataArray:
    """Make sure attributes and projection correct"""
    # make sure dimension information is preserved
    if new_data_array.rio._x_dim is None:
        new_data_array.rio._x_dim = src_data_array.rio.x_dim
    if new_data_array.rio._y_dim is None:
        new_data_array.rio._y_dim = src_data_array.rio.y_dim

    # make sure attributes preserved
    new_attrs = _generate_attrs(src_data_array=src_data_array, dst_nodata=None)
    # remove fill value if it already exists in the encoding
    # this is for data arrays pulling the encoding from a
    # source data array instead of being generated anew.
    if "_FillValue" in new_data_array.encoding:
        new_attrs.pop("_FillValue", None)

    new_data_array.rio.set_attrs(new_attrs, inplace=True)

    # make sure projection added
    new_data_array.rio.write_grid_mapping(src_data_array.rio.grid_mapping, inplace=True)
    new_data_array.rio.write_crs(src_data_array.rio.crs, inplace=True)
    new_data_array.rio.write_coordinate_system(inplace=True)
    new_data_array.rio.write_transform(inplace=True)
    # make sure encoding added
    new_data_array.encoding = src_data_array.encoding.copy()
    return new_data_array


def _make_dst_affine(
    *,
    src_data_array: xarray.DataArray,
    src_crs: rasterio.crs.CRS,
    dst_crs: rasterio.crs.CRS,
    dst_resolution: Optional[Union[float, tuple[float, float]]] = None,
    dst_shape: Optional[tuple[float, float]] = None,
    **kwargs,
):
    """Determine the affine of the new projected `xarray.DataArray`"""
    src_bounds = ()
    if (
        "gcps" not in kwargs
        and "rpcs" not in kwargs
        and "src_geoloc_array" not in kwargs
    ):
        src_bounds = src_data_array.rio.bounds()
    src_height, src_width = src_data_array.rio.shape
    dst_height, dst_width = dst_shape if dst_shape is not None else (None, None)
    # pylint: disable=isinstance-second-argument-not-valid-type
    if isinstance(dst_resolution, Iterable):
        dst_resolution = tuple(abs(res_val) for res_val in dst_resolution)  # type: ignore
    elif dst_resolution is not None:
        dst_resolution = abs(dst_resolution)  # type: ignore

    for key, value in (
        ("resolution", dst_resolution),
        ("dst_height", dst_height),
        ("dst_width", dst_width),
    ):
        if value is not None:
            kwargs[key] = value
    dst_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        src_width,
        src_height,
        *src_bounds,
        **kwargs,
    )
    return dst_affine, dst_width, dst_height


def _clip_from_disk(
    xds: xarray.DataArray,
    *,
    geometries: Iterable,
    all_touched: bool,
    drop: bool,
    invert: bool,
) -> Optional[xarray.DataArray]:
    """
    clip from disk if the file object is available
    """
    try:
        rio_dataset = xds.rio._manager.acquire()
    except AttributeError:
        warnings.warn("File object not available, clipping in-memory.")
        return None

    out_image, out_transform = rasterio.mask.mask(
        rio_dataset,
        geometries,
        all_touched=all_touched,
        invert=invert,
        crop=drop,
    )
    if xds.rio.encoded_nodata is not None and not numpy.isnan(xds.rio.encoded_nodata):
        out_image = out_image.astype(numpy.float64)
        out_image[out_image == xds.rio.encoded_nodata] = numpy.nan

    height, width = out_image.shape[-2:]
    cropped_ds = xarray.DataArray(
        name=xds.name,
        data=out_image,
        coords=_make_coords(
            src_data_array=xds,
            dst_affine=out_transform,
            dst_width=width,
            dst_height=height,
        ),
        dims=xds.dims,
        attrs=xds.attrs,
    )
    cropped_ds.encoding = xds.encoding
    return cropped_ds


def _clip_xarray(
    xds: xarray.DataArray,
    *,
    geometries: Iterable,
    all_touched: bool,
    drop: bool,
    invert: bool,
) -> xarray.DataArray:
    """
    clip the xarray DataArray
    """
    clip_mask_arr = geometry_mask(
        geometries=geometries,
        out_shape=(int(xds.rio.height), int(xds.rio.width)),
        transform=xds.rio.transform(recalc=True),
        invert=not invert,
        all_touched=all_touched,
    )
    clip_mask_xray = xarray.DataArray(
        clip_mask_arr,
        dims=(xds.rio.y_dim, xds.rio.x_dim),
    )
    cropped_ds = xds.where(clip_mask_xray)
    if drop:
        cropped_ds.rio.set_spatial_dims(
            x_dim=xds.rio.x_dim, y_dim=xds.rio.y_dim, inplace=True
        )
        cropped_ds = cropped_ds.rio.isel_window(
            rasterio.windows.get_data_window(
                numpy.ma.masked_array(clip_mask_arr, ~clip_mask_arr)
            )
        )
    if xds.rio.nodata is not None and not numpy.isnan(xds.rio.nodata):
        cropped_ds = cropped_ds.fillna(xds.rio.nodata)

    return cropped_ds.astype(xds.dtype)
