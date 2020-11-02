# -- coding: utf-8 --
"""
This module is an extension for xarray to provide rasterio capabilities
to xarray datasets/dataarrays.

Credits: The `reproject` functionality was adopted from https://github.com/opendatacube/datacube-core  # noqa
Source file:
- https://github.com/opendatacube/datacube-core/blob/084c84d78cb6e1326c7fbbe79c5b5d0bef37c078/datacube/api/geo_xarray.py  # noqa
datacube is licensed under the Apache License, Version 2.0:
- https://github.com/opendatacube/datacube-core/blob/1d345f08a10a13c316f81100936b0ad8b1a374eb/LICENSE  # noqa

"""
import copy
import math
import warnings
from uuid import uuid4

import numpy as np
import pyproj
import rasterio.warp
import rasterio.windows
import xarray
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from scipy.interpolate import griddata

from rioxarray.crs import crs_to_wkt
from rioxarray.exceptions import (
    DimensionError,
    DimensionMissingCoordinateError,
    InvalidDimensionOrder,
    MissingCRS,
    NoDataInBounds,
    OneDimensionalRaster,
    RioXarrayError,
    TooManyDimensions,
)

FILL_VALUE_NAMES = ("_FillValue", "missing_value", "fill_value", "nodata")
UNWANTED_RIO_ATTRS = ("nodatavals", "crs", "is_tiled", "res")
DEFAULT_GRID_MAP = "spatial_ref"


def affine_to_coords(affine, width, height, x_dim="x", y_dim="y"):
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
    x_coords, _ = affine * (np.arange(width) + 0.5, np.zeros(width) + 0.5)
    _, y_coords = affine * (np.zeros(height) + 0.5, np.arange(height) + 0.5)
    return {y_dim: y_coords, x_dim: x_coords}


def _generate_attrs(src_data_array, dst_nodata):
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

    # add raster spatial information
    new_attrs["grid_mapping"] = src_data_array.rio.grid_mapping

    return new_attrs


def add_xy_grid_meta(coords, crs=None):
    raise RuntimeError(
        "add_xy_grid_meta has been removed. Use rio.write_coordinate_system instead.",
    )


def add_spatial_ref(in_ds, dst_crs, grid_mapping_name):
    raise RuntimeError("add_spatial_ref has been removed. Use rio.write_crs instead.")


def _add_attrs_proj(new_data_array, src_data_array):
    """Make sure attributes and projection correct"""
    # make sure dimension information is preserved
    if new_data_array.rio._x_dim is None:
        new_data_array.rio._x_dim = src_data_array.rio.x_dim
    if new_data_array.rio._y_dim is None:
        new_data_array.rio._y_dim = src_data_array.rio.y_dim

    # make sure attributes preserved
    new_attrs = _generate_attrs(src_data_array, None)
    # remove fill value if it already exists in the encoding
    # this is for data arrays pulling the encoding from a
    # source data array instead of being generated anew.
    if "_FillValue" in new_data_array.encoding:
        new_attrs.pop("_FillValue", None)

    new_data_array.rio.set_attrs(new_attrs, inplace=True)

    # make sure projection added
    new_data_array.rio.write_crs(src_data_array.rio.crs, inplace=True)
    new_data_array.rio.write_coordinate_system(inplace=True)
    new_data_array.rio.write_transform(inplace=True)
    # make sure encoding added
    new_data_array.encoding = src_data_array.encoding.copy()
    return new_data_array


def _warp_spatial_coords(data_array, affine, width, height):
    """get spatial coords in new transform"""
    new_spatial_coords = affine_to_coords(affine, width, height)
    return {
        "x": xarray.IndexVariable("x", new_spatial_coords["x"]),
        "y": xarray.IndexVariable("y", new_spatial_coords["y"]),
    }


def _get_nonspatial_coords(src_data_array):
    coords = {}
    for coord in set(src_data_array.coords) - {
        src_data_array.rio.x_dim,
        src_data_array.rio.y_dim,
        DEFAULT_GRID_MAP,
    }:
        if src_data_array[coord].dims:
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


def _make_coords(src_data_array, dst_affine, dst_width, dst_height):
    """Generate the coordinates of the new projected `xarray.DataArray`"""
    coords = _get_nonspatial_coords(src_data_array)
    new_coords = _warp_spatial_coords(src_data_array, dst_affine, dst_width, dst_height)
    new_coords.update(coords)
    return new_coords


def _make_dst_affine(
    src_data_array, src_crs, dst_crs, dst_resolution=None, dst_shape=None
):
    """Determine the affine of the new projected `xarray.DataArray`"""
    src_bounds = src_data_array.rio.bounds()
    src_height, src_width = src_data_array.rio.shape
    dst_height, dst_width = dst_shape if dst_shape is not None else (None, None)
    resolution_or_width_height = {
        k: v
        for k, v in [
            ("resolution", dst_resolution),
            ("dst_height", dst_height),
            ("dst_width", dst_width),
        ]
        if v is not None
    }
    dst_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        src_width,
        src_height,
        *src_bounds,
        **resolution_or_width_height,
    )
    return dst_affine, dst_width, dst_height


def _write_metatata_to_raster(raster_handle, xarray_dataset, tags):
    """
    Write the metadata stored in the xarray object to raster metadata
    """
    tags = xarray_dataset.attrs if tags is None else {**xarray_dataset.attrs, **tags}

    # write scales and offsets
    try:
        raster_handle.scales = tags["scales"]
    except KeyError:
        try:
            raster_handle.scales = (tags["scale_factor"],) * raster_handle.count
        except KeyError:
            pass
    try:
        raster_handle.offsets = tags["offsets"]
    except KeyError:
        try:
            raster_handle.offsets = (tags["add_offset"],) * raster_handle.count
        except KeyError:
            pass

    # filter out attributes that should be written in a different location
    skip_tags = (
        UNWANTED_RIO_ATTRS
        + FILL_VALUE_NAMES
        + ("transform", "scales", "scale_factor", "add_offset", "offsets")
    )
    # this is for when multiple values are used
    # in this case, it will be stored in the raster description
    if not isinstance(tags.get("long_name"), str):
        skip_tags += ("long_name",)
    tags = {key: value for key, value in tags.items() if key not in skip_tags}
    raster_handle.update_tags(**tags)

    # write band name information
    long_name = xarray_dataset.attrs.get("long_name")
    if isinstance(long_name, (tuple, list)):
        if len(long_name) != raster_handle.count:
            raise RioXarrayError(
                "Number of names in the 'long_name' attribute does not equal "
                "the number of bands."
            )
        for iii, band_description in enumerate(long_name):
            raster_handle.set_band_description(iii + 1, band_description)
    else:
        band_description = long_name or xarray_dataset.name
        if band_description:
            for iii in range(raster_handle.count):
                raster_handle.set_band_description(iii + 1, band_description)


def _get_data_var_message(obj):
    """
    Get message for named data variables.
    """
    try:
        return f" Data variable: {obj.name}" if obj.name else ""
    except AttributeError:
        return ""


def _ensure_nodata_dtype(original_nodata, new_dtype):
    """
    Convert the nodata to the new datatype and raise warning
    if the value of the nodata value changed.
    """
    original_nodata = float(original_nodata)
    nodata = np.dtype(new_dtype).type(original_nodata)
    if not np.isnan(nodata) and original_nodata != nodata:
        warnings.warn(
            f"The nodata value ({original_nodata}) has been automatically "
            f"changed to ({nodata}) to match the dtype of the data."
        )
    return nodata


class XRasterBase(object):
    """This is the base class for the GIS extensions for xarray"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        self._x_dim = None
        self._y_dim = None
        # Determine the spatial dimensions of the `xarray.DataArray`
        if "x" in self._obj.dims and "y" in self._obj.dims:
            self._x_dim = "x"
            self._y_dim = "y"
        elif "longitude" in self._obj.dims and "latitude" in self._obj.dims:
            self._x_dim = "longitude"
            self._y_dim = "latitude"
        else:
            # look for coordinates with CF attributes
            for coord in self._obj.coords:
                # make sure to only look in 1D coordinates
                # that has the same dimension name as the coordinate
                if self._obj.coords[coord].dims != (coord,):
                    continue
                elif (self._obj.coords[coord].attrs.get("axis", "").upper() == "X") or (
                    self._obj.coords[coord].attrs.get("standard_name", "").lower()
                    in ("longitude", "projection_x_coordinate")
                ):
                    self._x_dim = coord
                elif (self._obj.coords[coord].attrs.get("axis", "").upper() == "Y") or (
                    self._obj.coords[coord].attrs.get("standard_name", "").lower()
                    in ("latitude", "projection_y_coordinate")
                ):
                    self._y_dim = coord

        # properties
        self._count = None
        self._height = None
        self._width = None
        self._crs = None

    @property
    def crs(self):
        """:obj:`rasterio.crs.CRS`:
        Retrieve projection from :obj:`xarray.Dataset` | :obj:`xarray.DataArray`
        """
        if self._crs is not None:
            return None if self._crs is False else self._crs

        # look in grid_mapping
        try:
            self.set_crs(
                pyproj.CRS.from_cf(self._obj.coords[self.grid_mapping].attrs),
                inplace=True,
            )
        except (KeyError, pyproj.exceptions.CRSError):
            try:
                # look in attrs for 'crs'
                self.set_crs(self._obj.attrs["crs"], inplace=True)
            except KeyError:
                self._crs = False
                return None
        return self._crs

    def _get_obj(self, inplace):
        """
        Get the object to modify.

        Parameters
        ----------
        inplace: bool
            If True, returns self.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`
        """
        if inplace:
            return self._obj
        obj_copy = self._obj.copy(deep=True)
        # preserve attribute information
        obj_copy.rio._x_dim = self._x_dim
        obj_copy.rio._y_dim = self._y_dim
        obj_copy.rio._width = self._width
        obj_copy.rio._height = self._height
        obj_copy.rio._crs = self._crs
        return obj_copy

    def set_crs(self, input_crs, inplace=True):
        """
        Set the CRS value for the Dataset/DataArray without modifying
        the dataset/data array.

        Parameters
        ----------
        input_crs: object
            Anything accepted by `rasterio.crs.CRS.from_user_input`.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Dataset with crs attribute.
        """
        crs = CRS.from_wkt(crs_to_wkt(input_crs))
        obj = self._get_obj(inplace=inplace)
        obj.rio._crs = crs
        return obj

    @property
    def grid_mapping(self):
        """
        str: The CF grid_mapping attribute. 'spatial_ref' is the default.
        """
        try:
            return self._obj.attrs["grid_mapping"]
        except KeyError:
            pass
        grid_mapping = DEFAULT_GRID_MAP
        # search the dataset for the grid mapping name
        if hasattr(self._obj, "data_vars"):
            grid_mappings = set()
            for var in self._obj.data_vars:
                if (
                    self.x_dim in self._obj[var].dims
                    and self.y_dim in self._obj[var].dims
                ):
                    try:
                        grid_mapping = self._obj[var].attrs["grid_mapping"]
                        grid_mappings.add(grid_mapping)
                    except KeyError:
                        pass
            if len(grid_mappings) > 1:
                raise RioXarrayError("Multiple grid mappings exist.")
        return grid_mapping

    def write_grid_mapping(self, grid_mapping_name=DEFAULT_GRID_MAP, inplace=False):
        """
        Write the CF grid_mapping attribute.

        Parameters
        ----------
        grid_mapping_name: str, optional
            Name of the grid_mapping coordinate.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with CF compliant CRS information.
        """
        data_obj = self._get_obj(inplace=inplace)
        if hasattr(data_obj, "data_vars"):
            for var in data_obj.data_vars:
                if (
                    self.x_dim in data_obj[var].dims
                    and self.y_dim in data_obj[var].dims
                ):
                    data_obj[var].rio.update_attrs(
                        dict(grid_mapping=grid_mapping_name), inplace=True
                    ).rio.set_spatial_dims(
                        x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
                    )
        return data_obj.rio.update_attrs(
            dict(grid_mapping=grid_mapping_name), inplace=True
        )

    def write_crs(self, input_crs=None, grid_mapping_name=None, inplace=False):
        """
        Write the CRS to the dataset in a CF compliant manner.

        Parameters
        ----------
        input_crs: object
            Anything accepted by `rasterio.crs.CRS.from_user_input`.
        grid_mapping_name: str, optional
            Name of the grid_mapping coordinate to store the CRS information in.
            Default is the grid_mapping name of the dataset.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with CF compliant CRS information.
        """
        if input_crs is not None:
            data_obj = self.set_crs(input_crs, inplace=inplace)
        else:
            data_obj = self._get_obj(inplace=inplace)

        # get original transform
        transform = self._cached_transform()
        # remove old grid maping coordinate if exists
        grid_mapping_name = (
            self.grid_mapping if grid_mapping_name is None else grid_mapping_name
        )
        try:
            del data_obj.coords[grid_mapping_name]
        except KeyError:
            pass

        if data_obj.rio.crs is None:
            raise MissingCRS(
                "CRS not found. Please set the CRS with 'set_crs()' or 'write_crs()'."
            )
        # add grid mapping coordinate
        data_obj.coords[grid_mapping_name] = xarray.Variable((), 0)
        grid_map_attrs = pyproj.CRS.from_user_input(data_obj.rio.crs).to_cf()
        # spatial_ref is for compatibility with GDAL
        crs_wkt = crs_to_wkt(data_obj.rio.crs)
        grid_map_attrs["spatial_ref"] = crs_wkt
        grid_map_attrs["crs_wkt"] = crs_wkt
        if transform is not None:
            grid_map_attrs["GeoTransform"] = " ".join(
                [str(item) for item in transform.to_gdal()]
            )
        data_obj.coords[grid_mapping_name].rio.set_attrs(grid_map_attrs, inplace=True)

        return data_obj.rio.write_grid_mapping(
            grid_mapping_name=grid_mapping_name, inplace=True
        )

    def _cached_transform(self):
        """
        Get the transform from:
        1. The GeoTransform metatada property in the grid mapping
        2. The transform attribute.
        """
        try:
            # look in grid_mapping
            return Affine.from_gdal(
                *np.fromstring(
                    self._obj.coords[self.grid_mapping].attrs["GeoTransform"], sep=" "
                )
            )
        except KeyError:
            try:
                return Affine(*self._obj.attrs["transform"][:6])
            except KeyError:
                pass
        return None

    def write_transform(self, transform=None, grid_mapping_name=None, inplace=False):
        """
        .. versionadded:: 0.0.30

        Write the GeoTransform to the dataset where GDAL can read it in.

        https://gdal.org/drivers/raster/netcdf.html#georeference

        Parameters
        ----------
        transform: affine.Affine, optional
            The transform of the dataset. If not provided, it will be calculated.
        grid_mapping_name: str, optional
            Name of the grid_mapping coordinate to store the transform information in.
            Default is the grid_mapping name of the dataset.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with Geo Transform written.
        """
        transform = transform or self.transform(recalc=True)
        data_obj = self._get_obj(inplace=inplace)
        # delete the old attribute to prevent confusion
        data_obj.attrs.pop("transform", None)
        grid_mapping_name = (
            self.grid_mapping if grid_mapping_name is None else grid_mapping_name
        )
        try:
            grid_map_attrs = data_obj.coords[grid_mapping_name].attrs.copy()
        except KeyError:
            data_obj.coords[grid_mapping_name] = xarray.Variable((), 0)
            grid_map_attrs = data_obj.coords[grid_mapping_name].attrs.copy()
        grid_map_attrs["GeoTransform"] = " ".join(
            [str(item) for item in transform.to_gdal()]
        )
        data_obj.coords[grid_mapping_name].rio.set_attrs(grid_map_attrs, inplace=True)
        return data_obj.rio.write_grid_mapping(
            grid_mapping_name=grid_mapping_name, inplace=True
        )

    def transform(self, recalc=False):
        """
        Parameters
        ----------
        recalc: bool, optional
            If True, it will re-calculate the transform instead of using
            the cached transform.

        Returns
        -------
        :obj:`affine.Afffine`:
            The affine of the :obj:`xarray.Dataset` | :obj:`xarray.DataArray`
        """
        try:
            src_left, _, _, src_top = self.bounds(recalc=recalc)
            src_resolution_x, src_resolution_y = self.resolution(recalc=recalc)
        except (DimensionMissingCoordinateError, DimensionError):
            return Affine.identity()
        return Affine.translation(src_left, src_top) * Affine.scale(
            src_resolution_x, src_resolution_y
        )

    def write_coordinate_system(self, inplace=False):
        """
        Write the coordinate system CF metadata.

        .. versionadded:: 0.0.30

        Parameters
        ----------
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            The dataset with the CF coordinate system attributes added.
        """
        data_obj = self._get_obj(inplace=inplace)
        # add metadata to x,y coordinates
        is_projected = data_obj.rio.crs and data_obj.rio.crs.is_projected
        is_geographic = data_obj.rio.crs and data_obj.rio.crs.is_geographic
        x_coord_attrs = dict(data_obj.coords[self.x_dim].attrs)
        x_coord_attrs["axis"] = "X"
        y_coord_attrs = dict(data_obj.coords[self.y_dim].attrs)
        y_coord_attrs["axis"] = "Y"
        if is_projected:
            units = None
            if hasattr(data_obj.rio.crs, "linear_units_factor"):
                unit_factor = data_obj.rio.crs.linear_units_factor[-1]
                if unit_factor != 1:
                    units = f"{unit_factor} metre"
                else:
                    units = "metre"
            # X metadata
            x_coord_attrs["long_name"] = "x coordinate of projection"
            x_coord_attrs["standard_name"] = "projection_x_coordinate"
            if units:
                x_coord_attrs["units"] = units
            # Y metadata
            y_coord_attrs["long_name"] = "y coordinate of projection"
            y_coord_attrs["standard_name"] = "projection_y_coordinate"
            if units:
                y_coord_attrs["units"] = units
        elif is_geographic:
            # X metadata
            x_coord_attrs["long_name"] = "longitude"
            x_coord_attrs["standard_name"] = "longitude"
            x_coord_attrs["units"] = "degrees_east"
            # Y metadata
            y_coord_attrs["long_name"] = "latitude"
            y_coord_attrs["standard_name"] = "latitude"
            y_coord_attrs["units"] = "degrees_north"
        data_obj.coords[self.y_dim].attrs = y_coord_attrs
        data_obj.coords[self.x_dim].attrs = x_coord_attrs
        return data_obj

    def set_attrs(self, new_attrs, inplace=False):
        """
        Set the attributes of the dataset/dataarray and reset
        rioxarray properties to re-search for them.

        Parameters
        ----------
        new_attrs: dict
            A dictionary of new attributes.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with new attributes.
        """
        data_obj = self._get_obj(inplace=inplace)
        # set the attributes
        data_obj.attrs = new_attrs
        # reset rioxarray properties depending
        # on attributes to be generated
        data_obj.rio._nodata = None
        data_obj.rio._crs = None
        return data_obj

    def update_attrs(self, new_attrs, inplace=False):
        """
        Update the attributes of the dataset/dataarray and reset
        rioxarray properties to re-search for them.

        Parameters
        ----------
        new_attrs: dict
            A dictionary of new attributes to update with.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with updated attributes.
        """
        data_attrs = dict(self._obj.attrs)
        data_attrs.update(**new_attrs)
        return self.set_attrs(data_attrs, inplace=inplace)

    def set_spatial_dims(self, x_dim, y_dim, inplace=True):
        """
        This sets the spatial dimensions of the dataset.

        Parameters
        ----------
        x_dim: str
            The name of the x dimension.
        y_dim: str
            The name of the y dimension.
        inplace: bool, optional
            If True, it will modify the dataframe in place.
            Otherwise it will return a modified copy.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Dataset with spatial dimensions set.
        """

        def set_dims(obj, in_x_dim, in_y_dim):
            if in_x_dim in obj.dims:
                obj.rio._x_dim = x_dim
            else:
                raise DimensionError(
                    f"x dimension ({x_dim}) not found.{_get_data_var_message(obj)}"
                )
            if y_dim in obj.dims:
                obj.rio._y_dim = y_dim
            else:
                raise DimensionError(
                    f"y dimension ({x_dim}) not found.{_get_data_var_message(obj)}"
                )

        data_obj = self._get_obj(inplace=inplace)
        set_dims(data_obj, x_dim, y_dim)
        return data_obj

    @property
    def x_dim(self):
        """str: The dimension for the X-axis."""
        if self._x_dim is not None:
            return self._x_dim
        raise DimensionError(
            "x dimension not found. 'set_spatial_dims()' can address this."
            f"{_get_data_var_message(self._obj)}"
        )

    @property
    def y_dim(self):
        """str: The dimension for the Y-axis."""
        if self._y_dim is not None:
            return self._y_dim
        raise DimensionError(
            "x dimension not found. 'set_spatial_dims()' can address this."
            f"{_get_data_var_message(self._obj)}"
        )

    @property
    def width(self):
        """int: Returns the width of the dataset (x dimension size)"""
        if self._width is not None:
            return self._width
        self._width = self._obj[self.x_dim].size
        return self._width

    @property
    def height(self):
        """int: Returns the height of the dataset (y dimension size)"""
        if self._height is not None:
            return self._height
        self._height = self._obj[self.y_dim].size
        return self._height

    @property
    def shape(self):
        """tuple(int, int): Returns the shape (height, width)"""
        return (self.height, self.width)

    def _check_dimensions(self):
        """
        This function validates that the dimensions 2D/3D and
        they are are in the proper order.

        Returns
        -------
        str or None: Name extra dimension.
        """
        extra_dims = list(set(list(self._obj.dims)) - set([self.x_dim, self.y_dim]))
        if len(extra_dims) > 1:
            raise TooManyDimensions(
                "Only 2D and 3D data arrays supported."
                f"{_get_data_var_message(self._obj)}"
            )
        elif extra_dims and self._obj.dims != (extra_dims[0], self.y_dim, self.x_dim):
            raise InvalidDimensionOrder(
                "Invalid dimension order. Expected order: {0}. "
                "You can use `DataArray.transpose{0}`"
                " to reorder your dimensions.".format(
                    (extra_dims[0], self.y_dim, self.x_dim)
                )
                + f"{_get_data_var_message(self._obj)}"
            )
        elif not extra_dims and self._obj.dims != (self.y_dim, self.x_dim):
            raise InvalidDimensionOrder(
                "Invalid dimension order. Expected order: {0}"
                "You can use `DataArray.transpose{0}` "
                "to reorder your dimensions.".format((self.y_dim, self.x_dim))
                + f"{_get_data_var_message(self._obj)}"
            )
        return extra_dims[0] if extra_dims else None

    @property
    def count(self):
        """int: Returns the band count (z dimension size)"""
        if self._count is not None:
            return self._count
        extra_dim = self._check_dimensions()
        self._count = 1
        if extra_dim is not None:
            self._count = self._obj[extra_dim].size
        return self._count

    def _internal_bounds(self):
        """Determine the internal bounds of the `xarray.DataArray`"""
        if self.x_dim not in self._obj.coords:
            raise DimensionMissingCoordinateError(f"{self.x_dim} missing coordinates.")
        elif self.y_dim not in self._obj.coords:
            raise DimensionMissingCoordinateError(f"{self.y_dim} missing coordinates.")
        try:
            left = float(self._obj[self.x_dim][0])
            right = float(self._obj[self.x_dim][-1])
            top = float(self._obj[self.y_dim][0])
            bottom = float(self._obj[self.y_dim][-1])
        except IndexError:
            raise NoDataInBounds(
                "Unable to determine bounds from coordinates."
                f"{_get_data_var_message(self._obj)}"
            )
        return left, bottom, right, top

    def resolution(self, recalc=False):
        """
        Parameters
        ----------
        recalc: bool, optional
            Will force the resolution to be recalculated instead of using the
            transform attribute.

        Returns
        -------
        x_resolution, y_resolution: float
            The resolution of the `xarray.DataArray` | `xarray.Dataset`
        """
        transform = self._cached_transform()

        if (
            not recalc or self.width == 1 or self.height == 1
        ) and transform is not None:
            resolution_x = transform.a
            resolution_y = transform.e
            return resolution_x, resolution_y

        # if the coordinates of the spatial dimensions are missing
        # use the cached transform resolution
        try:
            left, bottom, right, top = self._internal_bounds()
        except DimensionMissingCoordinateError:
            if transform is None:
                raise
            resolution_x = transform.a
            resolution_y = transform.e
            return resolution_x, resolution_y

        if self.width == 1 or self.height == 1:
            raise OneDimensionalRaster(
                "Only 1 dimenional array found. Cannot calculate the resolution."
                f"{_get_data_var_message(self._obj)}"
            )

        resolution_x = (right - left) / (self.width - 1)
        resolution_y = (bottom - top) / (self.height - 1)
        return resolution_x, resolution_y

    def bounds(self, recalc=False):
        """
        Parameters
        ----------
        recalc: bool, optional
            Will force the bounds to be recalculated instead of using the
            transform attribute.

        Returns
        -------
        left, bottom, right, top: float
            Outermost coordinates of the `xarray.DataArray` | `xarray.Dataset`.
        """
        resolution_x, resolution_y = self.resolution(recalc=recalc)

        try:
            # attempt to get bounds from xarray coordinate values
            left, bottom, right, top = self._internal_bounds()
            left -= resolution_x / 2.0
            right += resolution_x / 2.0
            top -= resolution_y / 2.0
            bottom += resolution_y / 2.0
        except DimensionMissingCoordinateError:
            transform = self._cached_transform()
            left = transform.c
            top = transform.f
            right = left + resolution_x * self.width
            bottom = top + resolution_y * self.height

        return left, bottom, right, top

    def isel_window(self, window):
        """
        Use a rasterio.window.Window to select a subset of the data.

        .. warning:: Float indices are converted to integers.

        Parameters
        ----------
        window: :class:`rasterio.window.Window`
            The window of the dataset to read.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            The data in the window.
        """
        (row_start, row_stop), (col_start, col_stop) = window.toranges()
        row_start = math.ceil(row_start) if row_start < 0 else math.floor(row_start)
        row_stop = math.floor(row_stop) if row_stop < 0 else math.ceil(row_stop)
        col_start = math.ceil(col_start) if col_start < 0 else math.floor(col_start)
        col_stop = math.floor(col_stop) if col_stop < 0 else math.ceil(col_stop)
        row_slice = slice(int(row_start), int(row_stop))
        col_slice = slice(int(col_start), int(col_stop))
        return (
            self._obj.isel({self.y_dim: row_slice, self.x_dim: col_slice})
            .copy()  # this is to prevent sharing coordinates with the original dataset
            .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
            .rio.write_transform(
                transform=rasterio.windows.transform(
                    rasterio.windows.Window.from_slices(
                        rows=row_slice,
                        cols=col_slice,
                        width=self.width,
                        height=self.height,
                    ),
                    self.transform(recalc=True),
                ),
                inplace=True,
            )
        )

    def slice_xy(self, minx, miny, maxx, maxy):
        """Slice the array by x,y bounds.

        Parameters
        ----------
        minx: float
            Minimum bound for x coordinate.
        miny: float
            Minimum bound for y coordinate.
        maxx: float
            Maximum bound for x coordinate.
        maxy: float
            Maximum bound for y coordinate.


        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            The data in the slice.
        """
        left, bottom, right, top = self._internal_bounds()
        if top > bottom:
            y_slice = slice(maxy, miny)
        else:
            y_slice = slice(miny, maxy)

        if left > right:
            x_slice = slice(maxx, minx)
        else:
            x_slice = slice(minx, maxx)

        subset = (
            self._obj.sel({self.x_dim: x_slice, self.y_dim: y_slice})
            .copy()  # this is to prevent sharing coordinates with the original dataset
            .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
            .rio.write_transform(inplace=True)
        )
        return subset

    def transform_bounds(self, dst_crs, densify_pts=21, recalc=False):
        """Transform bounds from src_crs to dst_crs.

        Optionally densifying the edges (to account for nonlinear transformations
        along these edges) and extracting the outermost bounds.

        Note: this does not account for the antimeridian.

        Parameters
        ----------
        dst_crs: str, :obj:`rasterio.crs.CRS`, or dict
            Target coordinate reference system.
        densify_pts: uint, optional
            Number of points to add to each edge to account for nonlinear
            edges produced by the transform process.  Large numbers will produce
            worse performance.  Default: 21 (gdal default).
        recalc: bool, optional
            Will force the bounds to be recalculated instead of using the transform
            attribute.

        Returns
        -------
        left, bottom, right, top: float
            Outermost coordinates in target coordinate reference system.
        """
        return rasterio.warp.transform_bounds(
            self.crs, dst_crs, *self.bounds(recalc=recalc), densify_pts=densify_pts
        )


@xarray.register_dataarray_accessor("rio")
class RasterArray(XRasterBase):
    """This is the GIS extension for :obj:`xarray.DataArray`"""

    def __init__(self, xarray_obj):
        super(RasterArray, self).__init__(xarray_obj)
        # properties
        self._nodata = None

    def set_nodata(self, input_nodata, inplace=True):
        """
        Set the nodata value for the DataArray without modifying
        the data array.

        Parameters
        ----------
        input_nodata: object
            Valid nodata for dtype.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.DataArray`:
            Dataset with nodata attribute set.
        """
        obj = self._get_obj(inplace=inplace)
        obj.rio._nodata = input_nodata
        return obj

    def write_nodata(self, input_nodata, inplace=False):
        """
        Write the nodata to the DataArray in a CF compliant manner.

        Parameters
        ----------
        input_nodata: object
            Nodata value for the DataArray.
            If input_nodata is None, it will remove the _FillValue attribute.
        inplace: bool, optional
            If True, it will write to the existing DataArray. Default is False.

        Returns
        -------
        :obj:`xarray.DataArray`:
            Modified DataArray with CF compliant nodata information.
        """
        data_obj = self._get_obj(inplace=inplace)
        input_nodata = False if input_nodata is None else input_nodata
        if input_nodata is not False:
            input_nodata = _ensure_nodata_dtype(input_nodata, self._obj.dtype)
            data_obj.rio.update_attrs(dict(_FillValue=input_nodata), inplace=True)
        else:
            new_vars = dict(data_obj.attrs)
            new_vars.pop("_FillValue", None)
            data_obj.rio.set_attrs(new_vars, inplace=True)
        data_obj.rio.set_nodata(input_nodata, inplace=True)
        return data_obj

    @property
    def encoded_nodata(self):
        """Return the encoded nodata value for the dataset if encoded."""
        encoded_nodata = self._obj.encoding.get("_FillValue")
        if encoded_nodata is None:
            return None
        return _ensure_nodata_dtype(encoded_nodata, self._obj.dtype)

    @property
    def nodata(self):
        """Get the nodata value for the dataset."""
        if self._nodata is not None:
            return None if self._nodata is False else self._nodata

        if self.encoded_nodata is not None:
            self._nodata = np.nan
        else:
            self._nodata = self._obj.attrs.get(
                "_FillValue",
                self._obj.attrs.get(
                    "missing_value",
                    self._obj.attrs.get("fill_value", self._obj.attrs.get("nodata")),
                ),
            )

        # look in places used by `xarray.open_rasterio`
        if self._nodata is None:
            try:
                self._nodata = self._obj._file_obj.acquire().nodata
            except AttributeError:
                try:
                    self._nodata = self._obj.attrs["nodatavals"][0]
                except (KeyError, IndexError):
                    pass

        if self._nodata is None:
            self._nodata = False
            return None

        self._nodata = _ensure_nodata_dtype(self._nodata, self._obj.dtype)
        return self._nodata

    def reproject(
        self,
        dst_crs,
        resolution=None,
        shape=None,
        transform=None,
        resampling=Resampling.nearest,
    ):
        """
        Reproject :obj:`xarray.DataArray` objects

        Powered by `rasterio.warp.reproject`

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        .. versionadded:: 0.0.27 shape
        .. versionadded:: 0.0.28 transform

        Parameters
        ----------
        dst_crs: str
            OGC WKT string or Proj.4 string.
        resolution: float or tuple(float, float), optional
            Size of a destination pixel in destination projection units
            (e.g. degrees or metres).
        shape: tuple(int, int), optional
            Shape of the destination in pixels (dst_height, dst_width). Cannot be used
            together with resolution.
        transform: optional
            The destination transform.
        resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.


        Returns
        -------
        :obj:`xarray.DataArray`:
            The reprojected DataArray.
        """
        if resolution is not None and (shape is not None or transform is not None):
            raise RioXarrayError("resolution cannot be used with shape or transform.")
        if self.crs is None:
            raise MissingCRS(
                "CRS not found. Please set the CRS with 'set_crs()' or 'write_crs()'."
                f"{_get_data_var_message(self._obj)}"
            )
        src_affine = self.transform(recalc=True)
        if transform is None:
            dst_affine, dst_width, dst_height = _make_dst_affine(
                self._obj, self.crs, dst_crs, resolution, shape
            )
        else:
            dst_affine = transform
            if shape is not None:
                dst_height, dst_width = shape
            else:
                dst_height, dst_width = self.shape

        extra_dim = self._check_dimensions()
        if extra_dim:
            dst_data = np.zeros(
                (self._obj[extra_dim].size, dst_height, dst_width),
                dtype=self._obj.dtype.type,
            )
        else:
            dst_data = np.zeros((dst_height, dst_width), dtype=self._obj.dtype.type)

        try:
            dst_nodata = self._obj.dtype.type(
                self.nodata if self.nodata is not None else -9999
            )
        except ValueError:
            # if integer, set nodata to -9999
            dst_nodata = self._obj.dtype.type(-9999)

        src_nodata = self._obj.dtype.type(
            self.nodata if self.nodata is not None else dst_nodata
        )
        rasterio.warp.reproject(
            source=self._obj.values,
            destination=dst_data,
            src_transform=src_affine,
            src_crs=self.crs,
            src_nodata=src_nodata,
            dst_transform=dst_affine,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
        # add necessary attributes
        new_attrs = _generate_attrs(self._obj, dst_nodata)
        # make sure dimensions with coordinates renamed to x,y
        dst_dims = []
        for dim in self._obj.dims:
            if dim == self.x_dim:
                dst_dims.append("x")
            elif dim == self.y_dim:
                dst_dims.append("y")
            else:
                dst_dims.append(dim)
        xda = xarray.DataArray(
            name=self._obj.name,
            data=dst_data,
            coords=_make_coords(self._obj, dst_affine, dst_width, dst_height),
            dims=tuple(dst_dims),
            attrs=new_attrs,
        )
        xda.encoding = self._obj.encoding
        xda.rio.write_transform(dst_affine, inplace=True)
        xda.rio.write_crs(dst_crs, inplace=True)
        xda.rio.write_coordinate_system(inplace=True)
        return xda

    def reproject_match(self, match_data_array, resampling=Resampling.nearest):
        """
        Reproject a DataArray object to match the resolution, projection,
        and region of another DataArray.

        Powered by `rasterio.warp.reproject`

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        Parameters
        ----------
        match_data_array:  :obj:`xarray.DataArray` | :obj:`xarray.Dataset`
            DataArray of the target resolution and projection.
        resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.


        Returns
        --------
        :obj:`xarray.DataArray`:
            Contains the data from the src_data_array, reprojected to match
            match_data_array.
        """
        dst_crs = crs_to_wkt(match_data_array.rio.crs)
        return self.reproject(
            dst_crs,
            transform=match_data_array.rio.transform(recalc=True),
            shape=match_data_array.rio.shape,
            resampling=resampling,
        )

    def pad_xy(self, minx, miny, maxx, maxy, constant_values):
        """Pad the array to x,y bounds.

        .. versionadded:: 0.0.29

        Parameters
        ----------
        minx: float
            Minimum bound for x coordinate.
        miny: float
            Minimum bound for y coordinate.
        maxx: float
            Maximum bound for x coordinate.
        maxy: float
            Maximum bound for y coordinate.
        constant_values: scalar
            The value used for padding. If None, nodata will be used if it is
            set, and np.nan otherwise.


        Returns
        -------
        :obj:`xarray.DataArray`:
            The padded object.
        """
        left, bottom, right, top = self._internal_bounds()
        resolution_x, resolution_y = self.resolution()
        y_before = y_after = 0
        x_before = x_after = 0
        y_coord = self._obj[self.y_dim]
        x_coord = self._obj[self.x_dim]

        if top - resolution_y < maxy:
            new_y_coord = np.arange(bottom, maxy, -resolution_y)[::-1]
            y_before = len(new_y_coord) - len(y_coord)
            y_coord = new_y_coord
            top = y_coord[0]
        if bottom + resolution_y > miny:
            new_y_coord = np.arange(top, miny, resolution_y)
            y_after = len(new_y_coord) - len(y_coord)
            y_coord = new_y_coord
            bottom = y_coord[-1]

        if left - resolution_x > minx:
            new_x_coord = np.arange(right, minx, -resolution_x)[::-1]
            x_before = len(new_x_coord) - len(x_coord)
            x_coord = new_x_coord
            left = x_coord[0]
        if right + resolution_x < maxx:
            new_x_coord = np.arange(left, maxx, resolution_x)
            x_after = len(new_x_coord) - len(x_coord)
            x_coord = new_x_coord
            right = x_coord[-1]

        if constant_values is None:
            constant_values = np.nan if self.nodata is None else self.nodata

        superset = self._obj.pad(
            pad_width={
                self.x_dim: (x_before, x_after),
                self.y_dim: (y_before, y_after),
            },
            constant_values=constant_values,
        ).rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
        superset[self.x_dim] = x_coord
        superset[self.y_dim] = y_coord
        superset.rio.write_transform(inplace=True)
        return superset

    def pad_box(self, minx, miny, maxx, maxy, constant_values=None):
        """Pad the :obj:`xarray.DataArray` to a bounding box

        .. versionadded:: 0.0.29

        Parameters
        ----------
        minx: float
            Minimum bound for x coordinate.
        miny: float
            Minimum bound for y coordinate.
        maxx: float
            Maximum bound for x coordinate.
        maxy: float
            Maximum bound for y coordinate.
        constant_values: scalar
            The value used for padding. If None, nodata will be used if it is
            set, and np.nan otherwise.


        Returns
        -------
        :obj:`xarray.DataArray`:
            The padded object.
        """
        resolution_x, resolution_y = self.resolution()

        pad_minx = minx - abs(resolution_x) / 2.0
        pad_miny = miny - abs(resolution_y) / 2.0
        pad_maxx = maxx + abs(resolution_x) / 2.0
        pad_maxy = maxy + abs(resolution_y) / 2.0

        pd_array = self.pad_xy(pad_minx, pad_miny, pad_maxx, pad_maxy, constant_values)

        # make sure correct attributes preserved & projection added
        _add_attrs_proj(pd_array, self._obj)

        return pd_array

    def clip_box(self, minx, miny, maxx, maxy, auto_expand=False, auto_expand_limit=3):
        """Clip the :obj:`xarray.DataArray` by a bounding box.

        Parameters
        ----------
        minx: float
            Minimum bound for x coordinate.
        miny: float
            Minimum bound for y coordinate.
        maxx: float
            Maximum bound for x coordinate.
        maxy: float
            Maximum bound for y coordinate.
        auto_expand: bool
            If True, it will expand clip search if only 1D raster found with clip.
        auto_expand_limit: int
            maximum number of times the clip will be retried before raising
            an exception.

        Returns
        -------
        :obj:`xarray.DataArray`:
            The clipped object.
        """
        if self.width == 1 or self.height == 1:
            raise OneDimensionalRaster(
                "At least one of the raster x,y coordinates has only one point."
                f"{_get_data_var_message(self._obj)}"
            )

        # make sure that if the coordinates are
        # in reverse order that it still works
        resolution_x, resolution_y = self.resolution()
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

        # pull the data out
        window = rasterio.windows.from_bounds(
            left=np.array(left).item(),
            bottom=np.array(bottom).item(),
            right=np.array(right).item(),
            top=np.array(top).item(),
            transform=self.transform(recalc=True),
            width=self.width,
            height=self.height,
        )
        cl_array = self.isel_window(window)

        # check that the window has data in it
        if cl_array.rio.width <= 1 or cl_array.rio.height <= 1:
            if auto_expand and auto_expand < auto_expand_limit:
                resolution_x, resolution_y = self.resolution()
                return self.clip_box(
                    minx=minx - abs(resolution_x) / 2.0,
                    miny=miny - abs(resolution_y) / 2.0,
                    maxx=maxx + abs(resolution_x) / 2.0,
                    maxy=maxy + abs(resolution_y) / 2.0,
                    auto_expand=int(auto_expand) + 1,
                    auto_expand_limit=auto_expand_limit,
                )
            if cl_array.rio.width < 1 or cl_array.rio.height < 1:
                raise NoDataInBounds(
                    f"No data found in bounds.{_get_data_var_message(self._obj)}"
                )
            elif cl_array.rio.width == 1 or cl_array.rio.height == 1:
                raise OneDimensionalRaster(
                    "At least one of the clipped raster x,y coordinates"
                    " has only one point."
                    f"{_get_data_var_message(self._obj)}"
                )

        # make sure correct attributes preserved & projection added
        _add_attrs_proj(cl_array, self._obj)
        return cl_array

    def clip(self, geometries, crs=None, all_touched=False, drop=True, invert=False):
        """
        Crops a :obj:`xarray.DataArray` by geojson like geometry dicts.

        Powered by `rasterio.features.geometry_mask`.

        Examples:

            >>> geometry = ''' {"type": "Polygon",
            ...                 "coordinates": [
            ...                 [[-94.07955380199459, 41.69085871273774],
            ...                 [-94.06082436942204, 41.69103313774798],
            ...                 [-94.06063203899649, 41.67932439500822],
            ...                 [-94.07935807746362, 41.679150041277325],
            ...                 [-94.07955380199459, 41.69085871273774]]]}'''
            >>> cropping_geometries = [geojson.loads(geometry)]
            >>> xds = xarray.open_rasterio('cool_raster.tif')
            >>> cropped = xds.rio.clip(geometries=cropping_geometries, crs=4326)


        Parameters
        ----------
        geometries: list
            A list of geojson geometry dicts.
        crs: :obj:`rasterio.crs.CRS`, optional
            The CRS of the input geometries. Default is to assume it is the same
            as the dataset.
        all_touched : bool, optional
            If True, all pixels touched by geometries will be burned in.  If
            false, only pixels whose center is within the polygon or that
            are selected by Bresenham's line algorithm will be burned in.
        drop: bool, optional
            If True, drop the data outside of the extent of the mask geoemtries
            Otherwise, it will return the same raster with the data masked.
            Default is True.
        invert: boolean, optional
            If False, pixels that do not overlap shapes will be set as nodata.
            Otherwise, pixels that overlap the shapes will be set as nodata.
            False by default.

        Returns
        -------
        :obj:`xarray.DataArray`:
            The clipped object.
        """
        if self.crs is None:
            raise MissingCRS(
                "CRS not found. Please set the CRS with 'set_crs()' or 'write_crs()'."
                f"{_get_data_var_message(self._obj)}"
            )
        crs = CRS.from_wkt(crs_to_wkt(crs)) if crs is not None else self.crs
        if self.crs != crs:
            geometries = [
                rasterio.warp.transform_geom(crs, self.crs, geometry)
                for geometry in geometries
            ]

        clip_mask_arr = geometry_mask(
            geometries=geometries,
            out_shape=(int(self.height), int(self.width)),
            transform=self.transform(recalc=True),
            invert=not invert,
            all_touched=all_touched,
        )
        clip_mask_xray = xarray.DataArray(
            clip_mask_arr,
            dims=(self.y_dim, self.x_dim),
        )
        cropped_ds = self._obj.where(clip_mask_xray)
        if drop:
            cropped_ds.rio.set_spatial_dims(
                x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
            )
            cropped_ds = cropped_ds.rio.isel_window(
                rasterio.windows.get_data_window(
                    np.ma.masked_array(clip_mask_arr, ~clip_mask_arr)
                )
            )
        if self.nodata is not None and not np.isnan(self.nodata):
            cropped_ds = cropped_ds.fillna(self.nodata)

        cropped_ds = cropped_ds.astype(self._obj.dtype)

        if (
            cropped_ds.coords[self.x_dim].size < 1
            or cropped_ds.coords[self.y_dim].size < 1
        ):
            raise NoDataInBounds(
                f"No data found in bounds.{_get_data_var_message(self._obj)}"
            )

        # make sure correct attributes preserved & projection added
        _add_attrs_proj(cropped_ds, self._obj)

        return cropped_ds

    def _interpolate_na(self, src_data, method="nearest"):
        """
        This method uses scipy.interpolate.griddata to interpolate missing data.

        Parameters
        ----------
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :class:`numpy.ndarray`:
            An interpolated :class:`numpy.ndarray`.
        """
        src_data_flat = src_data.flatten()
        try:
            data_isnan = np.isnan(self.nodata)
        except TypeError:
            data_isnan = False
        if not data_isnan:
            data_bool = src_data_flat != self.nodata
        else:
            data_bool = ~np.isnan(src_data_flat)

        if not data_bool.any():
            return src_data

        x_coords, y_coords = np.meshgrid(
            self._obj.coords[self.x_dim].values, self._obj.coords[self.y_dim].values
        )

        return griddata(
            points=(x_coords.flatten()[data_bool], y_coords.flatten()[data_bool]),
            values=src_data_flat[data_bool],
            xi=(x_coords, y_coords),
            method=method,
            fill_value=self.nodata,
        )

    def interpolate_na(self, method="nearest"):
        """
        This method uses scipy.interpolate.griddata to interpolate missing data.

        Parameters
        ----------
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :obj:`xarray.DataArray`:
            An interpolated :obj:`xarray.DataArray` object.
        """
        extra_dim = self._check_dimensions()
        if extra_dim:
            interp_data = []
            for _, sub_xds in self._obj.groupby(extra_dim):
                interp_data.append(
                    self._interpolate_na(sub_xds.load().data, method=method)
                )
            interp_data = np.array(interp_data)
        else:
            interp_data = self._interpolate_na(self._obj.load().data, method=method)

        interp_array = xarray.DataArray(
            name=self._obj.name,
            data=interp_data,
            coords=self._obj.coords,
            dims=self._obj.dims,
            attrs=self._obj.attrs,
        )
        interp_array.encoding = self._obj.encoding

        # make sure correct attributes preserved & projection added
        _add_attrs_proj(interp_array, self._obj)

        return interp_array

    def to_raster(
        self,
        raster_path,
        driver="GTiff",
        dtype=None,
        tags=None,
        windowed=False,
        recalc_transform=True,
        **profile_kwargs,
    ):
        """
        Export the DataArray to a raster file.

        Parameters
        ----------
        raster_path: str
            The path to output the raster to.
        driver: str, optional
            The name of the GDAL/rasterio driver to use to export the raster.
            Default is "GTiff".
        dtype: str, optional
            The data type to write the raster to. Default is the datasets dtype.
        tags: dict, optional
            A dictionary of tags to write to the raster.
        windowed: bool, optional
            If True, it will write using the windows of the output raster.
            This only works if the output raster is tiled. As such, if you
            set this to True, the output raster will be tiled.
            Default is False.
        **profile_kwargs
            Additional keyword arguments to pass into writing the raster. The
            nodata, transform, crs, count, width, and height attributes
            are ignored.

        """
        dtype = str(self._obj.dtype) if dtype is None else dtype
        # get the output profile from the rasterio object
        # if opened with xarray.open_rasterio()
        try:
            out_profile = self._obj._file_obj.acquire().profile
        except AttributeError:
            out_profile = {}
        out_profile.update(profile_kwargs)

        # filter out the generated attributes
        out_profile = {
            key: value
            for key, value in out_profile.items()
            if key
            not in (
                "driver",
                "height",
                "width",
                "crs",
                "transform",
                "nodata",
                "count",
                "dtype",
            )
        }
        rio_nodata = (
            self.encoded_nodata if self.encoded_nodata is not None else self.nodata
        )
        if rio_nodata is not None:
            # Ensure dtype of output data matches the expected dtype.
            # This check is added here as the dtype of the data is
            # converted right before writing.
            rio_nodata = _ensure_nodata_dtype(rio_nodata, dtype)

        with rasterio.open(
            raster_path,
            "w",
            driver=driver,
            height=int(self.height),
            width=int(self.width),
            count=int(self.count),
            dtype=dtype,
            crs=self.crs,
            transform=self.transform(recalc=recalc_transform),
            nodata=rio_nodata,
            **out_profile,
        ) as dst:

            _write_metatata_to_raster(dst, self._obj, tags)

            # write data to raster
            if windowed:
                window_iter = dst.block_windows(1)
            else:
                window_iter = [(None, None)]
            for _, window in window_iter:
                if window is not None:
                    out_data = self.isel_window(window)
                else:
                    out_data = self._obj
                if self.encoded_nodata is not None:
                    out_data = out_data.fillna(self.encoded_nodata)
                data = out_data.values.astype(dtype)
                if data.ndim == 2:
                    dst.write(data, 1, window=window)
                else:
                    dst.write(data, window=window)


@xarray.register_dataset_accessor("rio")
class RasterDataset(XRasterBase):
    """This is the GIS extension for :class:`xarray.Dataset`"""

    @property
    def vars(self):
        """list: Returns non-coordinate varibles"""
        return list(self._obj.data_vars)

    @property
    def crs(self):
        """:obj:`rasterio.crs.CRS`:
        Retrieve projection from `xarray.Dataset`
        """
        if self._crs is not None:
            return None if self._crs is False else self._crs
        self._crs = super().crs
        if self._crs is not None:
            return self._crs
        # ensure all the CRS of the variables are the same
        crs_list = []
        for var in self.vars:
            if self._obj[var].rio.crs is not None:
                crs_list.append(self._obj[var].rio.crs)
        try:
            crs = crs_list[0]
        except IndexError:
            crs = None
        if crs is None:
            self._crs = False
            return None
        elif all(crs_i == crs for crs_i in crs_list):
            self._crs = crs
        else:
            raise RioXarrayError(
                "CRS in DataArrays differ in the Dataset: {}".format(crs_list)
            )
        return self._crs

    def reproject(
        self,
        dst_crs,
        resolution=None,
        shape=None,
        transform=None,
        resampling=Resampling.nearest,
    ):
        """
        Reproject :class:`xarray.Dataset` objects

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        .. versionadded:: 0.0.27 shape
        .. versionadded:: 0.0.28 transform

        Parameters
        ----------
        dst_crs: str
            OGC WKT string or Proj.4 string.
        resolution: float or tuple(float, float), optional
            Size of a destination pixel in destination projection units
            (e.g. degrees or metres).
        shape: tuple(int, int), optional
            Shape of the destination in pixels (dst_height, dst_width). Cannot be used
            together with resolution.
        transform: optional
            The destination transform.
        resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.


        Returns
        --------
        :class:`xarray.Dataset`:
            The reprojected Dataset.
        """
        resampled_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            resampled_dataset[var] = (
                self._obj[var]
                .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
                .rio.reproject(
                    dst_crs,
                    resolution=resolution,
                    shape=shape,
                    transform=transform,
                    resampling=resampling,
                )
            )
        return resampled_dataset

    def reproject_match(self, match_data_array, resampling=Resampling.nearest):
        """
        Reproject a Dataset object to match the resolution, projection,
        and region of another DataArray.

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.


        Parameters
        ----------
        match_data_array: :obj:`xarray.DataArray` | :obj:`xarray.Dataset`
            Dataset with the target resolution and projection.
        resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.


        Returns
        --------
        :obj:`xarray.Dataset`:
            Contains the data from the src_data_array,
            reprojected to match match_data_array.
        """
        resampled_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            resampled_dataset[var] = (
                self._obj[var]
                .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
                .rio.reproject_match(match_data_array, resampling=resampling)
            )
        return resampled_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def pad_box(self, minx, miny, maxx, maxy):
        """Pad the :class:`xarray.Dataset` to a bounding box.

        .. warning:: Only works if all variables in the dataset have the
                     same coordinates.

        Parameters
        ----------
        minx: float
            Minimum bound for x coordinate.
        miny: float
            Minimum bound for y coordinate.
        maxx: float
            Maximum bound for x coordinate.
        maxy: float
            Maximum bound for y coordinate.

        Returns
        -------
        :obj:`xarray.Dataset`:
            The padded object.
        """
        padded_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            padded_dataset[var] = (
                self._obj[var]
                .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
                .rio.pad_box(minx, miny, maxx, maxy)
            )
        return padded_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def clip_box(self, minx, miny, maxx, maxy, auto_expand=False, auto_expand_limit=3):
        """Clip the :class:`xarray.Dataset` by a bounding box.

        .. warning:: Only works if all variables in the dataset have the
                     same coordinates.

        Parameters
        ----------
        minx: float
            Minimum bound for x coordinate.
        miny: float
            Minimum bound for y coordinate.
        maxx: float
            Maximum bound for x coordinate.
        maxy: float
            Maximum bound for y coordinate.
        auto_expand: bool
            If True, it will expand clip search if only 1D raster found with clip.
        auto_expand_limit: int
            maximum number of times the clip will be retried before raising
            an exception.

        Returns
        -------
        :obj:`Dataset`:
            The clipped object.
        """
        clipped_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            clipped_dataset[var] = (
                self._obj[var]
                .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
                .rio.clip_box(
                    minx,
                    miny,
                    maxx,
                    maxy,
                    auto_expand=auto_expand,
                    auto_expand_limit=auto_expand_limit,
                )
            )
        return clipped_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def clip(self, geometries, crs=None, all_touched=False, drop=True, invert=False):
        """
        Crops a :class:`xarray.Dataset` by geojson like geometry dicts.

        .. warning:: Only works if all variables in the dataset have the same
                     coordinates.

        Powered by `rasterio.features.geometry_mask`.

        Examples:

            >>> geometry = ''' {"type": "Polygon",
            ...                 "coordinates": [
            ...                 [[-94.07955380199459, 41.69085871273774],
            ...                 [-94.06082436942204, 41.69103313774798],
            ...                 [-94.06063203899649, 41.67932439500822],
            ...                 [-94.07935807746362, 41.679150041277325],
            ...                 [-94.07955380199459, 41.69085871273774]]]}'''
            >>> cropping_geometries = [geojson.loads(geometry)]
            >>> xds = xarray.open_rasterio('cool_raster.tif')
            >>> cropped = xds.rio.clip(geometries=cropping_geometries, crs=4326)

        Parameters
        ----------
        geometries: list
            A list of geojson geometry dicts.
        crs: :obj:`rasterio.crs.CRS`, optional
            The CRS of the input geometries. Default is to assume it is the same
            as the dataset.
        all_touched : boolean, optional
            If True, all pixels touched by geometries will be burned in.  If
            false, only pixels whose center is within the polygon or that
            are selected by Bresenham's line algorithm will be burned in.
        drop: bool, optional
            If True, drop the data outside of the extent of the mask geoemtries
            Otherwise, it will return the same raster with the data masked.
            Default is True.
        invert: boolean, optional
            If False, pixels that do not overlap shapes will be set as nodata.
            Otherwise, pixels that overlap the shapes will be set as nodata.
            False by default.

        Returns
        -------
        :obj:`xarray.Dataset`:
            The clipped object.
        """
        clipped_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            clipped_dataset[var] = (
                self._obj[var]
                .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
                .rio.clip(
                    geometries,
                    crs=crs,
                    all_touched=all_touched,
                    drop=drop,
                    invert=invert,
                )
            )
        return clipped_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def interpolate_na(self, method="nearest"):
        """
        This method uses `scipy.interpolate.griddata` to interpolate missing data.

        Parameters
        ----------
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :obj:`xarray.DataArray`:
             The interpolated object.
        """
        interpolated_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            interpolated_dataset[var] = (
                self._obj[var]
                .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
                .rio.interpolate_na(method=method)
            )
        return interpolated_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def to_raster(
        self,
        raster_path,
        driver="GTiff",
        dtype=None,
        tags=None,
        windowed=False,
        recalc_transform=True,
        **profile_kwargs,
    ):
        """
        Export the Dataset to a raster file. Only works with 2D data.

        Parameters
        ----------
        raster_path: str
            The path to output the raster to.
        driver: str, optional
            The name of the GDAL/rasterio driver to use to export the raster.
            Default is "GTiff".
        dtype: str, optional
            The data type to write the raster to. Default is the datasets dtype.
        tags: dict, optional
            A dictionary of tags to write to the raster.
        windowed: bool, optional
            If True, it will write using the windows of the output raster.
            This only works if the output raster is tiled. As such, if you
            set this to True, the output raster will be tiled.
            Default is False.
        **profile_kwargs
            Additional keyword arguments to pass into writing the raster. The
            nodata, transform, crs, count, width, and height attributes
            are ignored.

        """
        variable_dim = "band_{}".format(uuid4())
        data_array = self._obj.to_array(dim=variable_dim)
        # write data array names to raster
        data_array.attrs["long_name"] = data_array[variable_dim].values.tolist()
        # ensure raster metadata preserved
        scales = []
        offsets = []
        nodatavals = []
        for data_var in data_array[variable_dim].values:
            scales.append(self._obj[data_var].attrs.get("scale_factor", 1.0))
            offsets.append(self._obj[data_var].attrs.get("add_offset", 0.0))
            nodatavals.append(self._obj[data_var].rio.nodata)
        data_array.attrs["scales"] = scales
        data_array.attrs["offsets"] = offsets
        nodata = nodatavals[0]
        if (
            all(nodataval == nodata for nodataval in nodatavals)
            or np.isnan(nodatavals).all()
        ):
            data_array.rio.write_nodata(nodata, inplace=True)
        else:
            raise RioXarrayError(
                "All nodata values must be the same when exporting to raster. "
                "Current values: {}".format(nodatavals)
            )
        if self.crs is not None:
            data_array.rio.write_crs(self.crs, inplace=True)
        # write it to a raster
        data_array.rio.to_raster(
            raster_path=raster_path,
            driver=driver,
            dtype=dtype,
            tags=tags,
            windowed=windowed,
            recalc_transform=recalc_transform,
            **profile_kwargs,
        )
