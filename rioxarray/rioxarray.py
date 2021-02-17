"""
This module is an extension for xarray to provide rasterio capabilities
to xarray datasets/dataarrays.
"""
import math

import numpy as np
import pyproj
import rasterio.warp
import rasterio.windows
import xarray
from affine import Affine
from rasterio.crs import CRS

from rioxarray._options import EXPORT_GRID_MAPPING, get_option
from rioxarray.crs import crs_from_user_input
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


def _warp_spatial_coords(affine, width, height):
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
    new_coords = _warp_spatial_coords(dst_affine, dst_width, dst_height)
    new_coords.update(coords)
    return new_coords


def _get_data_var_message(obj):
    """
    Get message for named data variables.
    """
    try:
        return f" Data variable: {obj.name}" if obj.name else ""
    except AttributeError:
        return ""


class XRasterBase:
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
                if (self._obj.coords[coord].attrs.get("axis", "").upper() == "X") or (
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

        # look in wkt attributes to avoid using
        # pyproj CRS if possible for performance
        for crs_attr in ("spatial_ref", "crs_wkt"):
            try:
                self.set_crs(
                    self._obj.coords[self.grid_mapping].attrs[crs_attr],
                    inplace=True,
                )
                return self._crs
            except KeyError:
                pass

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
        crs = crs_from_user_input(input_crs)
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
                try:
                    # pylint: disable=pointless-statement
                    self._obj[var].rio.x_dim
                    self._obj[var].rio.y_dim
                except DimensionError:
                    continue
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
                try:
                    x_dim = data_obj[var].rio.x_dim
                    y_dim = data_obj[var].rio.y_dim
                except DimensionError:
                    continue

                data_obj[var].rio.update_attrs(
                    dict(grid_mapping=grid_mapping_name), inplace=True
                ).rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
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
                "CRS not found. Please set the CRS with 'rio.write_crs()'."
            )
        # add grid mapping coordinate
        data_obj.coords[grid_mapping_name] = xarray.Variable((), 0)
        if get_option(EXPORT_GRID_MAPPING):
            grid_map_attrs = pyproj.CRS.from_user_input(data_obj.rio.crs).to_cf()
        else:
            grid_map_attrs = {}
        # spatial_ref is for compatibility with GDAL
        crs_wkt = data_obj.rio.crs.to_wkt()
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

    def estimate_utm_crs(self, datum_name="WGS 84"):
        """Returns the estimated UTM CRS based on the bounds of the dataset.

        .. versionadded:: 0.2

        .. note:: Requires pyproj 3+

        Parameters
        ----------
        datum_name : str, optional
            The name of the datum to use in the query. Default is WGS 84.

        Returns
        -------
        rasterio.crs.CRS
        """
        # pylint: disable=import-outside-toplevel
        try:
            from pyproj.aoi import AreaOfInterest
            from pyproj.database import query_utm_crs_info
        except ImportError:
            raise RuntimeError("pyproj 3+ required for estimate_utm_crs.") from None

        if self.crs is None:
            raise RuntimeError("crs must be set to estimate UTM CRS.")

        # ensure using geographic coordinates
        if self.crs.is_geographic:
            minx, miny, maxx, maxy = self.bounds(recalc=True)
        else:
            minx, miny, maxx, maxy = self.transform_bounds("EPSG:4326", recalc=True)

        x_center = np.mean([minx, maxx])
        y_center = np.mean([miny, maxy])

        utm_crs_list = query_utm_crs_info(
            datum_name=datum_name,
            area_of_interest=AreaOfInterest(
                west_lon_degree=x_center,
                south_lat_degree=y_center,
                east_lon_degree=x_center,
                north_lat_degree=y_center,
            ),
        )
        try:
            return CRS.from_epsg(utm_crs_list[0].code)
        except IndexError:
            raise RuntimeError("Unable to determine UTM CRS") from None

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

        data_obj = self._get_obj(inplace=inplace)
        if x_dim in data_obj.dims:
            data_obj.rio._x_dim = x_dim
        else:
            raise DimensionError(
                f"x dimension ({x_dim}) not found.{_get_data_var_message(data_obj)}"
            )
        if y_dim in data_obj.dims:
            data_obj.rio._y_dim = y_dim
        else:
            raise DimensionError(
                f"y dimension ({y_dim}) not found.{_get_data_var_message(data_obj)}"
            )
        return data_obj

    @property
    def x_dim(self):
        """str: The dimension for the X-axis."""
        if self._x_dim is not None:
            return self._x_dim
        raise DimensionError(
            "x dimension not found. 'rio.set_spatial_dims()' or "
            "using 'rename()' to change the dimension name to 'x' can address this."
            f"{_get_data_var_message(self._obj)}"
        )

    @property
    def y_dim(self):
        """str: The dimension for the Y-axis."""
        if self._y_dim is not None:
            return self._y_dim
        raise DimensionError(
            "y dimension not found. 'rio.set_spatial_dims()' or "
            "using 'rename()' to change the dimension name to 'y' can address this."
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
        if extra_dims and self._obj.dims != (extra_dims[0], self.y_dim, self.x_dim):
            raise InvalidDimensionOrder(
                "Invalid dimension order. Expected order: {0}. "
                "You can use `DataArray.transpose{0}`"
                " to reorder your dimensions.".format(
                    (extra_dims[0], self.y_dim, self.x_dim)
                )
                + f"{_get_data_var_message(self._obj)}"
            )
        if not extra_dims and self._obj.dims != (self.y_dim, self.x_dim):
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
        if self.y_dim not in self._obj.coords:
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
            ) from None
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
