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
from abc import abstractmethod
from datetime import datetime

import numpy as np
import rasterio.warp
import xarray
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from scipy.interpolate import griddata

from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster
from rioxarray.crs import crs_to_wkt

FILL_VALUE_NAMES = ("_FillValue", "missing_value", "fill_value")
UNWANTED_RIO_ATTRS = ("nodata", "nodatavals", "crs", "is_tiled", "res")
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
    x_coords, _ = (np.arange(width) + 0.5, np.zeros(width) + 0.5) * affine
    _, y_coords = (np.zeros(height) + 0.5, np.arange(height) + 0.5) * affine
    return {y_dim: y_coords, x_dim: x_coords}


def _get_grid_map_name(src_data_array):
    """Get the grid map name of the variable."""
    try:
        return src_data_array.attrs["grid_mapping"]
    except KeyError:
        return DEFAULT_GRID_MAP


def _generate_attrs(src_data_array, dst_affine, dst_nodata):

    # add original attributes
    new_attrs = copy.deepcopy(src_data_array.attrs)
    # remove all nodata information
    for unwanted_attr in FILL_VALUE_NAMES + UNWANTED_RIO_ATTRS:
        try:
            del new_attrs[unwanted_attr]
        except KeyError:
            pass

    # add nodata information
    new_attrs["_FillValue"] = src_data_array.encoding.get("_FillValue", dst_nodata)

    # add raster spatial information
    new_attrs["transform"] = tuple(dst_affine)
    new_attrs["grid_mapping"] = _get_grid_map_name(src_data_array)

    return new_attrs


def add_xy_grid_meta(coords):
    """Add x,y metadata to coordinates"""
    # add metadata to x,y coordinates
    if "x" in coords:
        x_coord_attrs = dict(coords["x"].attrs)
        x_coord_attrs["long_name"] = "x coordinate of projection"
        x_coord_attrs["standard_name"] = "projection_x_coordinate"
        coords["x"].attrs = x_coord_attrs
    elif "longitude" in coords:
        x_coord_attrs = dict(coords["longitude"].attrs)
        x_coord_attrs["long_name"] = "longitude"
        x_coord_attrs["standard_name"] = "longitude"
        coords["longitude"].attrs = x_coord_attrs

    if "y" in coords:
        y_coord_attrs = dict(coords["y"].attrs)
        y_coord_attrs["long_name"] = "y coordinate of projection"
        y_coord_attrs["standard_name"] = "projection_y_coordinate"
        coords["y"].attrs = y_coord_attrs
    elif "latitude" in coords:
        x_coord_attrs = dict(coords["latitude"].attrs)
        x_coord_attrs["long_name"] = "latitude"
        x_coord_attrs["standard_name"] = "latitude"
        coords["latitude"].attrs = x_coord_attrs
    return coords


def add_spatial_ref(in_ds, dst_crs, grid_map_name):
    # remove old grid map if exists
    try:
        del in_ds.coords[grid_map_name]
    except KeyError:
        pass

    # add grid mapping variable
    in_ds.coords[grid_map_name] = xarray.Variable((), 0)
    match_proj = crs_to_wkt(CRS.from_user_input(dst_crs))

    grid_map_attrs = dict()
    # add grid mapping variable
    grid_map_attrs["spatial_ref"] = match_proj
    # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#appendix-grid-mappings
    # http://desktop.arcgis.com/en/arcmap/10.3/manage-data/netcdf/spatial-reference-for-netcdf-data.htm
    grid_map_attrs["crs_wkt"] = match_proj
    in_ds.coords[grid_map_name].attrs = grid_map_attrs
    return in_ds


def _add_attrs_proj(new_data_array, src_data_array):
    """Make sure attributes and projection correct"""
    # make sure attributes preserved
    new_attrs = _generate_attrs(
        src_data_array,
        new_data_array.rio.transform(recalc=True),
        new_data_array.rio.nodata,
    )
    if new_data_array.rio.nodata is None:
        try:
            new_attrs["_FillValue"] = src_data_array.attrs["nodata"]
        except KeyError:
            pass
        try:
            new_attrs["_FillValue"] = src_data_array.attrs["nodatavals"][0]
        except KeyError:
            pass
    # remove fill value if it already exists in the encoding
    # this is for data arrays pulling the encoding from a
    # source data array instead of being generated anew.
    if "_FillValue" in new_data_array.encoding:
        del new_attrs["_FillValue"]

    new_data_array.attrs = new_attrs

    # make sure projection added
    add_xy_grid_meta(new_data_array.coords)
    return add_spatial_ref(
        new_data_array, src_data_array.rio.crs, _get_grid_map_name(src_data_array)
    )


def _warp_spatial_coords(data_array, affine, width, height):
    """get spatial coords in new transform"""
    new_spatial_coords = affine_to_coords(affine, width, height)
    return {
        "x": xarray.IndexVariable("x", new_spatial_coords["x"]),
        "y": xarray.IndexVariable("y", new_spatial_coords["y"]),
    }


def _make_coords(src_data_array, dst_affine, dst_width, dst_height, dst_crs):
    """Generate the coordinates of the new projected `xarray.DataArray`"""
    # step 1: collect old nonspatial coordinates
    coords = {}
    for coord in set(src_data_array.coords) - {
        src_data_array.rio.x_dim,
        src_data_array.rio.y_dim,
        "spatial_ref",
    }:
        coords[coord] = xarray.IndexVariable(
            src_data_array[coord].dims,
            src_data_array[coord].values,
            src_data_array[coord].attrs,
        )
    new_coords = _warp_spatial_coords(src_data_array, dst_affine, dst_width, dst_height)
    new_coords.update(coords)
    return add_xy_grid_meta(new_coords)


def _make_dst_affine(src_data_array, src_crs, dst_crs, dst_resolution=None):
    """Determine the affine of the new projected `xarray.DataArray`"""
    src_bounds = src_data_array.rio.bounds()
    src_width, src_height = src_data_array.rio.shape
    dst_affine, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds, resolution=dst_resolution
    )
    return dst_affine, dst_width, dst_height


class XRasterBase(object):
    """This is the base class for the GIS extensions for xarray"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        # Determine the spatial dimensions of the `xarray.DataArray`
        if "x" in self._obj.dims and "y" in self._obj.dims:
            self.x_dim = "x"
            self.y_dim = "y"
        elif "longitude" in self._obj.dims and "latitude" in self._obj.dims:
            self.x_dim = "longitude"
            self.y_dim = "latitude"
        else:
            raise KeyError("Missing x,y dimensions ...")

        # properties
        self._shape = None
        self._crs = None

    @property
    @abstractmethod
    def crs(self):
        """:obj:`rasterio.crs.CRS`:
            The projection of the dataset.
        """
        return self._crs

    @property
    def shape(self):
        """tuple: Returns the shape (x_size, y_size)"""
        if self._shape is None:
            self._shape = self._obj[self.x_dim].size, self._obj[self.y_dim].size
        return self._shape


@xarray.register_dataarray_accessor("rio")
class RasterArray(XRasterBase):
    """This is the GIS extension for :class:`xarray.DataArray`"""

    def __init__(self, xarray_obj):
        super(RasterArray, self).__init__(xarray_obj)
        # properties
        self._nodata = None

    @property
    def crs(self):
        """:obj:`rasterio.crs.CRS`:
            Retrieve projection from `xarray.DataArray`
        """
        if self._crs is not None:
            return self._crs

        try:
            # look in grid_mapping
            grid_mapping_coord = self._obj.attrs["grid_mapping"]
            self._crs = CRS.from_user_input(
                self._obj.coords[grid_mapping_coord].attrs["spatial_ref"]
            )
        except KeyError:
            # look in attrs for 'crs' from rasterio xarray
            self._crs = CRS.from_user_input(self._obj.attrs["crs"])
        return self._crs

    @property
    def nodata(self):
        """Get the nodata value for the dataset."""
        if self._nodata is not None:
            return self._nodata

        if self._obj.encoding.get("_FillValue") is not None:
            self._nodata = np.nan
        else:
            self._nodata = self._obj.attrs.get(
                "_FillValue",
                self._obj.attrs.get("missing_value", self._obj.attrs.get("fill_value")),
            )
        return self._nodata

    def resolution(self, recalc=False):
        """Determine the resolution of the `xarray.DataArray`

        Parameters
        ----------
        recalc: bool, optional
            Will force the resolution to be recalculated instead of using the
            transform attribute.

        """
        width, height = self.shape
        if not recalc or width == 1 or height == 1:
            try:
                # get resolution from xarray rasterio
                data_transform = Affine(*self._obj.attrs["transform"][:6])
                resolution_x = data_transform.a
                resolution_y = data_transform.e
                recalc = False
            except KeyError:
                recalc = True

        if recalc:
            left, bottom, right, top = self._int_bounds()

            if width == 1 or height == 1:
                raise ValueError(
                    "Only 1 dimenional array found. Cannot calculate the resolution."
                )

            resolution_x = (right - left) / (width - 1)
            resolution_y = (bottom - top) / (height - 1)

        return resolution_x, resolution_y

    def _int_bounds(self):
        """Determine the internal bounds of the `xarray.DataArray`"""
        left = float(self._obj[self.x_dim][0])
        right = float(self._obj[self.x_dim][-1])
        top = float(self._obj[self.y_dim][0])
        bottom = float(self._obj[self.y_dim][-1])
        return left, bottom, right, top

    def bounds(self, recalc=False):
        """Determine the bounds of the `xarray.DataArray`

        Parameters
        ----------
        recalc: bool, optional
            Will force the bounds to be recalculated instead of using the
            transform attribute.

        Returns
        -------
        left, bottom, right, top: float
            Outermost coordinates.
        """
        left, bottom, right, top = self._int_bounds()
        src_resolution_x, src_resolution_y = self.resolution(recalc=recalc)
        left -= src_resolution_x / 2.0
        right += src_resolution_x / 2.0
        top -= src_resolution_y / 2.0
        bottom += src_resolution_y / 2.0
        return left, bottom, right, top

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

    def transform(self, recalc=False):
        """Determine the affine of the `xarray.DataArray`"""
        if not recalc:
            try:
                # get affine from xarray rasterio
                return Affine(*self._obj.attrs["transform"][:6])
            except KeyError:
                pass
        src_bounds = self.bounds(recalc=recalc)
        src_left, _, _, src_top = src_bounds
        src_resolution_x, src_resolution_y = self.resolution(recalc=recalc)
        return Affine.translation(src_left, src_top) * Affine.scale(
            src_resolution_x, src_resolution_y
        )

    def reproject(
        self,
        dst_crs,
        resolution=None,
        dst_affine_width_height=None,
        resampling=Resampling.nearest,
    ):
        """
        Reproject :class:`xarray.DataArray` objects

        Powered by `rasterio.warp.reproject`

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        Parameters
        ----------
        dst_crs: str
            OGC WKT string or Proj.4 string.
        resolution: float or tuple(float, float), optional
            Size of a destination pixel in destination projection units
            (e.g. degrees or metres).
        dst_affine_width_height: tuple(dst_affine, dst_width, dst_height), optional
            Tuple with the destination affine, width, and height.
        resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.


        Returns
        -------
        :class:`xarray.DataArray`: A reprojected DataArray.

        """
        # TODO: Support lazy loading of data with dask imperative function
        src_data = np.copy(self._obj.load().data)

        src_affine = self.transform()
        if dst_affine_width_height is not None:
            dst_affine, dst_width, dst_height = dst_affine_width_height
        else:
            dst_affine, dst_width, dst_height = _make_dst_affine(
                self._obj, self.crs, dst_crs, resolution
            )
        extra_dims = list(set(list(self._obj.dims)) - set([self.x_dim, self.y_dim]))
        if len(extra_dims) > 1:
            raise RuntimeError("Reproject only supports 2D and 3D datasets.")
        if extra_dims:
            assert self._obj.dims == (extra_dims[0], self.y_dim, self.x_dim)
            dst_data = np.zeros(
                (self._obj[extra_dims[0]].size, dst_height, dst_width),
                dtype=self._obj.dtype.type,
            )
        else:
            assert self._obj.dims == (self.y_dim, self.x_dim)
            dst_data = np.zeros((dst_height, dst_width), dtype=self._obj.dtype.type)

        try:
            dst_nodata = self._obj.dtype.type(self.nodata or -9999)
        except ValueError:
            # if integer, set nodata to -9999
            dst_nodata = self._obj.dtype.type(-9999)

        src_nodata = self._obj.dtype.type(self._obj.attrs.get("nodata", dst_nodata))
        rasterio.warp.reproject(
            source=src_data,
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
        new_attrs = _generate_attrs(self._obj, dst_affine, dst_nodata)
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
            coords=_make_coords(self._obj, dst_affine, dst_width, dst_height, dst_crs),
            dims=tuple(dst_dims),
            attrs=new_attrs,
        )
        return add_spatial_ref(xda, dst_crs, DEFAULT_GRID_MAP)

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
        match_data_array: :obj:`xarray.DataArray`
            DataArray of the target resolution and projection.
        resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.


        Returns
        --------
        :obj:`xarray.DataArray`
            Contains the data from the src_data_array, reprojected to match
            match_data_array.

        """
        dst_crs = crs_to_wkt(match_data_array.rio.crs)
        dst_affine = match_data_array.rio.transform()
        dst_width, dst_height = match_data_array.rio.shape

        return self.reproject(
            dst_crs,
            dst_affine_width_height=(dst_affine, dst_width, dst_height),
            resampling=resampling,
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
        DataArray: A sliced :class:`xarray.DataArray` object.

        """
        if self._obj.y[0] > self._obj.y[-1]:
            y_slice = slice(maxy, miny)
        else:
            y_slice = slice(miny, maxy)

        if self._obj.x[0] > self._obj.x[-1]:
            x_slice = slice(maxx, minx)
        else:
            x_slice = slice(minx, maxx)

        return self._obj.sel(x=x_slice, y=y_slice)

    def clip_box(self, minx, miny, maxx, maxy, auto_expand=False, auto_expand_limit=3):
        """Clip the :class:`xarray.DataArray` by a bounding box.

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
        DataArray: A clipped :class:`xarray.DataArray` object.

        """
        if self._obj.coords["x"].size == 1 or self._obj.coords["y"].size == 1:
            raise OneDimensionalRaster(
                "At least one of the raster x,y coordinates" " has only one point."
            )

        resolution_x, resolution_y = self.resolution()

        clip_minx = minx - abs(resolution_x) / 2.0
        clip_miny = miny - abs(resolution_y) / 2.0
        clip_maxx = maxx + abs(resolution_x) / 2.0
        clip_maxy = maxy + abs(resolution_y) / 2.0

        cl_array = self.slice_xy(clip_minx, clip_miny, clip_maxx, clip_maxy)

        if cl_array.coords["x"].size < 1 or cl_array.coords["y"].size < 1:
            raise NoDataInBounds("No data found in bounds.")

        if cl_array.coords["x"].size == 1 or cl_array.coords["y"].size == 1:
            if auto_expand and auto_expand < auto_expand_limit:
                return self.clip_box(
                    clip_minx,
                    clip_miny,
                    clip_maxx,
                    clip_maxy,
                    auto_expand=int(auto_expand) + 1,
                    auto_expand_limit=auto_expand_limit,
                )
            raise OneDimensionalRaster(
                "At least one of the clipped raster x,y coordinates"
                " has only one point."
            )

        # make sure correct attributes preserved & projection added
        _add_attrs_proj(cl_array, self._obj)

        return cl_array

    def clip(self, geometries, crs, all_touched=False):
        """
        Crops a :class:`xarray.DataArray` by geojson like geometry dicts.

        Powered by `rasterio.features.geometry_mask`.

        Parameters
        ----------
        geometries: list
            A list of geojson geometry dicts.
        crs: :obj:`rasterio.crs.CRS`
            The CRS of the input geometries.
        all_touched : boolean, optional
            If True, all pixels touched by geometries will be burned in.  If
            false, only pixels whose center is within the polygon or that
            are selected by Bresenham's line algorithm will be burned in.

        Returns
        -------
        DataArray: A clipped :class:`xarray.DataArray` object.


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
        """
        geometries = [
            rasterio.warp.transform_geom(self.crs, CRS.from_user_input(crs), geometry)
            for geometry in geometries
        ]

        width, height = self.shape
        clip_mask_arr = geometry_mask(
            geometries=geometries,
            out_shape=(int(height), int(width)),
            transform=self.transform(),
            invert=True,
            all_touched=all_touched,
        )
        clip_mask_xray = xarray.DataArray(
            clip_mask_arr,
            coords={
                self.y_dim: self._obj.coords[self.y_dim],
                self.x_dim: self._obj.coords[self.x_dim],
            },
            dims=(self.y_dim, self.x_dim),
        )

        cropped_ds = self._obj.where(clip_mask_xray, drop=True).astype(self._obj.dtype)

        if (
            cropped_ds.coords[self.x_dim].size < 1
            or cropped_ds.coords[self.y_dim].size < 1
        ):
            raise NoDataInBounds("No data found in bounds.")

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
        :class:`numpy.ndarray`: An interpolated :class:`numpy.ndarray`.

        """
        src_data_flat = np.copy(src_data).flatten()
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
        :class:`xarray.DataArray`: An interpolated :class:`xarray.DataArray` object.

        """
        extra_dims = list(set(list(self._obj.dims)) - set([self.x_dim, self.y_dim]))
        if len(extra_dims) > 1:
            raise RuntimeError("Interpolate only supports 2D and 3D datasets.")
        if extra_dims:
            assert self._obj.dims == (extra_dims[0], self.y_dim, self.x_dim)
            interp_data = []
            for _, sub_xds in self._obj.groupby(extra_dims[0]):
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

        # make sure correct attributes preserved & projection added
        _add_attrs_proj(interp_array, self._obj)

        return interp_array


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
        if self._crs is None:
            self._crs = self._obj[self.vars[0]].rio.crs
        return self._crs

    def reproject(
        self,
        dst_crs,
        resolution=None,
        dst_affine_width_height=None,
        resampling=Resampling.nearest,
    ):
        """
        Reproject :class:`xarray.Dataset` objects

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.


        Parameters
        ----------
        dst_crs: str
            OGC WKT string or Proj.4 string.
        resolution: float or tuple(float, float), optional
            Size of a destination pixel in destination projection units
            (e.g. degrees or metres).
        dst_affine_width_height: tuple(dst_affine, dst_width, dst_height), optional
            Tuple with the destination affine, width, and height.
        resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.


        Returns
        --------
        :class:`xarray.Dataset`: A reprojected Dataset.

        """
        resampled_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            resampled_dataset[var] = self._obj[var].rio.reproject(
                dst_crs,
                resolution=resolution,
                dst_affine_width_height=dst_affine_width_height,
                resampling=resampling,
            )
        resampled_dataset.attrs["creation_date"] = str(datetime.utcnow())
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
        match_data_array: :obj:`xarray.DataArray`
            DataArray of the target resolution and projection.
        resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.


        Returns
        --------
        :obj:`xarray.Dataset`
            Contains the data from the src_data_array,
            reprojected to match match_data_array.
        """
        resampled_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            resampled_dataset[var] = self._obj[var].rio.reproject_match(
                match_data_array, resampling=resampling
            )
        resampled_dataset.attrs["creation_date"] = str(datetime.utcnow())
        return resampled_dataset

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
        DataArray: A clipped :class:`xarray.Dataset` object.

        """
        clipped_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            clipped_dataset[var] = self._obj[var].rio.clip_box(
                minx,
                miny,
                maxx,
                maxy,
                auto_expand=auto_expand,
                auto_expand_limit=auto_expand_limit,
            )
        clipped_dataset.attrs["creation_date"] = str(datetime.utcnow())
        return clipped_dataset

    def clip(self, geometries, crs, all_touched=False):
        """
        Crops a :class:`xarray.Dataset` by geojson like geometry dicts.

        .. warning:: Only works if all variables in the dataset have the same
                     coordinates.

        Powered by `rasterio.features.geometry_mask`.

        Parameters
        ----------
        geometries: list
            A list of geojson geometry dicts.
        crs: :obj:`rasterio.crs.CRS`
            The CRS of the input geometries.
        all_touched : boolean, optional
            If True, all pixels touched by geometries will be burned in.  If
            false, only pixels whose center is within the polygon or that
            are selected by Bresenham's line algorithm will be burned in.

        Returns
        -------
        Dataset: A clipped :class:`xarray.Dataset` object.


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
        """
        clipped_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            clipped_dataset[var] = self._obj[var].rio.clip(
                geometries, crs=crs, all_touched=all_touched
            )
        clipped_dataset.attrs["creation_date"] = str(datetime.utcnow())
        return clipped_dataset

    def interpolate_na(self, method="nearest"):
        """
        This method uses `scipy.interpolate.griddata` to interpolate missing data.

        Parameters
        ----------
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :class:`xarray.DataArray`: An interpolated :class:`xarray.DataArray` object.

        """
        interpolated_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            interpolated_dataset[var] = self._obj[var].rio.interpolate_na(method=method)
        interpolated_dataset.attrs["creation_date"] = str(datetime.utcnow())
        return interpolated_dataset
