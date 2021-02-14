"""
This module is an extension for xarray to provide rasterio capabilities
to xarray dataarrays.

Credits: The `reproject` functionality was adopted from https://github.com/opendatacube/datacube-core # noqa: E501
Source file:
- https://github.com/opendatacube/datacube-core/blob/084c84d78cb6e1326c7fbbe79c5b5d0bef37c078/datacube/api/geo_xarray.py  # noqa: E501
datacube is licensed under the Apache License, Version 2.0:
- https://github.com/opendatacube/datacube-core/blob/1d345f08a10a13c316f81100936b0ad8b1a374eb/LICENSE  # noqa: E501

"""
import copy
import warnings
from distutils.version import LooseVersion
from typing import Iterable

import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp
import xarray
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from scipy.interpolate import griddata

from rioxarray.crs import crs_from_user_input
from rioxarray.exceptions import (
    MissingCRS,
    NoDataInBounds,
    OneDimensionalRaster,
    RioXarrayError,
)
from rioxarray.raster_writer import FILL_VALUE_NAMES, UNWANTED_RIO_ATTRS, RasterioWriter
from rioxarray.rioxarray import XRasterBase, _get_data_var_message, _make_coords


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


def _make_dst_affine(
    src_data_array, src_crs, dst_crs, dst_resolution=None, dst_shape=None
):
    """Determine the affine of the new projected `xarray.DataArray`"""
    src_bounds = src_data_array.rio.bounds()
    src_height, src_width = src_data_array.rio.shape
    dst_height, dst_width = dst_shape if dst_shape is not None else (None, None)
    # pylint: disable=isinstance-second-argument-not-valid-type
    if isinstance(dst_resolution, Iterable):
        dst_resolution = tuple(abs(res_val) for res_val in dst_resolution)
    elif dst_resolution is not None:
        dst_resolution = abs(dst_resolution)
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


def _clip_from_disk(xds, geometries, all_touched, drop, invert):
    """
    clip from disk if the file object is available
    """
    try:
        out_image, out_transform = rasterio.mask.mask(
            xds._file_obj.acquire(),
            geometries,
            all_touched=all_touched,
            invert=invert,
            crop=drop,
        )
        if xds.rio.encoded_nodata is not None and not np.isnan(xds.rio.encoded_nodata):
            out_image = out_image.astype(np.float64)
            out_image[out_image == xds.rio.encoded_nodata] = np.nan

        height, width = out_image.shape[-2:]
        cropped_ds = xarray.DataArray(
            name=xds.name,
            data=out_image,
            coords=_make_coords(xds, out_transform, width, height),
            dims=xds.dims,
            attrs=xds.attrs,
        )
        cropped_ds.encoding = xds.encoding
        return cropped_ds
    except AttributeError:
        return None


def _clip_xarray(xds, geometries, all_touched, drop, invert):
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
                np.ma.masked_array(clip_mask_arr, ~clip_mask_arr)
            )
        )
    if xds.rio.nodata is not None and not np.isnan(xds.rio.nodata):
        cropped_ds = cropped_ds.fillna(xds.rio.nodata)

    return cropped_ds.astype(xds.dtype)


@xarray.register_dataarray_accessor("rio")
class RasterArray(XRasterBase):
    """This is the GIS extension for :obj:`xarray.DataArray`"""

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
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
                "CRS not found. Please set the CRS with 'rio.write_crs()'."
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

        dst_nodata = self._obj.dtype.type(
            self.nodata if self.nodata is not None else -9999
        )
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
        return self.reproject(
            match_data_array.rio.crs,
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
        # pylint: disable=too-many-locals
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
            if cl_array.rio.width == 1 or cl_array.rio.height == 1:
                raise OneDimensionalRaster(
                    "At least one of the clipped raster x,y coordinates"
                    " has only one point."
                    f"{_get_data_var_message(self._obj)}"
                )

        # make sure correct attributes preserved & projection added
        _add_attrs_proj(cl_array, self._obj)
        return cl_array

    def clip(
        self,
        geometries,
        crs=None,
        all_touched=False,
        drop=True,
        invert=False,
        from_disk=False,
    ):
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


        .. versionadded:: 0.2 from_disk

        Parameters
        ----------
        geometries: list
            A list of geojson geometry dicts or objects with __geom_interface__ with
            if you have rasterio 1.2+.
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
        from_disk: boolean, optional
            If True, it will clip from disk using rasterio.mask.mask if possible.
            This is beneficial when the size of the data is larger than memory.
            Default is False.

        Returns
        -------
        :obj:`xarray.DataArray`:
            The clipped object.
        """
        if self.crs is None:
            raise MissingCRS(
                "CRS not found. Please set the CRS with 'rio.write_crs()'."
                f"{_get_data_var_message(self._obj)}"
            )
        crs = crs_from_user_input(crs) if crs is not None else self.crs
        if self.crs != crs:
            if LooseVersion(rasterio.__version__) >= LooseVersion("1.2"):
                geometries = rasterio.warp.transform_geom(crs, self.crs, geometries)
            else:
                geometries = [
                    rasterio.warp.transform_geom(crs, self.crs, geometry)
                    for geometry in geometries
                ]
        cropped_ds = None
        if from_disk:
            cropped_ds = _clip_from_disk(
                self._obj,
                geometries=geometries,
                all_touched=all_touched,
                drop=drop,
                invert=invert,
            )
        if cropped_ds is None:
            cropped_ds = _clip_xarray(
                self._obj,
                geometries=geometries,
                all_touched=all_touched,
                drop=drop,
                invert=invert,
            )

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
        driver=None,
        dtype=None,
        tags=None,
        windowed=False,
        recalc_transform=True,
        lock=None,
        compute=True,
        **profile_kwargs,
    ):
        """
        Export the DataArray to a raster file.

        ..versionadded:: 0.2 lock

        Parameters
        ----------
        raster_path: str
            The path to output the raster to.
        driver: str, optional
            The name of the GDAL/rasterio driver to use to export the raster.
            Default is "GTiff" if rasterio < 1.2 otherwise it will autodetect.
        dtype: str, optional
            The data type to write the raster to. Default is the datasets dtype.
        tags: dict, optional
            A dictionary of tags to write to the raster.
        windowed: bool, optional
            If True, it will write using the windows of the output raster.
            This is useful for loading data in chunks when writing. Does not
            do anything when writing with dask.
            Default is False.
        lock: boolean or Lock, optional
            Lock to use to write data using dask.
            If not supplied, it will use a single process for writing.
        compute: bool, optional
            If True and data is a dask array, then compute and save
            the data immediately. If False, return a dask Delayed object.
            Call ".compute()" on the Delayed object to compute the result
            later. Call ``dask.compute(delayed1, delayed2)`` to save
            multiple delayed files at once. Default is True.
        **profile_kwargs
            Additional keyword arguments to pass into writing the raster. The
            nodata, transform, crs, count, width, and height attributes
            are ignored.

        Returns
        -------
        :obj:`dask.Delayed`:
            If the data array is a dask array and compute
            is True. Otherwise None is returned.

        """
        if driver is None and LooseVersion(rasterio.__version__) < LooseVersion("1.2"):
            driver = "GTiff"

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

        return RasterioWriter(raster_path=raster_path).to_raster(
            xarray_dataarray=self._obj,
            tags=tags,
            driver=driver,
            height=int(self.height),
            width=int(self.width),
            count=int(self.count),
            dtype=dtype,
            crs=self.crs,
            transform=self.transform(recalc=recalc_transform),
            nodata=rio_nodata,
            windowed=windowed,
            lock=lock,
            compute=compute,
            **out_profile,
        )
