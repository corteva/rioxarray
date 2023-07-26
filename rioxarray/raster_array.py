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
import os
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy
import rasterio
import rasterio.mask
import rasterio.warp
import xarray
from affine import Affine
from rasterio.dtypes import dtype_rev
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from xarray.backends.file_manager import FileManager
from xarray.core.dtypes import get_fill_value

from rioxarray._io import FILL_VALUE_NAMES, UNWANTED_RIO_ATTRS
from rioxarray.crs import crs_from_user_input
from rioxarray.exceptions import (
    MissingCRS,
    NoDataInBounds,
    OneDimensionalRaster,
    RioXarrayError,
)
from rioxarray.raster_writer import RasterioWriter, _ensure_nodata_dtype
from rioxarray.rioxarray import (
    XRasterBase,
    _get_data_var_message,
    _make_coords,
    _order_bounds,
)

# DTYPE TO NODATA MAP
# Based on: https://github.com/OSGeo/gdal/blob/
# dee861e7c91c2da7ef8ff849947713e4d9bd115c/
# swig/python/gdal-utils/osgeo_utils/gdal_calc.py#L61
_NODATA_DTYPE_MAP = {
    1: 255,  # GDT_Byte
    2: 65535,  # GDT_UInt16
    3: -32768,  # GDT_Int16
    4: 4294967293,  # GDT_UInt32
    5: -2147483647,  # GDT_Int32
    6: 3.402823466e38,  # GDT_Float32
    7: 1.7976931348623158e308,  # GDT_Float64
    8: None,  # GDT_CInt16
    9: None,  # GDT_CInt32
    10: 3.402823466e38,  # GDT_CFloat32
    11: 1.7976931348623158e308,  # GDT_CFloat64
    12: None,  # GDT_Int64
    13: None,  # GDT_UInt64
    14: None,  # GDT_Int8
}


def _generate_attrs(
    src_data_array: xarray.DataArray, dst_nodata: Optional[float]
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
    new_data_array: xarray.DataArray, src_data_array: xarray.DataArray
) -> xarray.DataArray:
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
    new_data_array.rio.write_grid_mapping(src_data_array.rio.grid_mapping, inplace=True)
    new_data_array.rio.write_crs(src_data_array.rio.crs, inplace=True)
    new_data_array.rio.write_coordinate_system(inplace=True)
    new_data_array.rio.write_transform(inplace=True)
    # make sure encoding added
    new_data_array.encoding = src_data_array.encoding.copy()
    return new_data_array


def _make_dst_affine(
    src_data_array: xarray.DataArray,
    src_crs: rasterio.crs.CRS,
    dst_crs: rasterio.crs.CRS,
    dst_resolution: Optional[Union[float, tuple[float, float]]] = None,
    dst_shape: Optional[tuple[float, float]] = None,
    **kwargs,
):
    """Determine the affine of the new projected `xarray.DataArray`"""
    src_bounds = () if "gcps" in kwargs else src_data_array.rio.bounds()
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
    geometries: Iterable,
    all_touched: bool,
    drop: bool,
    invert: bool,
) -> Optional[xarray.DataArray]:
    """
    clip from disk if the file object is available
    """
    try:
        out_image, out_transform = rasterio.mask.mask(
            xds.rio._manager.acquire(),
            geometries,
            all_touched=all_touched,
            invert=invert,
            crop=drop,
        )
        if xds.rio.encoded_nodata is not None and not numpy.isnan(
            xds.rio.encoded_nodata
        ):
            out_image = out_image.astype(numpy.float64)
            out_image[out_image == xds.rio.encoded_nodata] = numpy.nan

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


def _clip_xarray(
    xds: xarray.DataArray,
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


@xarray.register_dataarray_accessor("rio")
class RasterArray(XRasterBase):
    """This is the GIS extension for :obj:`xarray.DataArray`"""

    def __init__(self, xarray_obj: xarray.DataArray):
        super().__init__(xarray_obj)
        self._obj: xarray.DataArray
        # properties
        self._nodata: Optional[float] = None
        self._manager: Optional[
            FileManager
        ] = None  # https://github.com/corteva/rioxarray/issues/254

    def set_nodata(
        self, input_nodata: Optional[float], inplace: bool = True
    ) -> xarray.DataArray:
        """
        Set the nodata value for the DataArray without modifying
        the data array.

        Parameters
        ----------
        input_nodata: Optional[float]
            Valid nodata for dtype.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is True.

        Returns
        -------
        :obj:`xarray.DataArray`:
            Dataset with nodata attribute set.
        """
        obj: xarray.DataArray = self._get_obj(inplace=inplace)  # type: ignore
        obj.rio._nodata = input_nodata
        return obj

    def write_nodata(
        self, input_nodata: Optional[float], encoded: bool = False, inplace=False
    ) -> xarray.DataArray:
        """
        Write the nodata to the DataArray in a CF compliant manner.

        Parameters
        ----------
        input_nodata: Optional[float]
            Nodata value for the DataArray.
            If input_nodata is None, it will remove the _FillValue attribute.
        encoded: bool, optional
            If True, it will write the nodata value in the encoding and remove
            the fill value from the attributes. This is useful for masking
            with nodata. Default is False.
        inplace: bool, optional
            If True, it will write to the existing DataArray. Default is False.

        Returns
        -------
        :obj:`xarray.DataArray`:
            Modified DataArray with CF compliant nodata information.

        Examples
        --------
        To write the nodata value if it is missing:

        >>> raster.rio.write_nodata(-9999, inplace=True)

        To write the nodata value on a copy:

        >>> raster = raster.rio.write_nodata(-9999)

        To mask with nodata:

        >>> nodata = raster.rio.nodata
        >>> raster = raster.where(raster != nodata)
        >>> raster.rio.write_nodata(nodata, encoded=True, inplace=True)

        """
        data_obj: xarray.DataArray = self._get_obj(inplace=inplace)  # type: ignore
        input_nodata = False if input_nodata is None else input_nodata
        if input_nodata is not False:
            input_nodata = _ensure_nodata_dtype(input_nodata, self._obj.dtype)
            if encoded:
                data_obj.rio.update_encoding({"_FillValue": input_nodata}, inplace=True)
            else:
                data_obj.rio.update_attrs({"_FillValue": input_nodata}, inplace=True)
        if input_nodata is False or encoded:
            new_attrs = dict(data_obj.attrs)
            new_attrs.pop("_FillValue", None)
            data_obj.rio.set_attrs(new_attrs, inplace=True)
        if input_nodata is False and encoded:
            new_encoding = dict(data_obj.encoding)
            new_encoding.pop("_FillValue", None)
            data_obj.rio.set_encoding(new_encoding, inplace=True)
        if not encoded:
            data_obj.rio.set_nodata(input_nodata, inplace=True)
        return data_obj

    @property
    def encoded_nodata(self) -> Optional[float]:
        """Return the encoded nodata value for the dataset if encoded."""
        encoded_nodata = self._obj.encoding.get("_FillValue")
        if encoded_nodata is None:
            return None
        return _ensure_nodata_dtype(encoded_nodata, self._obj.dtype)

    @property
    def nodata(self) -> Optional[float]:
        """Get the nodata value for the dataset."""
        if self._nodata is not None:
            return None if self._nodata is False else self._nodata

        if self.encoded_nodata is not None:
            self._nodata = get_fill_value(self._obj.dtype)
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
                self._nodata = self._manager.acquire().nodata  # type: ignore
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
        dst_crs: Any,
        resolution: Optional[Union[float, tuple[float, float]]] = None,
        shape: Optional[tuple[int, int]] = None,
        transform: Optional[Affine] = None,
        resampling: Resampling = Resampling.nearest,
        nodata: Optional[float] = None,
        **kwargs,
    ) -> xarray.DataArray:
        """
        Reproject :obj:`xarray.DataArray` objects

        Powered by :func:`rasterio.warp.reproject`

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        .. note:: To re-project with dask, see
            `odc-geo <https://odc-geo.readthedocs.io/>`__ &
            `pyresample <https://pyresample.readthedocs.io/>`__.

        .. versionadded:: 0.0.27 shape
        .. versionadded:: 0.0.28 transform
        .. versionadded:: 0.5.0 nodata, kwargs

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
        transform: Affine, optional
            The destination transform.
        resampling: rasterio.enums.Resampling, optional
            See :func:`rasterio.warp.reproject` for more details.
        nodata: float, optional
            The nodata value used to initialize the destination;
            it will remain in all areas not covered by the reprojected source.
            Defaults to the nodata value of the source image if none provided
            and exists or attempts to find an appropriate value by dtype.
        **kwargs: dict
            Additional keyword arguments to pass into :func:`rasterio.warp.reproject`.
            To override:
            - src_transform: `rio.write_transform`
            - src_crs: `rio.write_crs`
            - src_nodata: `rio.write_nodata`


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
        gcps = self.get_gcps()
        if gcps:
            kwargs.setdefault("gcps", gcps)

        gcps_or_rpcs = "gcps" in kwargs or "rpcs" in kwargs
        src_affine = None if gcps_or_rpcs else self.transform(recalc=True)
        if transform is None:
            dst_affine, dst_width, dst_height = _make_dst_affine(
                self._obj, self.crs, dst_crs, resolution, shape, **kwargs
            )
        else:
            dst_affine = transform
            if shape is not None:
                dst_height, dst_width = shape
            else:
                dst_height, dst_width = self.shape

        dst_data = self._create_dst_data(dst_height, dst_width)

        dst_nodata = self._get_dst_nodata(nodata)

        rasterio.warp.reproject(
            source=self._obj.values,
            destination=dst_data,
            src_transform=src_affine,
            src_crs=self.crs,
            src_nodata=self.nodata,
            dst_transform=dst_affine,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
            **kwargs,
        )
        # add necessary attributes
        new_attrs = _generate_attrs(self._obj, dst_nodata)
        # make sure dimensions with coordinates renamed to x,y
        dst_dims: list[Hashable] = []
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
            coords=_make_coords(
                src_data_array=self._obj,
                dst_affine=dst_affine,
                dst_width=dst_width,
                dst_height=dst_height,
                force_generate=gcps_or_rpcs,
            ),
            dims=tuple(dst_dims),
            attrs=new_attrs,
        )
        xda.encoding = self._obj.encoding
        xda.rio.write_transform(dst_affine, inplace=True)
        xda.rio.write_crs(dst_crs, inplace=True)
        xda.rio.write_coordinate_system(inplace=True)
        return xda

    def _get_dst_nodata(self, nodata: Optional[float]) -> Optional[float]:
        default_nodata = (
            _NODATA_DTYPE_MAP.get(dtype_rev[self._obj.dtype.name])
            if self.nodata is None
            else self.nodata
        )
        dst_nodata = default_nodata if nodata is None else nodata
        return dst_nodata

    def _create_dst_data(self, dst_height: int, dst_width: int) -> numpy.ndarray:
        extra_dim = self._check_dimensions()
        if extra_dim:
            dst_data = numpy.zeros(
                (self._obj[extra_dim].size, dst_height, dst_width),
                dtype=self._obj.dtype.type,
            )
        else:
            dst_data = numpy.zeros((dst_height, dst_width), dtype=self._obj.dtype.type)
        return dst_data

    def reproject_match(
        self,
        match_data_array: Union[xarray.DataArray, xarray.Dataset],
        resampling: Resampling = Resampling.nearest,
        **reproject_kwargs,
    ) -> xarray.DataArray:
        """
        Reproject a DataArray object to match the resolution, projection,
        and region of another DataArray.

        Powered by :func:`rasterio.warp.reproject`

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        .. versionadded:: 0.9 reproject_kwargs

        Parameters
        ----------
        match_data_array:  :obj:`xarray.DataArray` | :obj:`xarray.Dataset`
            DataArray of the target resolution and projection.
        resampling: rasterio.enums.Resampling, optional
            See :func:`rasterio.warp.reproject` for more details.
        **reproject_kwargs:
            Other options to pass to :meth:`rioxarray.raster_array.RasterArray.reproject`

        Returns
        --------
        :obj:`xarray.DataArray`:
            Contains the data from the src_data_array, reprojected to match
            match_data_array.
        """
        reprojected_data_array = self.reproject(
            match_data_array.rio.crs,
            transform=match_data_array.rio.transform(recalc=True),
            shape=match_data_array.rio.shape,
            resampling=resampling,
            **reproject_kwargs,
        )
        # hack to resolve: https://github.com/corteva/rioxarray/issues/298
        # may be resolved in the future by flexible indexes:
        # https://github.com/pydata/xarray/pull/4489#issuecomment-831809607
        x_attrs = reprojected_data_array[reprojected_data_array.rio.x_dim].attrs.copy()
        y_attrs = reprojected_data_array[reprojected_data_array.rio.y_dim].attrs.copy()
        # ensure coords the same
        reprojected_data_array = reprojected_data_array.assign_coords(
            {
                reprojected_data_array.rio.x_dim: copy.copy(
                    match_data_array[match_data_array.rio.x_dim].values
                ),
                reprojected_data_array.rio.y_dim: copy.copy(
                    match_data_array[match_data_array.rio.y_dim].values
                ),
            }
        )
        # ensure attributes copied
        reprojected_data_array[reprojected_data_array.rio.x_dim].attrs = x_attrs
        reprojected_data_array[reprojected_data_array.rio.y_dim].attrs = y_attrs
        return reprojected_data_array

    def pad_xy(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        constant_values: Union[
            float, tuple[int, int], Mapping[Any, tuple[int, int]], None
        ] = None,
    ) -> xarray.DataArray:
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
        constant_values: scalar, tuple or mapping of hashable to tuple
            The value used for padding. If None, nodata will be used if it is
            set, and numpy.nan otherwise.


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
        y_coord: Union[xarray.DataArray, numpy.ndarray] = self._obj[self.y_dim]
        x_coord: Union[xarray.DataArray, numpy.ndarray] = self._obj[self.x_dim]

        if top - resolution_y < maxy:
            new_y_coord: numpy.ndarray = numpy.arange(bottom, maxy, -resolution_y)[::-1]
            y_before = len(new_y_coord) - len(y_coord)
            y_coord = new_y_coord
            top = y_coord[0]
        if bottom + resolution_y > miny:
            new_y_coord = numpy.arange(top, miny, resolution_y)
            y_after = len(new_y_coord) - len(y_coord)
            y_coord = new_y_coord
            bottom = y_coord[-1]

        if left - resolution_x > minx:
            new_x_coord: numpy.ndarray = numpy.arange(right, minx, -resolution_x)[::-1]
            x_before = len(new_x_coord) - len(x_coord)
            x_coord = new_x_coord
            left = x_coord[0]
        if right + resolution_x < maxx:
            new_x_coord = numpy.arange(left, maxx, resolution_x)
            x_after = len(new_x_coord) - len(x_coord)
            x_coord = new_x_coord
            right = x_coord[-1]

        if constant_values is None:
            constant_values = numpy.nan if self.nodata is None else self.nodata

        superset = self._obj.pad(
            pad_width={
                self.x_dim: (x_before, x_after),
                self.y_dim: (y_before, y_after),
            },
            constant_values=constant_values,  # type: ignore
        ).rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
        superset[self.x_dim] = x_coord
        superset[self.y_dim] = y_coord
        superset.rio.write_transform(inplace=True)
        return superset

    def pad_box(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        constant_values: Union[
            float, tuple[int, int], Mapping[Any, tuple[int, int]], None
        ] = None,
    ) -> xarray.DataArray:
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
        constant_values: scalar, tuple or mapping of hashable to tuple
            The value used for padding. If None, nodata will be used if it is
            set, and numpy.nan otherwise.


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

    def clip_box(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        auto_expand: Union[bool, int] = False,
        auto_expand_limit: int = 3,
        crs: Optional[Any] = None,
    ) -> xarray.DataArray:
        """Clip the :obj:`xarray.DataArray` by a bounding box.

        .. versionadded:: 0.12 crs

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
        auto_expand: Union[bool, int]
            If True, it will expand clip search if only 1D raster found with clip.
        auto_expand_limit: int
            maximum number of times the clip will be retried before raising
            an exception.
        crs: :obj:`rasterio.crs.CRS`, optional
            The CRS of the bounding box. Default is to assume it is the same
            as the dataset.

        Returns
        -------
        xarray.DataArray:
            The clipped object.
        """
        if self.width == 1 or self.height == 1:
            raise OneDimensionalRaster(
                "At least one of the raster x,y coordinates has only one point."
                f"{_get_data_var_message(self._obj)}"
            )

        if crs is not None and self.crs is None:
            raise MissingCRS(
                "CRS not found. Please set the CRS with 'rio.write_crs()'."
                f"{_get_data_var_message(self._obj)}"
            )

        crs = crs_from_user_input(crs) if crs is not None else self.crs
        if self.crs != crs:
            minx, miny, maxx, maxy = rasterio.warp.transform_bounds(
                src_crs=crs,
                dst_crs=self.crs,
                left=minx,
                bottom=miny,
                right=maxx,
                top=maxy,
            )
            if (
                self.crs is not None
                and self.crs.is_geographic  # pylint: disable=no-member
                and minx > maxx
            ):
                raise RioXarrayError(
                    "Transformed bounds crossed the antimeridian. "
                    "Please transform your bounds manually using "
                    "rasterio.warp.transform_bounds and clip using "
                    "the bounding box(es) desired."
                )

        resolution_x, resolution_y = self.resolution()
        # make sure that if the coordinates are
        # in reverse order that it still works
        left, bottom, right, top = _order_bounds(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
        )

        # pull the data out
        window_error = None
        try:
            window = rasterio.windows.from_bounds(
                left=numpy.array(left).item(),
                bottom=numpy.array(bottom).item(),
                right=numpy.array(right).item(),
                top=numpy.array(top).item(),
                transform=self.transform(recalc=True),
            )
            cl_array: xarray.DataArray = self.isel_window(window)  # type: ignore
        except rasterio.errors.WindowError as err:
            window_error = err

        # check that the window has data in it
        if window_error or cl_array.rio.width <= 1 or cl_array.rio.height <= 1:
            if auto_expand and auto_expand < auto_expand_limit:
                return self.clip_box(
                    minx=minx - abs(resolution_x) / 2.0,
                    miny=miny - abs(resolution_y) / 2.0,
                    maxx=maxx + abs(resolution_x) / 2.0,
                    maxy=maxy + abs(resolution_y) / 2.0,
                    auto_expand=int(auto_expand) + 1,
                    auto_expand_limit=auto_expand_limit,
                )
            if window_error:
                raise window_error
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
        geometries: Iterable,
        crs: Optional[Any] = None,
        all_touched: bool = False,
        drop: bool = True,
        invert: bool = False,
        from_disk: bool = False,
    ) -> xarray.DataArray:
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
        geometries: Iterable
            A list of geojson geometry dicts or objects with __geo_interface__ with
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
            geometries = rasterio.warp.transform_geom(crs, self.crs, geometries)
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

    def _interpolate_na(
        self, src_data: Any, method: Literal["linear", "nearest", "cubic"] = "nearest"
    ) -> numpy.ndarray:
        """
        This method uses scipy.interpolate.griddata to interpolate missing data.

        Parameters
        ----------
        src_data: Any
            Input data array.
        method: {'linear', 'nearest', 'cubic'}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :class:`numpy.ndarray`:
            An interpolated :class:`numpy.ndarray`.
        """
        try:
            from scipy.interpolate import (  # pylint: disable=import-outside-toplevel,import-error
                griddata,
            )
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "scipy is not found. Use rioxarray[interp] to install."
            ) from err

        src_data_flat = src_data.flatten()
        try:
            data_isnan = numpy.isnan(self.nodata)  # type: ignore
        except TypeError:
            data_isnan = False
        if not data_isnan:
            data_bool = src_data_flat != self.nodata
        else:
            data_bool = ~numpy.isnan(src_data_flat)

        if not data_bool.any():
            return src_data

        x_coords, y_coords = numpy.meshgrid(
            self._obj.coords[self.x_dim].values, self._obj.coords[self.y_dim].values
        )

        return griddata(
            points=(x_coords.flatten()[data_bool], y_coords.flatten()[data_bool]),
            values=src_data_flat[data_bool],
            xi=(x_coords, y_coords),
            method=method,
            fill_value=self.nodata,
        )

    def interpolate_na(
        self, method: Literal["linear", "nearest", "cubic"] = "nearest"
    ) -> xarray.DataArray:
        """
        This method uses scipy.interpolate.griddata to interpolate missing data.

        .. warning:: scipy is an optional dependency.

        Parameters
        ----------
        method: {'linear', 'nearest', 'cubic'}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :obj:`xarray.DataArray`:
            An interpolated :obj:`xarray.DataArray` object.
        """
        if self.nodata is None:
            raise RioXarrayError(
                "nodata not found. Please set the nodata with 'rio.write_nodata()'."
                f"{_get_data_var_message(self._obj)}"
            )

        extra_dim = self._check_dimensions()
        if extra_dim:
            interp_data = []
            for _, sub_xds in self._obj.groupby(extra_dim):
                interp_data.append(
                    self._interpolate_na(sub_xds.load().data, method=method)
                )
            interp_data = numpy.array(interp_data)  # type: ignore
        else:
            interp_data = self._interpolate_na(self._obj.load().data, method=method)  # type: ignore

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
        raster_path: Union[str, os.PathLike],
        driver: Optional[str] = None,
        dtype: Optional[Union[str, numpy.dtype]] = None,
        tags: Optional[dict[str, str]] = None,
        windowed: bool = False,
        recalc_transform: bool = True,
        lock: Optional[bool] = None,
        compute: bool = True,
        **profile_kwargs,
    ) -> None:
        """
        Export the DataArray to a raster file.

        ..versionadded:: 0.2 lock

        Parameters
        ----------
        raster_path: Union[str, os.PathLike]
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
        recalc_transform: bool, optional
            If False, it will write the raster with the cached transform from
            the dataarray rather than recalculating it.
            Default is True.
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
        if driver is None:
            extension = Path(raster_path).suffix
            # https://github.com/rasterio/rasterio/pull/2008
            if extension in (".tif", ".tiff"):
                driver = "GTiff"

        # get the output profile from the rasterio object
        # if opened with xarray.open_rasterio()
        try:
            out_profile = self._manager.acquire().profile  # type: ignore
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
            gcps=self.get_gcps(),
            nodata=rio_nodata,
            windowed=windowed,
            lock=lock,
            compute=compute,
            **out_profile,
        )
