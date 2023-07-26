"""
This module is an extension for xarray to provide rasterio capabilities
to xarray datasets.
"""
import os
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional, Union
from uuid import uuid4

import numpy
import rasterio.crs
import xarray
from affine import Affine
from rasterio.enums import Resampling

from rioxarray._options import SKIP_MISSING_SPATIAL_DIMS, get_option
from rioxarray.exceptions import MissingSpatialDimensionError, RioXarrayError
from rioxarray.rioxarray import XRasterBase, _get_spatial_dims


@xarray.register_dataset_accessor("rio")
class RasterDataset(XRasterBase):
    """This is the GIS extension for :class:`xarray.Dataset`"""

    @property
    def vars(self) -> list:
        """list: Returns non-coordinate varibles"""
        return list(self._obj.data_vars)

    @property
    def crs(self) -> Optional[rasterio.crs.CRS]:
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
        if all(crs_i == crs for crs_i in crs_list):
            self._crs = crs
        else:
            raise RioXarrayError(f"CRS in DataArrays differ in the Dataset: {crs_list}")
        return self._crs

    def reproject(
        self,
        dst_crs: Any,
        resolution: Optional[Union[float, tuple[float, float]]] = None,
        shape: Optional[tuple[int, int]] = None,
        transform: Optional[Affine] = None,
        resampling: Resampling = Resampling.nearest,
        nodata: Optional[float] = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Reproject :class:`xarray.Dataset` objects

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Others are appended as is.
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
        --------
        :class:`xarray.Dataset`:
            The reprojected Dataset.
        """
        resampled_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            try:
                x_dim, y_dim = _get_spatial_dims(self._obj, var)
                resampled_dataset[var] = (
                    self._obj[var]
                    .rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                    .rio.reproject(
                        dst_crs,
                        resolution=resolution,
                        shape=shape,
                        transform=transform,
                        resampling=resampling,
                        nodata=nodata,
                        **kwargs,
                    )
                )
            except MissingSpatialDimensionError:
                if len(self._obj[var].dims) >= 2 and not get_option(
                    SKIP_MISSING_SPATIAL_DIMS
                ):
                    raise
                resampled_dataset[var] = self._obj[var].copy()
        return resampled_dataset

    def reproject_match(
        self,
        match_data_array: Union[xarray.DataArray, xarray.Dataset],
        resampling: Resampling = Resampling.nearest,
        **reproject_kwargs,
    ) -> xarray.Dataset:
        """
        Reproject a Dataset object to match the resolution, projection,
        and region of another DataArray.

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Others are appended as is.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        .. versionadded:: 0.9 reproject_kwargs

        Parameters
        ----------
        match_data_array: :obj:`xarray.DataArray` | :obj:`xarray.Dataset`
            Dataset with the target resolution and projection.
        resampling: rasterio.enums.Resampling, optional
            See :func:`rasterio.warp.reproject` for more details.
        **reproject_kwargs:
            Other options to pass to :meth:`rioxarray.raster_dataset.RasterDataset.reproject`

        Returns
        --------
        :obj:`xarray.Dataset`:
            Contains the data from the src_data_array,
            reprojected to match match_data_array.
        """
        resampled_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            try:
                x_dim, y_dim = _get_spatial_dims(self._obj, var)
                resampled_dataset[var] = (
                    self._obj[var]
                    .rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                    .rio.reproject_match(
                        match_data_array, resampling=resampling, **reproject_kwargs
                    )
                )
            except MissingSpatialDimensionError:
                if len(self._obj[var].dims) >= 2 and not get_option(
                    SKIP_MISSING_SPATIAL_DIMS
                ):
                    raise
                resampled_dataset[var] = self._obj[var].copy()
        return resampled_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def pad_box(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        constant_values: Union[
            float, tuple[int, int], Mapping[Any, tuple[int, int]], None
        ] = None,
    ) -> xarray.Dataset:
        """Pad the :class:`xarray.Dataset` to a bounding box.

        .. warning:: Only works if all variables in the dataset have the
                     same coordinates.

        .. warning:: Pads variables that have dimensions 'x'/'y'. Others are appended as is.

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
        :obj:`xarray.Dataset`:
            The padded object.
        """
        padded_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            try:
                x_dim, y_dim = _get_spatial_dims(self._obj, var)
                padded_dataset[var] = (
                    self._obj[var]
                    .rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                    .rio.pad_box(
                        minx, miny, maxx, maxy, constant_values=constant_values
                    )
                )
            except MissingSpatialDimensionError:
                if len(self._obj[var].dims) >= 2 and not get_option(
                    SKIP_MISSING_SPATIAL_DIMS
                ):
                    raise
                padded_dataset[var] = self._obj[var].copy()
        return padded_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def clip_box(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        auto_expand: Union[bool, int] = False,
        auto_expand_limit: int = 3,
        crs: Optional[Any] = None,
    ) -> xarray.Dataset:
        """Clip the :class:`xarray.Dataset` by a bounding box in dimensions 'x'/'y'.

        .. warning:: Clips variables that have dimensions 'x'/'y'. Others are appended as is.

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
        auto_expand: bool
            If True, it will expand clip search if only 1D raster found with clip.
        auto_expand_limit: int
            maximum number of times the clip will be retried before raising
            an exception.
        crs: :obj:`rasterio.crs.CRS`, optional
            The CRS of the bounding box. Default is to assume it is the same
            as the dataset.

        Returns
        -------
        Dataset:
            The clipped object.
        """
        clipped_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            try:
                x_dim, y_dim = _get_spatial_dims(self._obj, var)
                clipped_dataset[var] = (
                    self._obj[var]
                    .rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                    .rio.clip_box(
                        minx,
                        miny,
                        maxx,
                        maxy,
                        auto_expand=auto_expand,
                        auto_expand_limit=auto_expand_limit,
                        crs=crs,
                    )
                )
            except MissingSpatialDimensionError:
                if len(self._obj[var].dims) >= 2 and not get_option(
                    SKIP_MISSING_SPATIAL_DIMS
                ):
                    raise
                clipped_dataset[var] = self._obj[var].copy()
        return clipped_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def clip(
        self,
        geometries: Iterable,
        crs: Optional[Any] = None,
        all_touched: bool = False,
        drop: bool = True,
        invert: bool = False,
        from_disk: bool = False,
    ) -> xarray.Dataset:
        """
        Crops a :class:`xarray.Dataset` by geojson like geometry dicts in dimensions 'x'/'y'.

        .. warning:: Clips variables that have dimensions 'x'/'y'. Others are appended as is.

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
            A list of geojson geometry dicts.
        crs: :obj:`rasterio.crs.CRS`, optional
            The CRS of the input geometries. Default is to assume it is the same
            as the dataset.
        all_touched : boolean, optional
            If True, all pixels touched by geometries will be burned in.  If
            false, only pixels whose center is within the polygon or that
            are selected by Bresenham's line algorithm will be burned in.
        drop: bool, optional
            If True, drop the data outside of the extent of the mask geometries
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
        :obj:`xarray.Dataset`:
            The clipped object.
        """
        clipped_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            try:
                x_dim, y_dim = _get_spatial_dims(self._obj, var)
                clipped_dataset[var] = (
                    self._obj[var]
                    .rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                    .rio.clip(
                        geometries,
                        crs=crs,
                        all_touched=all_touched,
                        drop=drop,
                        invert=invert,
                        from_disk=from_disk,
                    )
                )
            except MissingSpatialDimensionError:
                if len(self._obj[var].dims) >= 2 and not get_option(
                    SKIP_MISSING_SPATIAL_DIMS
                ):
                    raise
                clipped_dataset[var] = self._obj[var].copy()
        return clipped_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

    def interpolate_na(
        self, method: Literal["linear", "nearest", "cubic"] = "nearest"
    ) -> xarray.Dataset:
        """
        This method uses `scipy.interpolate.griddata` to interpolate missing data.

        .. warning:: scipy is an optional dependency.

        .. warning:: Interpolates variables that have dimensions 'x'/'y'. Others are appended as is.

        Parameters
        ----------
        method: {'linear', 'nearest', 'cubic'}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :obj:`xarray.DataArray`:
             The interpolated object.
        """
        interpolated_dataset = xarray.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            try:
                x_dim, y_dim = _get_spatial_dims(self._obj, var)
                interpolated_dataset[var] = (
                    self._obj[var]
                    .rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                    .rio.interpolate_na(method=method)
                )
            except MissingSpatialDimensionError:
                if len(self._obj[var].dims) >= 2 and not get_option(
                    SKIP_MISSING_SPATIAL_DIMS
                ):
                    raise
                interpolated_dataset[var] = self._obj[var].copy()
        return interpolated_dataset.rio.set_spatial_dims(
            x_dim=self.x_dim, y_dim=self.y_dim, inplace=True
        )

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
        Export the Dataset to a raster file. Only works with 2D data.

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
        recalc_transform: bool, optional
            If False, it will write the raster with the cached transform from
            the dataset rather than recalculating it.
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
        variable_dim = f"band_{uuid4()}"
        data_array = self._obj.to_array(dim=variable_dim)
        # ensure raster metadata preserved
        scales = []
        offsets = []
        nodatavals = []
        band_tags = []
        long_name = []
        for data_var in data_array[variable_dim].values:
            scales.append(self._obj[data_var].attrs.get("scale_factor", 1.0))
            offsets.append(self._obj[data_var].attrs.get("add_offset", 0.0))
            long_name.append(self._obj[data_var].attrs.get("long_name", data_var))
            nodatavals.append(self._obj[data_var].rio.nodata)
            band_tags.append(self._obj[data_var].attrs.copy())
        data_array.attrs["scales"] = scales
        data_array.attrs["offsets"] = offsets
        data_array.attrs["band_tags"] = band_tags
        data_array.attrs["long_name"] = long_name

        nodata = nodatavals[0]
        if (
            all(nodataval == nodata for nodataval in nodatavals)
            or numpy.isnan(nodatavals).all()
        ):
            data_array.rio.write_nodata(nodata, inplace=True)
        else:
            raise RioXarrayError(
                "All nodata values must be the same when exporting to raster. "
                f"Current values: {nodatavals}"
            )
        if self.crs is not None:
            data_array.rio.write_crs(self.crs, inplace=True)
        # write it to a raster
        return data_array.rio.set_spatial_dims(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            inplace=True,
        ).rio.to_raster(
            raster_path=raster_path,
            driver=driver,
            dtype=dtype,
            tags=tags,
            windowed=windowed,
            recalc_transform=recalc_transform,
            lock=lock,
            compute=compute,
            **profile_kwargs,
        )
