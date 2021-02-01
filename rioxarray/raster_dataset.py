"""
This module is an extension for xarray to provide rasterio capabilities
to xarray datasets.
"""
from uuid import uuid4

import numpy as np
import xarray
from rasterio.enums import Resampling

from rioxarray.exceptions import RioXarrayError
from rioxarray.rioxarray import XRasterBase


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
        if all(crs_i == crs for crs_i in crs_list):
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
                    from_disk=from_disk,
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
        return data_array.rio.to_raster(
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
