"""
This module contains a dataset writer for Dask.

Credits:

RasterioWriter dask write functionality was adopted from https://github.com/dymaxionlabs/dask-rasterio  # noqa: E501
Source file:
- https://github.com/dymaxionlabs/dask-rasterio/blob/8dd7fdece7ad094a41908c0ae6b4fe6ca49cf5e1/dask_rasterio/write.py  # noqa: E501

"""
import rasterio
from rasterio.windows import Window

from rioxarray.exceptions import RioXarrayError

try:
    import dask.array
    from dask import is_dask_collection
except ImportError:

    def is_dask_collection(_):
        """
        Replacement method to check if it is a dask collection
        """
        # if you cannot import dask, then it cannot be a dask array
        return False


FILL_VALUE_NAMES = ("_FillValue", "missing_value", "fill_value", "nodata")
UNWANTED_RIO_ATTRS = ("nodatavals", "crs", "is_tiled", "res")


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


class RasterioWriter:
    """

    ..versionadded:: 0.2

    Rasterio wrapper to allow dask.array.store to do window saving or to
    save using the rasterio write method.
    """

    def __init__(self, raster_path):
        """
        raster_path: str
            The path to output the raster to.
        """
        # https://github.com/dymaxionlabs/dask-rasterio/issues/3#issuecomment-514781825
        # Rasterio datasets can't be pickled and can't be shared between
        # processes or threads. The work around is to distribute dataset
        # identifiers (paths or URIs) and then open them in new threads.
        # See mapbox/rasterio#1731.
        self.raster_path = raster_path

    def __setitem__(self, key, item):
        """Put the data chunk in the image"""
        if len(key) == 3:
            index_range, yyy, xxx = key
            indexes = list(
                range(
                    index_range.start + 1, index_range.stop + 1, index_range.step or 1
                )
            )
        else:
            indexes = 1
            yyy, xxx = key

        chy_off = yyy.start
        chy = yyy.stop - yyy.start
        chx_off = xxx.start
        chx = xxx.stop - xxx.start

        with rasterio.open(self.raster_path, "r+") as rds:
            rds.write(item, window=Window(chx_off, chy_off, chx, chy), indexes=indexes)

    def to_raster(self, xarray_dataarray, tags, windowed, lock, compute, **kwargs):
        """
        This method writes to the raster on disk.

        xarray_dataarray: xarray.DataArray
            The input data array to write to disk.
        tags: dict, optional
            A dictionary of tags to write to the raster.
        windowed: bool
            If True and the data array is not a dask array, it will write
            the data to disk using rasterio windows.
        lock: boolean or Lock, optional
            Lock to use to write data using dask.
            If not supplied, it will use a single process.
        compute: bool
            If True (default) and data is a dask array, then compute and save
            the data immediately. If False, return a dask Delayed object.
            Call ".compute()" on the Delayed object to compute the result
            later. Call ``dask.compute(delayed1, delayed2)`` to save
            multiple delayed files at once.
        **kwargs
            Keyword arguments to pass into writing the raster.
        """
        dtype = kwargs["dtype"]
        # generate initial output file
        with rasterio.open(self.raster_path, "w", **kwargs) as rds:
            _write_metatata_to_raster(rds, xarray_dataarray, tags)

            if not (lock and is_dask_collection(xarray_dataarray.data)):
                # write data to raster immmediately if not dask array
                if windowed:
                    window_iter = rds.block_windows(1)
                else:
                    window_iter = [(None, None)]
                for _, window in window_iter:
                    if window is not None:
                        out_data = xarray_dataarray.rio.isel_window(window)
                    else:
                        out_data = xarray_dataarray
                    if xarray_dataarray.rio.encoded_nodata is not None:
                        out_data = out_data.fillna(xarray_dataarray.rio.encoded_nodata)
                    data = out_data.values.astype(dtype)
                    if data.ndim == 2:
                        rds.write(data, 1, window=window)
                    else:
                        rds.write(data, window=window)

        if lock and is_dask_collection(xarray_dataarray.data):
            if xarray_dataarray.rio.encoded_nodata is not None:
                out_dataarray = xarray_dataarray.fillna(
                    xarray_dataarray.rio.encoded_nodata
                )
            else:
                out_dataarray = xarray_dataarray
            return dask.array.store(
                out_dataarray.data.astype(dtype),
                self,
                lock=lock,
                compute=compute,
            )
