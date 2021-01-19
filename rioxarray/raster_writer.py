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
    from dask import is_dask_collection
except ImportError:
    # if you cannot import dask, then it cannot be a dask array
    def is_dask_collection(*args, **kwargs):
        return False


FILL_VALUE_NAMES = ("_FillValue", "missing_value", "fill_value", "nodata")
UNWANTED_RIO_ATTRS = ("nodatavals", "crs", "is_tiled", "res")


class RasterioWriter:
    """
    Rasterio wrapper to allow dask.array.store to do window saving or to
    save using the rasterio write method.

    Example::

        >> rows = cols = 21696
        >> a = da.ones((4, rows, cols), dtype=np.float64, chunks=(1, 4096, 4096) )
        >> a = a * np.array([255., 255., 255., 255.])[:, None, None]
        >> a = a.astype(np.int16)
        >>> with RasterioDaskWriter(
        ...     'test.tif',
        ...     'w',
        ...     driver='GTiff',
        ...     width=cols,
        ...     height=rows,
        ...     count=4,
        ...     dtype=a.dtype,
        ... ) as r_file:
        ...     da.store(a, r_file, lock=True)

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.dataset = None

    def __setitem__(self, key, item):
        """Put the data chunk in the image"""
        if len(key) == 3:
            index_range, y, x = key
            indexes = list(
                range(
                    index_range.start + 1, index_range.stop + 1, index_range.step or 1
                )
            )
        else:
            indexes = 1
            y, x = key

        chy_off = y.start
        chy = y.stop - y.start
        chx_off = x.start
        chx = x.stop - x.start

        self.dataset.write(
            item, window=Window(chx_off, chy_off, chx, chy), indexes=indexes
        )

    def __enter__(self):
        self.dataset = rasterio.open(*self.args, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dataset.close()
        self.dataset = None

    def write_data(self, xarray_dataarray, dtype, windowed):
        """
        This method writes the data values to the raster on disk.

        xarray_dataarray: xarray.DataArray
            The input data array to write to disk.
        dtype: str
            The data type to convert the data to.
        windowed: bool
            If True and the data array is not a dask array, it will write
            the data to disk using rasterio windows.
        """
        encoded_nodata = xarray_dataarray.rio.encoded_nodata
        if is_dask_collection(xarray_dataarray.data):
            import dask.array

            if encoded_nodata is not None:
                out_dataarray = xarray_dataarray.fillna(encoded_nodata)
            else:
                out_dataarray = xarray_dataarray
            dask.array.store(out_dataarray.data.astype(dtype), self, lock=True)
        else:
            # write data to raster
            if windowed:
                window_iter = self.dataset.block_windows(1)
            else:
                window_iter = [(None, None)]
            for _, window in window_iter:
                if window is not None:
                    out_data = xarray_dataarray.rio.isel_window(window)
                else:
                    out_data = xarray_dataarray
                if encoded_nodata is not None:
                    out_data = out_data.fillna(encoded_nodata)
                data = out_data.values.astype(dtype)
                if data.ndim == 2:
                    self.dataset.write(data, 1, window=window)
                else:
                    self.dataset.write(data, window=window)

    def write_metadata(self, xarray_dataarray, tags):
        """
        Write the metadata stored in the xarray object to the raster.

        xarray_dataarray: xarray.DataArray
            The input data array to get metadata from.
        tags: dict
            A dictionary of tags to write to the raster.
        """
        tags = (
            xarray_dataarray.attrs
            if tags is None
            else {**xarray_dataarray.attrs, **tags}
        )

        # write scales and offsets
        try:
            self.dataset.scales = tags["scales"]
        except KeyError:
            try:
                self.dataset.scales = (tags["scale_factor"],) * self.dataset.count
            except KeyError:
                pass
        try:
            self.dataset.offsets = tags["offsets"]
        except KeyError:
            try:
                self.dataset.offsets = (tags["add_offset"],) * self.dataset.count
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
        self.dataset.update_tags(**tags)

        # write band name information
        long_name = xarray_dataarray.attrs.get("long_name")
        if isinstance(long_name, (tuple, list)):
            if len(long_name) != self.dataset.count:
                raise RioXarrayError(
                    "Number of names in the 'long_name' attribute does not equal "
                    "the number of bands."
                )
            for iii, band_description in enumerate(long_name):
                self.dataset.set_band_description(iii + 1, band_description)
        else:
            band_description = long_name or xarray_dataarray.name
            if band_description:
                for iii in range(self.dataset.count):
                    self.dataset.set_band_description(iii + 1, band_description)
