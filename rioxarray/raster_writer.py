"""
This module contains a dataset writer for Dask.

Credits:

RasterioWriter dask write functionality was adopted from https://github.com/dymaxionlabs/dask-rasterio  # noqa: E501
Source file:
- https://github.com/dymaxionlabs/dask-rasterio/blob/8dd7fdece7ad094a41908c0ae6b4fe6ca49cf5e1/dask_rasterio/write.py  # noqa: E501

"""
import warnings

import numpy
import rasterio
from rasterio.windows import Window
from xarray.conventions import encode_cf_variable

from rioxarray._io import FILL_VALUE_NAMES, UNWANTED_RIO_ATTRS, _get_unsigned_dtype
from rioxarray.exceptions import RioXarrayError

try:
    import dask.array
    from dask import is_dask_collection
except ImportError:

    def is_dask_collection(_) -> bool:  # type: ignore
        """
        Replacement method to check if it is a dask collection
        """
        # if you cannot import dask, then it cannot be a dask array
        return False


# Note: transform & crs are removed in write_transform/write_crs


def _write_tags(raster_handle, tags):
    """
    Write tags to raster dataset
    """
    # filter out attributes that should be written in a different location
    skip_tags = (
        UNWANTED_RIO_ATTRS
        + FILL_VALUE_NAMES
        + (
            "crs",
            "transform",
            "scales",
            "scale_factor",
            "add_offset",
            "offsets",
            "grid_mapping",
        )
    )
    # this is for when multiple values are used
    # in this case, it will be stored in the raster description
    if not isinstance(tags.get("long_name"), str):
        skip_tags += ("long_name",)
    band_tags = tags.pop("band_tags", [])
    tags = {key: value for key, value in tags.items() if key not in skip_tags}
    raster_handle.update_tags(**tags)

    if isinstance(band_tags, list):
        for iii, band_tag in enumerate(band_tags):
            raster_handle.update_tags(iii + 1, **band_tag)


def _write_band_description(raster_handle, xarray_dataset):
    """
    Write band descriptions using the long name
    """
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


def _write_metatata_to_raster(raster_handle, xarray_dataset, tags):
    """
    Write the metadata stored in the xarray object to raster metadata
    """
    tags = (
        xarray_dataset.attrs.copy()
        if tags is None
        else {**xarray_dataset.attrs, **tags}
    )

    # write scales and offsets
    try:
        raster_handle.scales = tags["scales"]
    except KeyError:
        scale_factor = tags.get(
            "scale_factor", xarray_dataset.encoding.get("scale_factor")
        )
        if scale_factor is not None:
            raster_handle.scales = (scale_factor,) * raster_handle.count
    try:
        raster_handle.offsets = tags["offsets"]
    except KeyError:
        add_offset = tags.get("add_offset", xarray_dataset.encoding.get("add_offset"))
        if add_offset is not None:
            raster_handle.offsets = (add_offset,) * raster_handle.count

    _write_tags(raster_handle=raster_handle, tags=tags)
    _write_band_description(raster_handle=raster_handle, xarray_dataset=xarray_dataset)


def _ensure_nodata_dtype(original_nodata, new_dtype):
    """
    Convert the nodata to the new datatype and raise warning
    if the value of the nodata value changed.
    """
    # Complex-valued rasters can have real-valued nodata
    if str(new_dtype).startswith("c"):
        nodata = original_nodata
    else:
        original_nodata = float(original_nodata)
        nodata = numpy.dtype(new_dtype).type(original_nodata)
        if not numpy.isnan(nodata) and original_nodata != nodata:
            warnings.warn(
                f"The nodata value ({original_nodata}) has been automatically "
                f"changed to ({nodata}) to match the dtype of the data."
            )

    return nodata


def _get_dtypes(rasterio_dtype, encoded_rasterio_dtype, dataarray_dtype):
    """
    Determines the rasterio dtype and numpy dtypes based on
    the rasterio dtype and the encoded rasterio dtype.

    Parameters
    ----------
    rasterio_dtype: Union[str, numpy.dtype]
        The rasterio dtype to write to.
    encoded_rasterio_dtype: Union[str, numpy.dtype, None]
        The value of the original rasterio dtype in the encoding.
    dataarray_dtype: Union[str, numpy.dtype]
        The value of the dtype of the data array.

    Returns
    -------
    tuple[Union[str, numpy.dtype], Union[str, numpy.dtype]]:
        The rasterio dtype and numpy dtype.
    """
    # SCENARIO 1: User wants to write to complex_int16
    if rasterio_dtype == "complex_int16":
        numpy_dtype = "complex64"
    # SCENARIO 2: File originally in complext_int16 and dtype unchanged
    elif (
        rasterio_dtype is None
        and encoded_rasterio_dtype == "complex_int16"
        and str(dataarray_dtype) == "complex64"
    ):
        numpy_dtype = "complex64"
        rasterio_dtype = "complex_int16"
    # SCENARIO 3: rasterio dtype not provided
    elif rasterio_dtype is None:
        numpy_dtype = dataarray_dtype
        rasterio_dtype = dataarray_dtype
    # SCENARIO 4: rasterio dtype and numpy dtype are the same
    else:
        numpy_dtype = rasterio_dtype
    return rasterio_dtype, numpy_dtype


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
        dtype: numpy.dtype
            Numpy-compliant dtype used to save raster. If data is not already
            represented by this dtype in memory it is recast. dtype='complex_int16'
            is a special case to write in-memory numpy.complex64 to CInt16.
        **kwargs
            Keyword arguments to pass into writing the raster.
        """
        xarray_dataarray = xarray_dataarray.copy()
        kwargs["dtype"], numpy_dtype = _get_dtypes(
            kwargs["dtype"],
            xarray_dataarray.encoding.get("rasterio_dtype"),
            xarray_dataarray.encoding.get("dtype", str(xarray_dataarray.dtype)),
        )
        # there is no equivalent for netCDF _Unsigned
        # across output GDAL formats. It is safest to convert beforehand.
        # https://github.com/OSGeo/gdal/issues/6352#issuecomment-1245981837
        if "_Unsigned" in xarray_dataarray.encoding:
            unsigned_dtype = _get_unsigned_dtype(
                unsigned=xarray_dataarray.encoding["_Unsigned"] == "true",
                dtype=numpy_dtype,
            )
            if unsigned_dtype is not None:
                numpy_dtype = unsigned_dtype
                kwargs["dtype"] = unsigned_dtype
                xarray_dataarray.encoding["rasterio_dtype"] = str(unsigned_dtype)
                xarray_dataarray.encoding["dtype"] = str(unsigned_dtype)

        if kwargs["nodata"] is not None:
            # Ensure dtype of output data matches the expected dtype.
            # This check is added here as the dtype of the data is
            # converted right before writing.
            kwargs["nodata"] = _ensure_nodata_dtype(kwargs["nodata"], numpy_dtype)

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
                    data = encode_cf_variable(out_data.variable).values.astype(
                        numpy_dtype
                    )
                    if data.ndim == 2:
                        rds.write(data, 1, window=window)
                    else:
                        rds.write(data, window=window)

        if lock and is_dask_collection(xarray_dataarray.data):
            return dask.array.store(
                encode_cf_variable(xarray_dataarray.variable).data.astype(numpy_dtype),
                self,
                lock=lock,
                compute=compute,
            )
        return None
