"""

Credits:

This file was adopted from: https://github.com/pydata/xarray # noqa
Source file: https://github.com/pydata/xarray/blob/1d7bcbdc75b6d556c04e2c7d7a042e4379e15303/xarray/backends/rasterio_.py # noqa
"""
# pylint: disable=too-many-lines
import contextlib
import functools
import importlib.metadata
import os
import re
import threading
import warnings
from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import Any, Optional, Union

import numpy
import rasterio
from numpy.typing import NDArray
from packaging import version
from rasterio.errors import NotGeoreferencedWarning
from rasterio.vrt import WarpedVRT
from xarray import Dataset, IndexVariable
from xarray.backends.common import BackendArray
from xarray.backends.file_manager import CachingFileManager, FileManager
from xarray.backends.locks import SerializableLock
from xarray.coding import times, variables
from xarray.core import indexing
from xarray.core.dataarray import DataArray
from xarray.core.dtypes import maybe_promote
from xarray.core.utils import is_scalar
from xarray.core.variable import as_variable

from rioxarray.exceptions import RioXarrayError
from rioxarray.rioxarray import _generate_spatial_coords

FILL_VALUE_NAMES = ("_FillValue", "missing_value", "fill_value", "nodata")
UNWANTED_RIO_ATTRS = ("nodatavals", "is_tiled", "res")
# TODO: should this be GDAL_LOCK instead?
RASTERIO_LOCK = SerializableLock()
NO_LOCK = contextlib.nullcontext()


def _ensure_warped_vrt(riods, vrt_params):
    """
    Ensuire the dataset is represented as a warped vrt
    """
    if vrt_params is None:
        return riods
    if isinstance(riods, SingleBandDatasetReader):
        riods._create_vrt(vrt_params)
    else:
        riods = WarpedVRT(riods, **vrt_params)
    return riods


class SingleBandDatasetReader:
    """
    Hack to have a DatasetReader behave like it only has one band
    """

    def __init__(self, riods, bidx, vrt_params=None) -> None:
        self._riods = riods
        self._bidx = bidx
        self._vrt_params = vrt_params
        self._create_vrt(vrt_params=vrt_params)

    def __getattr__(self, __name: str) -> Any:
        return getattr(self._riods, __name)

    def _create_vrt(self, vrt_params):
        if vrt_params is not None and not isinstance(self._riods, WarpedVRT):
            self._riods = WarpedVRT(self._riods, **vrt_params)
        self._vrt_params = vrt_params

    @property
    def name(self):
        """
        str: name of the dataset. Usually the path.
        """
        if isinstance(self._riods, rasterio.vrt.WarpedVRT):
            return self._riods.src_dataset.name
        return self._riods.name

    @property
    def count(self):
        """
        int: band count
        """
        return 1

    @property
    def nodata(self):
        """
        Nodata value for the band
        """
        return self._riods.nodatavals[self._bidx]

    @property
    def offsets(self):
        """
        Offset value for the band
        """
        return [self._riods.offsets[self._bidx]]

    @property
    def scales(self):
        """
        Scale value for the band
        """
        return [self._riods.scales[self._bidx]]

    @property
    def units(self):
        """
        Unit for the band
        """
        return [self._riods.units[self._bidx]]

    @property
    def descriptions(self):
        """
        Description for the band
        """
        return [self._riods.descriptions[self._bidx]]

    @property
    def dtypes(self):
        """
        dtype for the band
        """
        return [self._riods.dtypes[self._bidx]]

    @property
    def indexes(self):
        """
        indexes for the band
        """
        return [self._riods.indexes[self._bidx]]

    def read(self, indexes=None, **kwargs):  # pylint: disable=unused-argument
        """
        read data for the band
        """
        return self._riods.read(indexes=self._bidx + 1, **kwargs)

    def tags(self, bidx=None, **kwargs):  # pylint: disable=unused-argument
        """
        read tags for the band
        """
        return self._riods.tags(bidx=self._bidx + 1, **kwargs)


RasterioReader = Union[
    rasterio.io.DatasetReader, rasterio.vrt.WarpedVRT, SingleBandDatasetReader
]


try:
    _DASK_GTE_018 = version.parse(importlib.metadata.version("dask")) >= version.parse(
        "0.18.0"
    )
except importlib.metadata.PackageNotFoundError:
    _DASK_GTE_018 = False


def _get_unsigned_dtype(unsigned, dtype):
    """
    Based on: https://github.com/pydata/xarray/blob/abe1e613a96b000ae603c53d135828df532b952e/xarray/coding/variables.py#L306-L334
    """
    dtype = numpy.dtype(dtype)
    if unsigned is True and dtype.kind == "i":
        return numpy.dtype(f"u{dtype.itemsize}")
    if unsigned is False and dtype.kind == "u":
        return numpy.dtype(f"i{dtype.itemsize}")
    return None


class FileHandleLocal(threading.local):
    """
    This contains the thread local ThreadURIManager
    """

    def __init__(self):  # pylint: disable=super-init-not-called
        self.thread_manager = None  # Initialises in each thread


class ThreadURIManager:
    """
    This handles opening & closing file handles in each thread.
    """

    def __init__(
        self,
        opener,
        *args,
        mode="r",
        kwargs=None,
    ):
        self._opener = opener
        self._args = args
        self._mode = mode
        self._kwargs = {} if kwargs is None else dict(kwargs)
        self._file_handle = None

    @property
    def file_handle(self):
        """
        File handle returned by the opener.
        """
        if self._file_handle is not None:
            return self._file_handle
        self._file_handle = self._opener(*self._args, mode=self._mode, **self._kwargs)
        return self._file_handle

    def close(self):
        """
        Close file handle.
        """
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.close()


class URIManager(FileManager):
    """
    The URI manager is used for lockless reading
    """

    def __init__(
        self,
        opener,
        *args,
        mode="r",
        kwargs=None,
    ):
        self._opener = opener
        self._args = args
        self._mode = mode
        self._kwargs = {} if kwargs is None else dict(kwargs)
        self._local = FileHandleLocal()

    def acquire(self, needs_lock=True):
        if self._local.thread_manager is None:
            self._local.thread_manager = ThreadURIManager(
                self._opener, *self._args, mode=self._mode, kwargs=self._kwargs
            )
        return self._local.thread_manager.file_handle

    @contextlib.contextmanager
    def acquire_context(self, needs_lock=True):
        try:
            yield self.acquire(needs_lock=needs_lock)
        except Exception:
            self.close(needs_lock=needs_lock)
            raise

    def close(self, needs_lock=True):
        if self._local.thread_manager is not None:
            self._local.thread_manager.close()
            self._local.thread_manager = None

    def __del__(self):
        self.close(needs_lock=False)

    def __getstate__(self):
        """State for pickling."""
        return (self._opener, self._args, self._mode, self._kwargs)

    def __setstate__(self, state):
        """Restore from a pickle."""
        opener, args, mode, kwargs = state
        self.__init__(opener, *args, mode=mode, kwargs=kwargs)


class RasterioArrayWrapper(BackendArray):
    """A wrapper around rasterio dataset objects"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        manager,
        lock,
        name,
        vrt_params=None,
        masked=False,
        mask_and_scale=False,
        unsigned=False,
    ):
        self.manager = manager
        self.lock = lock
        self.masked = masked or mask_and_scale
        self.mask_and_scale = mask_and_scale

        # cannot save riods as an attribute: this would break pickleability
        riods = _ensure_warped_vrt(manager.acquire(), vrt_params)
        self.vrt_params = vrt_params
        self._shape = (riods.count, riods.height, riods.width)

        self._dtype = None
        self._unsigned_dtype = None
        self._fill_value = riods.nodata
        dtypes = riods.dtypes
        if not numpy.all(numpy.asarray(dtypes) == dtypes[0]):
            raise ValueError("All bands should have the same dtype")

        dtype = _rasterio_to_numpy_dtype(dtypes)
        if mask_and_scale and unsigned is not None:
            self._unsigned_dtype = _get_unsigned_dtype(
                unsigned=unsigned,
                dtype=dtype,
            )
            if self._unsigned_dtype is not None and self._fill_value is not None:
                self._fill_value = self._unsigned_dtype.type(self._fill_value)
            if self._unsigned_dtype is None and dtype.kind not in ("i", "u"):
                warnings.warn(
                    f"variable {name!r} has _Unsigned attribute but is not "
                    "of integer type. Ignoring attribute.",
                    variables.SerializationWarning,
                    stacklevel=3,
                )
        if self.masked:
            self._dtype, self._fill_value = maybe_promote(dtype)
        else:
            self._dtype = dtype

    @property
    def dtype(self):
        """
        Data type of the array
        """
        return self._dtype

    @property
    def fill_value(self):
        """
        Fill value of the array
        """
        return self._fill_value

    @property
    def shape(self):
        """
        Shape of the array
        """
        return self._shape

    def _get_indexer(self, key):
        """Get indexer for rasterio array.

        Parameter
        ---------
        key: tuple of int

        Returns
        -------
        band_key: an indexer for the 1st dimension
        window: two tuples. Each consists of (start, stop).
        squeeze_axis: axes to be squeezed
        np_ind: indexer for loaded numpy array

        See also
        --------
        indexing.decompose_indexer
        """
        if len(key) != 3:
            raise RioXarrayError("rasterio datasets should always be 3D")

        # bands cannot be windowed but they can be listed
        band_key = key[0]
        np_inds = []
        # bands (axis=0) cannot be windowed but they can be listed
        if isinstance(band_key, slice):
            start, stop, step = band_key.indices(self.shape[0])
            band_key = numpy.arange(start, stop, step)
        # be sure we give out a list
        band_key = (numpy.asarray(band_key) + 1).tolist()
        if isinstance(band_key, list):  # if band_key is not a scalar
            np_inds.append(slice(None))

        # but other dims can only be windowed
        window = []
        squeeze_axis = []
        for iii, (ikey, size) in enumerate(zip(key[1:], self.shape[1:])):
            if isinstance(ikey, slice):
                # step is always positive. see indexing.decompose_indexer
                start, stop, step = ikey.indices(size)
                np_inds.append(slice(None, None, step))
            elif is_scalar(ikey):
                # windowed operations will always return an array
                # we will have to squeeze it later
                squeeze_axis.append(-(2 - iii))
                start = ikey
                stop = ikey + 1
            else:
                start, stop = numpy.min(ikey), numpy.max(ikey) + 1
                np_inds.append(ikey - start)
            window.append((start, stop))

        if isinstance(key[1], numpy.ndarray) and isinstance(key[2], numpy.ndarray):
            # do outer-style indexing
            np_inds[-2:] = numpy.ix_(*np_inds[-2:])

        return band_key, tuple(window), tuple(squeeze_axis), tuple(np_inds)

    def _getitem(self, key):
        band_key, window, squeeze_axis, np_inds = self._get_indexer(key)
        if not band_key or any(start == stop for (start, stop) in window):
            # no need to do IO
            shape = (len(band_key),) + tuple(stop - start for (start, stop) in window)
            out = numpy.zeros(shape, dtype=self.dtype)
        else:
            with self.lock:
                riods = _ensure_warped_vrt(
                    self.manager.acquire(needs_lock=False), self.vrt_params
                )
                out = riods.read(band_key, window=window, masked=self.masked)
                if self._unsigned_dtype is not None:
                    out = out.astype(self._unsigned_dtype)
                if self.masked:
                    out = numpy.ma.filled(out.astype(self.dtype), self.fill_value)
                if self.mask_and_scale:
                    if not isinstance(band_key, Iterable):
                        out = (
                            out * riods.scales[band_key - 1]
                            + riods.offsets[band_key - 1]
                        )
                    else:
                        for iii, band_iii in enumerate(numpy.atleast_1d(band_key) - 1):
                            out[iii] = (
                                out[iii] * riods.scales[band_iii]
                                + riods.offsets[band_iii]
                            )

        if squeeze_axis:
            out = numpy.squeeze(out, axis=squeeze_axis)
        return out[np_inds]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )


def _parse_envi(meta):
    """Parse ENVI metadata into Python data structures.

    See the link for information on the ENVI header file format:
    http://www.harrisgeospatial.com/docs/enviheaderfiles.html

    Parameters
    ----------
    meta : dict
        Dictionary of keys and str values to parse, as returned by the rasterio
        tags(ns='ENVI') call.

    Returns
    -------
    parsed_meta : dict
        Dictionary containing the original keys and the parsed values

    """

    def parsevec(value):
        return numpy.fromstring(value.strip("{}"), dtype="float", sep=",")

    def default(value):
        return value.strip("{}")

    parse = {"wavelength": parsevec, "fwhm": parsevec}
    parsed_meta = {key: parse.get(key, default)(value) for key, value in meta.items()}
    return parsed_meta


def _rasterio_to_numpy_dtype(dtypes):
    """Numpy dtype from first entry of rasterio dataset.dtypes"""
    # rasterio has some special dtype names (complex_int16 -> numpy.complex64)
    if dtypes[0] == "complex_int16":
        dtype = numpy.dtype("complex64")
    else:
        dtype = numpy.dtype(dtypes[0])

    return dtype


def _to_numeric(value: Any) -> float:
    """
    Convert the value to a number
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass
    return value


def _parse_tag(key: str, value: Any) -> tuple[str, Any]:
    # NC_GLOBAL is appended to tags with netcdf driver and is not really needed
    key = key.split("NC_GLOBAL#")[-1]
    if value.startswith("{") and value.endswith("}"):
        try:
            new_val = numpy.fromstring(value.strip("{}"), dtype="float", sep=",")
            # pylint: disable=len-as-condition
            value = new_val if len(new_val) else _to_numeric(value)
        except ValueError:
            value = _to_numeric(value)
    else:
        value = _to_numeric(value)
    return key, value


def _parse_tags(tags: dict) -> dict:
    parsed_tags = {}
    for key, value in tags.items():
        key, value = _parse_tag(key, value)
        parsed_tags[key] = value
    return parsed_tags


NETCDF_DTYPE_MAP = {
    0: object,  # NC_NAT
    1: numpy.byte,  # NC_BYTE
    2: numpy.char,  # NC_CHAR
    3: numpy.short,  # NC_SHORT
    4: numpy.int_,  # NC_INT, NC_LONG
    5: float,  # NC_FLOAT
    6: numpy.double,  # NC_DOUBLE
    7: numpy.ubyte,  # NC_UBYTE
    8: numpy.ushort,  # NC_USHORT
    9: numpy.uint,  # NC_UINT
    10: numpy.int64,  # NC_INT64
    11: numpy.uint64,  # NC_UINT64
    12: object,  # NC_STRING
}


def _load_netcdf_attrs(tags: dict, data_array: DataArray) -> None:
    """
    Loads the netCDF attributes into the data array

    Attributes stored in this format:
    - variable_name#attr_name: attr_value
    """
    for key, value in tags.items():
        key, value = _parse_tag(key, value)
        key_split = key.split("#")
        if len(key_split) != 2:
            continue
        variable_name, attr_name = key_split
        if variable_name in data_array.coords:
            data_array.coords[variable_name].attrs.update({attr_name: value})


def _parse_netcdf_attr_array(attr: Union[NDArray, str], dtype=None) -> NDArray:
    """
    Expected format: '{2,6}' or '[2. 6.]'
    """
    value: Union[NDArray, str, list]
    if isinstance(attr, str):
        if attr.startswith("{"):
            value = attr.strip("{}").split(",")
        else:
            value = attr.strip("[]").split()
    elif not isinstance(attr, Iterable):
        value = [attr]
    else:
        value = attr
    return numpy.array(value, dtype=dtype)


def _load_netcdf_1d_coords(tags: dict) -> dict:
    """
    Dimension information:
        - NETCDF_DIM_EXTRA: '{time}' (comma separated list of dim names)
        - NETCDF_DIM_time_DEF: '{2,6}' or '[2. 6.]' (dim size, dim dtype)
        - NETCDF_DIM_time_VALUES: '{0,872712.659688}' (comma separated list of data) or [     0.       872712.659688]
    """
    dim_names = tags.get("NETCDF_DIM_EXTRA")
    if not dim_names:
        return {}
    dim_names = _parse_netcdf_attr_array(dim_names)
    coords = {}
    for dim_name in dim_names:
        dim_def = tags.get(f"NETCDF_DIM_{dim_name}_DEF")
        if dim_def is None:
            continue
        # pylint: disable=unused-variable
        dim_size, dim_dtype = _parse_netcdf_attr_array(dim_def)
        dim_dtype = NETCDF_DTYPE_MAP.get(int(float(dim_dtype)), object)
        dim_values = _parse_netcdf_attr_array(tags[f"NETCDF_DIM_{dim_name}_VALUES"])
        coords[dim_name] = IndexVariable(dim_name, dim_values)
    return coords


def build_subdataset_filter(
    group_names: Optional[Union[str, list[str], tuple[str, ...]]],
    variable_names: Optional[Union[str, list[str], tuple[str, ...]]],
):
    """
    Example::
        'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf":
        MODIS_Grid_2D:sur_refl_b01_1'

    Parameters
    ----------
    group_names: str or list or tuple
        Name or names of netCDF groups to filter by.

    variable_names: str or list or tuple
        Name or names of netCDF variables to filter by.

    Returns
    -------
    re.SRE_Pattern: output of re.compile()
    """
    variable_query = r"\w+"
    if variable_names is not None:
        if not isinstance(variable_names, (tuple, list)):
            variable_names = [variable_names]
        variable_names = [re.escape(variable_name) for variable_name in variable_names]
        variable_query = rf"(?:{'|'.join(variable_names)})"
    if group_names is not None:
        if not isinstance(group_names, (tuple, list)):
            group_names = [group_names]
        group_names = [re.escape(group_name) for group_name in group_names]
        group_query = rf"(?:{'|'.join(group_names)})"
    else:
        return re.compile(r"".join([r".*(?:\:/|\:)(/+)?", variable_query, r"$"]))
    return re.compile(
        r"".join(
            [r".*(?:\:/|\:)(/+)?", group_query, r"[:/](/+)?", variable_query, r"$"]
        )
    )


def _get_rasterio_attrs(riods: RasterioReader):
    """
    Get rasterio specific attributes
    """
    # pylint: disable=too-many-branches
    # Add rasterio attributes
    attrs = _parse_tags({**riods.tags(), **riods.tags(1)})
    # remove attributes with informaiton
    # that should be added by GDAL/rasterio
    for unwanted_attr in FILL_VALUE_NAMES + UNWANTED_RIO_ATTRS:
        attrs.pop(unwanted_attr, None)
    if riods.nodata is not None:
        # The nodata values for the raster bands
        attrs["_FillValue"] = riods.nodata
    # The scale values for the raster bands
    if len(set(riods.scales)) > 1:
        attrs["scales"] = riods.scales
        warnings.warn(
            "Offsets differ across bands. The 'scale_factor' attribute will "
            "not be added. See the 'scales' attribute."
        )
    else:
        attrs["scale_factor"] = riods.scales[0]
    # The offset values for the raster bands
    if len(set(riods.offsets)) > 1:
        attrs["offsets"] = riods.offsets
        warnings.warn(
            "Offsets differ across bands. The 'add_offset' attribute will "
            "not be added. See the 'offsets' attribute."
        )
    else:
        attrs["add_offset"] = riods.offsets[0]
    if any(riods.descriptions):
        if len(set(riods.descriptions)) == 1:
            attrs["long_name"] = riods.descriptions[0]
        else:
            # Descriptions for each dataset band
            attrs["long_name"] = riods.descriptions
    if any(riods.units):
        # A list of units string for each dataset band
        if len(riods.units) == 1:
            attrs["units"] = riods.units[0]
        else:
            attrs["units"] = riods.units

    return attrs


def _decode_datetime_cf(
    data_array: DataArray,
    decode_times: bool,
    decode_timedelta: Optional[bool],
) -> DataArray:
    """
    Decide the datetime based on CF conventions
    """
    if decode_timedelta is None:
        decode_timedelta = decode_times

    for coord in data_array.coords:
        time_var = None
        if decode_times and "since" in data_array[coord].attrs.get("units", ""):
            time_var = times.CFDatetimeCoder(use_cftime=True).decode(
                as_variable(data_array[coord]), name=coord
            )
        elif (
            decode_timedelta
            and data_array[coord].attrs.get("units") in times.TIME_UNITS
        ):
            time_var = times.CFTimedeltaCoder().decode(
                as_variable(data_array[coord]), name=coord
            )
        if time_var is not None:
            dimensions, data, attributes, encoding = variables.unpack_for_decoding(
                time_var
            )
            data_array = data_array.assign_coords(
                {
                    coord: IndexVariable(
                        dims=dimensions,
                        data=data,
                        attrs=attributes,
                        encoding=encoding,
                    )
                }
            )
    return data_array


def _parse_driver_tags(
    riods: RasterioReader,
    attrs: dict,
    coords: dict,
) -> None:
    # Parse extra metadata from tags, if supported
    parsers = {"ENVI": _parse_envi}

    driver = riods.driver
    if driver in parsers:
        meta = parsers[driver](riods.tags(ns=driver))

        for key, value in meta.items():
            # Add values as coordinates if they match the band count,
            # as attributes otherwise
            if isinstance(value, (list, numpy.ndarray)) and len(value) == riods.count:
                coords[key] = ("band", numpy.asarray(value))
            else:
                attrs[key] = value


def _pop_global_netcdf_attrs_from_vars(dataset_to_clean: Dataset) -> Dataset:
    # remove GLOBAL netCDF attributes from dataset variables
    for coord in dataset_to_clean.coords:
        for variable in dataset_to_clean.variables:
            dataset_to_clean[variable].attrs = {
                attr: value
                for attr, value in dataset_to_clean[variable].attrs.items()
                if attr not in dataset_to_clean.attrs
                and not attr.startswith(f"{coord}#")
            }
    return dataset_to_clean


def _subdataset_groups_to_dataset(
    dim_groups: dict[Hashable, dict[Hashable, DataArray]], global_tags: dict
) -> Union[Dataset, list[Dataset]]:
    if dim_groups:
        dataset: Union[Dataset, list[Dataset]] = []
        for dim_group in dim_groups.values():
            dataset_group = _pop_global_netcdf_attrs_from_vars(
                Dataset(dim_group, attrs=global_tags)
            )

            def _ds_close():
                # pylint: disable=cell-var-from-loop
                for data_var in dim_group.values():
                    data_var.close()

            dataset_group.set_close(_ds_close)
            dataset.append(dataset_group)
        if len(dataset) == 1:
            dataset = dataset.pop()
    else:
        dataset = Dataset(attrs=global_tags)
    return dataset


def _load_subdatasets(
    riods: RasterioReader,
    group: Optional[Union[str, list[str], tuple[str, ...]]],
    variable: Optional[Union[str, list[str], tuple[str, ...]]],
    parse_coordinates: bool,
    chunks: Optional[Union[int, tuple, dict]],
    cache: Optional[bool],
    lock: Any,
    masked: bool,
    mask_and_scale: bool,
    decode_times: bool,
    decode_timedelta: Optional[bool],
    **open_kwargs,
) -> Union[Dataset, list[Dataset]]:
    """
    Load in rasterio subdatasets
    """
    dim_groups: dict[Hashable, dict[Hashable, DataArray]] = defaultdict(dict)
    subdataset_filter = None
    if any((group, variable)):
        subdataset_filter = build_subdataset_filter(group, variable)
    for subdataset in riods.subdatasets:
        if subdataset_filter is not None and not subdataset_filter.match(subdataset):
            continue
        with rasterio.open(subdataset) as rds:
            shape = rds.shape
        rioda: DataArray = open_rasterio(  # type: ignore
            subdataset,
            parse_coordinates=shape not in dim_groups and parse_coordinates,
            chunks=chunks,
            cache=cache,
            lock=lock,
            masked=masked,
            mask_and_scale=mask_and_scale,
            default_name=subdataset.split(":")[-1].lstrip("/").replace("/", "_"),
            decode_times=decode_times,
            decode_timedelta=decode_timedelta,
            **open_kwargs,
        )
        dim_groups[shape][rioda.name] = rioda
    return _subdataset_groups_to_dataset(
        dim_groups=dim_groups, global_tags=_parse_tags(riods.tags())
    )


def _load_bands_as_variables(
    riods: RasterioReader,
    parse_coordinates: bool,
    chunks: Optional[Union[int, tuple, dict]],
    cache: Optional[bool],
    lock: Any,
    masked: bool,
    mask_and_scale: bool,
    decode_times: bool,
    decode_timedelta: Optional[bool],
    vrt_params: Optional[dict],
    **open_kwargs,
) -> Union[Dataset, list[Dataset]]:
    """
    Load in rasterio bands as variables
    """
    global_tags = _parse_tags(riods.tags())
    data_vars = {}
    for band in riods.indexes:
        band_riods = SingleBandDatasetReader(
            riods=riods,
            bidx=band - 1,
            vrt_params=vrt_params,
        )
        band_name = f"band_{band}"
        data_vars[band_name] = (
            open_rasterio(  # type: ignore
                band_riods,
                parse_coordinates=band == 1 and parse_coordinates,
                chunks=chunks,
                cache=cache,
                lock=lock,
                masked=masked,
                mask_and_scale=mask_and_scale,
                default_name=band_name,
                decode_times=decode_times,
                decode_timedelta=decode_timedelta,
                **open_kwargs,
            )
            .squeeze()  # type: ignore
            .drop("band")  # type: ignore
        )
    dataset = Dataset(data_vars, attrs=global_tags)

    def _ds_close():
        for data_var in data_vars.values():
            data_var.close()

    dataset.set_close(_ds_close)
    return dataset


def _prepare_dask(
    result: DataArray,
    riods: RasterioReader,
    filename: Union[str, os.PathLike],
    chunks: Union[int, tuple, dict],
) -> DataArray:
    """
    Prepare the data for dask computations
    """
    # pylint: disable=import-outside-toplevel
    from dask.base import tokenize

    # augment the token with the file modification time
    try:
        mtime = os.path.getmtime(filename)
    except (TypeError, OSError):
        # the filename is probably an s3 bucket rather than a regular file
        mtime = None

    if chunks in (True, "auto"):
        from dask.array.core import normalize_chunks

        if not _DASK_GTE_018:
            raise NotImplementedError("Automatic chunking requires dask >= 0.18.0")
        block_shape = (1,) + riods.block_shapes[0]
        chunks = normalize_chunks(
            chunks=(1, "auto", "auto"),
            shape=(riods.count, riods.height, riods.width),
            dtype=_rasterio_to_numpy_dtype(riods.dtypes),
            previous_chunks=tuple((c,) for c in block_shape),
        )
    token = tokenize(filename, mtime, chunks)
    name_prefix = f"open_rasterio-{token}"
    return result.chunk(chunks, name_prefix=name_prefix, token=token)


def _handle_encoding(
    result: DataArray,
    mask_and_scale: bool,
    masked: bool,
    da_name: Optional[Hashable],
    unsigned: Union[bool, None],
) -> None:
    """
    Make sure encoding handled properly
    """
    if "grid_mapping" in result.attrs:
        variables.pop_to(result.attrs, result.encoding, "grid_mapping", name=da_name)
    if mask_and_scale:
        if "scale_factor" in result.attrs:
            variables.pop_to(
                result.attrs, result.encoding, "scale_factor", name=da_name
            )
        if "add_offset" in result.attrs:
            variables.pop_to(result.attrs, result.encoding, "add_offset", name=da_name)
    if masked:
        if "_FillValue" in result.attrs:
            variables.pop_to(result.attrs, result.encoding, "_FillValue", name=da_name)
        if "missing_value" in result.attrs:
            variables.pop_to(
                result.attrs, result.encoding, "missing_value", name=da_name
            )

    if mask_and_scale and unsigned is not None and "_FillValue" in result.encoding:
        unsigned_dtype = _get_unsigned_dtype(
            unsigned=unsigned,
            dtype=result.encoding["dtype"],
        )
        if unsigned_dtype is not None:
            result.encoding["_FillValue"] = unsigned_dtype.type(
                result.encoding["_FillValue"]
            )


def _single_band_open(*args, bidx=0, **kwargs):
    """
    Open file as if it only has a single band
    """
    return SingleBandDatasetReader(
        riods=rasterio.open(*args, **kwargs),
        bidx=bidx,
    )


def open_rasterio(
    filename: Union[
        str,
        os.PathLike,
        rasterio.io.DatasetReader,
        rasterio.vrt.WarpedVRT,
        SingleBandDatasetReader,
    ],
    parse_coordinates: Optional[bool] = None,
    chunks: Optional[Union[int, tuple, dict]] = None,
    cache: Optional[bool] = None,
    lock: Optional[Any] = None,
    masked: bool = False,
    mask_and_scale: bool = False,
    variable: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    group: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    default_name: Optional[str] = None,
    decode_times: bool = True,
    decode_timedelta: Optional[bool] = None,
    band_as_variable: bool = False,
    **open_kwargs,
) -> Union[Dataset, DataArray, list[Dataset]]:
    # pylint: disable=too-many-statements,too-many-locals,too-many-branches
    """Open a file with rasterio (experimental).

    This should work with any file that rasterio can open (most often:
    geoTIFF). The x and y coordinates are generated automatically from the
    file's geoinformation, shifted to the center of each pixel (see
    `"PixelIsArea" Raster Space
    <http://web.archive.org/web/20160326194152/http://remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2>`_
    for more information).

    .. versionadded:: 0.13 band_as_variable

    Parameters
    ----------
    filename: str, rasterio.io.DatasetReader, or rasterio.vrt.WarpedVRT
        Path to the file to open. Or already open rasterio dataset.
    parse_coordinates: bool, optional
        Whether to parse the x and y coordinates out of the file's
        ``transform`` attribute or not. The default is to automatically
        parse the coordinates only if they are rectilinear (1D).
        It can be useful to set ``parse_coordinates=False``
        if your files are very large or if you don't need the coordinates.
    chunks: int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``. If chunks is provided, it used to load the new
        DataArray into a dask array. Chunks can also be set to
        ``True`` or ``"auto"`` to choose sensible chunk sizes according to
        ``dask.config.get("array.chunk-size")``.
    cache: bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False.
    lock: bool or dask.utils.SerializableLock, optional

        If chunks is provided, this argument is used to ensure that only one
        thread per process is reading from a rasterio file object at a time.

        By default and when a lock instance is provided,
        a :class:`xarray.backends.CachingFileManager` is used to cache File objects.
        Since rasterio also caches some data, this will make repeated reads from the
        same object fast.

        When ``lock=False``, no lock is used, allowing for completely parallel reads
        from multiple threads or processes. However, a new file handle is opened on
        each request.

    masked: bool, optional
        If True, read the mask and set values to NaN. Defaults to False.
    mask_and_scale: bool, default=False
        Lazily scale (using the `scales` and `offsets` from rasterio) and mask.
        If the _Unsigned attribute is present treat integer arrays as unsigned.
    variable: str or list or tuple, optional
        Variable name or names to use to filter loading.
    group: str or list or tuple, optional
        Group name or names to use to filter loading.
    default_name: str, optional
        The name of the data array if none exists. Default is None.
    decode_times: bool, default=True
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    decode_timedelta: bool, optional
        If True, decode variables and coordinates with time units in
        {“days”, “hours”, “minutes”, “seconds”, “milliseconds”, “microseconds”}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
    band_as_variable: bool, default=False
        If True, will load bands in a raster to separate variables.
    **open_kwargs: kwargs, optional
        Optional keyword arguments to pass into :func:`rasterio.open`.

    Returns
    -------
    :obj:`xarray.Dataset` | :obj:`xarray.DataArray` | list[:obj:`xarray.Dataset`]:
        The newly created dataset(s).
    """
    parse_coordinates = True if parse_coordinates is None else parse_coordinates
    masked = masked or mask_and_scale
    vrt_params = None
    file_opener = rasterio.open
    if isinstance(filename, SingleBandDatasetReader):
        file_opener = functools.partial(
            _single_band_open,
            bidx=filename._bidx,
        )
        vrt_params = filename._vrt_params
    if isinstance(filename, (rasterio.io.DatasetReader, SingleBandDatasetReader)):
        filename = filename.name
    elif isinstance(filename, rasterio.vrt.WarpedVRT):
        vrt = filename
        filename = vrt.src_dataset.name
        vrt_params = {
            "src_crs": vrt.src_crs.to_string() if vrt.src_crs else None,
            "crs": vrt.dst_crs.to_string() if vrt.dst_crs else None,
            "resampling": vrt.resampling,
            "tolerance": vrt.tolerance,
            "src_nodata": vrt.src_nodata,
            "nodata": vrt.dst_nodata,
            "width": vrt.dst_width,
            "height": vrt.dst_height,
            "src_transform": vrt.src_transform,
            "transform": vrt.dst_transform,
            "dtype": vrt.working_dtype,
            **vrt.warp_extras,
        }

    if lock in (True, None):
        lock = RASTERIO_LOCK
    elif lock is False:
        lock = NO_LOCK

    # ensure default for sharing is False
    # ref https://github.com/mapbox/rasterio/issues/1504
    open_kwargs["sharing"] = open_kwargs.get("sharing", False)

    with warnings.catch_warnings(record=True) as rio_warnings:
        if lock is not NO_LOCK and isinstance(filename, (str, os.PathLike)):
            manager: FileManager = CachingFileManager(
                file_opener, filename, lock=lock, mode="r", kwargs=open_kwargs
            )
        else:
            manager = URIManager(file_opener, filename, mode="r", kwargs=open_kwargs)
        riods = manager.acquire()
        captured_warnings = rio_warnings.copy()

    # raise the NotGeoreferencedWarning if applicable
    for rio_warning in captured_warnings:
        if not riods.subdatasets or not isinstance(
            rio_warning.message, NotGeoreferencedWarning
        ):
            warnings.warn(str(rio_warning.message), type(rio_warning.message))  # type: ignore

    # open the subdatasets if they exist
    if riods.subdatasets:
        subdataset_result = _load_subdatasets(
            riods=riods,
            group=group,
            variable=variable,
            parse_coordinates=parse_coordinates,
            chunks=chunks,
            cache=cache,
            lock=lock,
            masked=masked,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            decode_timedelta=decode_timedelta,
            **open_kwargs,
        )
        manager.close()
        return subdataset_result

    if band_as_variable:
        dataset_result = _load_bands_as_variables(
            riods=riods,
            parse_coordinates=parse_coordinates,
            chunks=chunks,
            cache=cache,
            lock=lock,
            masked=masked,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            decode_timedelta=decode_timedelta,
            vrt_params=vrt_params,
            **open_kwargs,
        )
        manager.close()
        return dataset_result

    if cache is None:
        cache = chunks is None

    riods = _ensure_warped_vrt(riods, vrt_params)

    # Get bands
    if riods.count < 1:
        raise ValueError("Unknown dims")

    # parse tags & load alternate coords
    attrs = _get_rasterio_attrs(riods=riods)
    coords = _load_netcdf_1d_coords(attrs)
    _parse_driver_tags(riods=riods, attrs=attrs, coords=coords)
    for coord in coords:
        if f"NETCDF_DIM_{coord}" in attrs:
            coord_name = coord
            attrs.pop(f"NETCDF_DIM_{coord}")
            break
        if f"NETCDF_DIM_{coord}_VALUES" in attrs:
            coord_name = coord
            attrs.pop(f"NETCDF_DIM_{coord}_VALUES")
            attrs.pop(f"NETCDF_DIM_{coord}_DEF", None)
            attrs.pop("NETCDF_DIM_EXTRA", None)
            break
    else:
        coord_name = "band"
        coords[coord_name] = numpy.asarray(riods.indexes)

    has_gcps = riods.gcps[0]
    if has_gcps:
        parse_coordinates = False

    # Get geospatial coordinates
    if parse_coordinates:
        coords.update(
            _generate_spatial_coords(riods.transform, riods.width, riods.height)
        )

    unsigned = None
    encoding: dict[Hashable, Any] = {}
    if mask_and_scale and "_Unsigned" in attrs:
        unsigned = variables.pop_to(attrs, encoding, "_Unsigned") == "true"

    if masked:
        encoding["dtype"] = str(_rasterio_to_numpy_dtype(riods.dtypes))

    da_name = attrs.pop("NETCDF_VARNAME", default_name)
    data: Any = indexing.LazilyOuterIndexedArray(
        RasterioArrayWrapper(
            manager,
            lock,
            name=da_name,
            vrt_params=vrt_params,
            masked=masked,
            mask_and_scale=mask_and_scale,
            unsigned=unsigned,
        )
    )

    # this lets you write arrays loaded with rasterio
    data = indexing.CopyOnWriteArray(data)
    if cache and chunks is None:
        data = indexing.MemoryCachedArray(data)

    result = DataArray(
        data=data, dims=(coord_name, "y", "x"), coords=coords, attrs=attrs, name=da_name
    )
    result.encoding = encoding

    # update attributes from NetCDF attributes
    _load_netcdf_attrs(riods.tags(), result)
    result = _decode_datetime_cf(
        result, decode_times=decode_times, decode_timedelta=decode_timedelta
    )

    # make sure the _FillValue is correct dtype
    if "_FillValue" in result.attrs:
        result.attrs["_FillValue"] = result.dtype.type(result.attrs["_FillValue"])

    # handle encoding
    _handle_encoding(result, mask_and_scale, masked, da_name, unsigned=unsigned)
    # Affine transformation matrix (always available)
    # This describes coefficients mapping pixel coordinates to CRS
    # For serialization store as tuple of 6 floats, the last row being
    # always (0, 0, 1) per definition (see
    # https://github.com/sgillies/affine)
    result.rio.write_transform(riods.transform, inplace=True)
    rio_crs = riods.crs or result.rio.crs
    if rio_crs:
        result.rio.write_crs(rio_crs, inplace=True)
    if has_gcps:
        result.rio.write_gcps(*riods.gcps, inplace=True)

    if chunks is not None:
        result = _prepare_dask(result, riods, filename, chunks)
    else:
        result.encoding["preferred_chunks"] = {
            result.rio.y_dim: riods.block_shapes[0][0],
            result.rio.x_dim: riods.block_shapes[0][1],
            coord_name: 1,
        }

    # add file path to encoding
    result.encoding["source"] = riods.name
    result.encoding["rasterio_dtype"] = str(riods.dtypes[0])
    # remove duplicate coordinate information
    for coord in result.coords:
        result.attrs = {
            attr: value
            for attr, value in result.attrs.items()
            if not attr.startswith(f"{coord}#")
        }
    # remove duplicate tags
    if result.name:
        result.attrs = {
            attr: value
            for attr, value in result.attrs.items()
            if not attr.startswith(f"{result.name}#")
        }
    # Make the file closeable
    result.set_close(manager.close)
    result.rio._manager = manager
    return result
