"""

Credits:

This file was adopted from: https://github.com/pydata/xarray # noqa
Source file: https://github.com/pydata/xarray/blob/1d7bcbdc75b6d556c04e2c7d7a042e4379e15303/xarray/backends/rasterio_.py # noqa
"""

import os
import re
import warnings
from collections import OrderedDict
from distutils.version import LooseVersion

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from xarray import DataArray, Dataset
from xarray.backends.common import BackendArray
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.locks import SerializableLock
from xarray.core import indexing
from xarray.core.utils import is_scalar

from rioxarray.exceptions import RioXarrayError
from rioxarray.rioxarray import affine_to_coords

# TODO: should this be GDAL_LOCK instead?
RASTERIO_LOCK = SerializableLock()


class RasterioArrayWrapper(BackendArray):
    """A wrapper around rasterio dataset objects"""

    def __init__(self, manager, lock, vrt_params=None, masked=False):
        from rasterio.vrt import WarpedVRT

        self.manager = manager
        self.lock = lock
        self.masked = masked

        # cannot save riods as an attribute: this would break pickleability
        riods = manager.acquire()
        if vrt_params is not None:
            riods = WarpedVRT(riods, **vrt_params)
        self.vrt_params = vrt_params
        self._shape = (riods.count, riods.height, riods.width)

        dtypes = riods.dtypes
        if not np.all(np.asarray(dtypes) == dtypes[0]):
            raise ValueError("All bands should have the same dtype")

        self._dtype = np.dtype("float64") if self.masked else np.dtype(dtypes[0])

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def _get_indexer(self, key):
        """ Get indexer for rasterio array.

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
            band_key = np.arange(start, stop, step)
        # be sure we give out a list
        band_key = (np.asarray(band_key) + 1).tolist()
        if isinstance(band_key, list):  # if band_key is not a scalar
            np_inds.append(slice(None))

        # but other dims can only be windowed
        window = []
        squeeze_axis = []
        for i, (k, n) in enumerate(zip(key[1:], self.shape[1:])):
            if isinstance(k, slice):
                # step is always positive. see indexing.decompose_indexer
                start, stop, step = k.indices(n)
                np_inds.append(slice(None, None, step))
            elif is_scalar(k):
                # windowed operations will always return an array
                # we will have to squeeze it later
                squeeze_axis.append(-(2 - i))
                start = k
                stop = k + 1
            else:
                start, stop = np.min(k), np.max(k) + 1
                np_inds.append(k - start)
            window.append((start, stop))

        if isinstance(key[1], np.ndarray) and isinstance(key[2], np.ndarray):
            # do outer-style indexing
            np_inds[-2:] = np.ix_(*np_inds[-2:])

        return band_key, tuple(window), tuple(squeeze_axis), tuple(np_inds)

    def _getitem(self, key):
        from rasterio.vrt import WarpedVRT

        band_key, window, squeeze_axis, np_inds = self._get_indexer(key)

        if not band_key or any(start == stop for (start, stop) in window):
            # no need to do IO
            shape = (len(band_key),) + tuple(stop - start for (start, stop) in window)
            out = np.zeros(shape, dtype=self.dtype)
        else:
            with self.lock:
                riods = self.manager.acquire(needs_lock=False)
                if self.vrt_params is not None:
                    riods = WarpedVRT(riods, **self.vrt_params)
                out = riods.read(band_key, window=window, masked=self.masked)
                if self.masked:
                    out = np.ma.filled(out.astype(self.dtype), np.nan)

        if squeeze_axis:
            out = np.squeeze(out, axis=squeeze_axis)
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

    def parsevec(s):
        return np.fromstring(s.strip("{}"), dtype="float", sep=",")

    def default(s):
        return s.strip("{}")

    parse = {"wavelength": parsevec, "fwhm": parsevec}
    parsed_meta = {k: parse.get(k, default)(v) for k, v in meta.items()}
    return parsed_meta


def _parse_tags(tags):
    def parsevec(s):
        return np.fromstring(s.strip("{}"), dtype="float", sep=",")

    parsed_tags = {}
    for key, value in tags.items():
        if value.startswith("{") and value.endswith("}"):
            new_val = parsevec(value)
            value = new_val if len(new_val) else value
        else:
            try:
                value = int(value)
            except (TypeError, ValueError):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
        parsed_tags[key] = value
    return parsed_tags


def build_subdataset_filter(group_names, variable_names):
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


def _rio_transform(riods):
    """
    Get the transform from a rasterio dataset
    reguardless of rasterio version.
    """
    try:
        return riods.transform
    except AttributeError:
        return riods.affine  # rasterio < 1.0


def _get_rasterio_attrs(riods, masked):
    """
    Get rasterio specific attributes/encoding
    """
    # Add rasterio attributes
    attrs = _parse_tags(riods.tags(1))
    encoding = dict()
    # Affine transformation matrix (always available)
    # This describes coefficients mapping pixel coordinates to CRS
    # For serialization store as tuple of 6 floats, the last row being
    # always (0, 0, 1) per definition (see
    # https://github.com/sgillies/affine)
    attrs["transform"] = tuple(_rio_transform(riods))[:6]
    if hasattr(riods, "nodata") and riods.nodata is not None:
        # The nodata values for the raster bands
        if masked:
            encoding["_FillValue"] = riods.nodata
        else:
            attrs["_FillValue"] = riods.nodata
    if hasattr(riods, "scales"):
        # The scale values for the raster bands
        attrs["scales"] = riods.scales
    if hasattr(riods, "offsets"):
        # The offset values for the raster bands
        attrs["offsets"] = riods.offsets
    if hasattr(riods, "descriptions") and any(riods.descriptions):
        if len(set(riods.descriptions)) == 1:
            attrs["long_name"] = riods.descriptions[0]
        else:
            # Descriptions for each dataset band
            attrs["long_name"] = riods.descriptions
    if hasattr(riods, "units") and any(riods.units):
        # A list of units string for each dataset band
        if len(riods.units) == 1:
            attrs["units"] = riods.units[0]
        else:
            attrs["units"] = riods.units

    return attrs, encoding


def _parse_driver_tags(riods, attrs, coords):
    # Parse extra metadata from tags, if supported
    parsers = {"ENVI": _parse_envi}

    driver = riods.driver
    if driver in parsers:
        meta = parsers[driver](riods.tags(ns=driver))

        for k, v in meta.items():
            # Add values as coordinates if they match the band count,
            # as attributes otherwise
            if isinstance(v, (list, np.ndarray)) and len(v) == riods.count:
                coords[k] = ("band", np.asarray(v))
            else:
                attrs[k] = v


def _load_subdatasets(
    riods, group, variable, parse_coordinates, chunks, cache, lock, masked
):
    """
    Load in rasterio subdatasets
    """
    base_tags = _parse_tags(riods.tags())
    dim_groups = {}
    subdataset_filter = None
    if any((group, variable)):
        subdataset_filter = build_subdataset_filter(group, variable)
    for iii, subdataset in enumerate(riods.subdatasets):
        if subdataset_filter is not None and not subdataset_filter.match(subdataset):
            continue
        with rasterio.open(subdataset) as rds:
            shape = rds.shape
        rioda = open_rasterio(
            subdataset,
            parse_coordinates=shape not in dim_groups and parse_coordinates,
            chunks=chunks,
            cache=cache,
            lock=lock,
            masked=masked,
            default_name=subdataset.split(":")[-1].lstrip("/").replace("/", "_"),
        )
        if shape not in dim_groups:
            dim_groups[shape] = {rioda.name: rioda}
        else:
            dim_groups[shape][rioda.name] = rioda

    if len(dim_groups) > 1:
        dataset = [
            Dataset(dim_group, attrs=base_tags) for dim_group in dim_groups.values()
        ]
    elif not dim_groups:
        dataset = Dataset(attrs=base_tags)
    else:
        dataset = Dataset(list(dim_groups.values())[0], attrs=base_tags)
    return dataset


def _prepare_dask(result, riods, filename, chunks):
    """
    Prepare the data for dask computations
    """
    from dask.base import tokenize

    # augment the token with the file modification time
    try:
        mtime = os.path.getmtime(filename)
    except OSError:
        # the filename is probably an s3 bucket rather than a regular file
        mtime = None

    if chunks in (True, "auto"):
        from dask.array.core import normalize_chunks
        import dask

        if LooseVersion(dask.__version__) < LooseVersion("0.18.0"):
            msg = (
                "Automatic chunking requires dask.__version__ >= 0.18.0 . "
                "You currently have version %s" % dask.__version__
            )
            raise NotImplementedError(msg)
        block_shape = (1,) + riods.block_shapes[0]
        chunks = normalize_chunks(
            chunks=(1, "auto", "auto"),
            shape=(riods.count, riods.height, riods.width),
            dtype=riods.dtypes[0],
            previous_chunks=tuple((c,) for c in block_shape),
        )
    token = tokenize(filename, mtime, chunks)
    name_prefix = "open_rasterio-%s" % token
    return result.chunk(chunks, name_prefix=name_prefix, token=token)


def open_rasterio(
    filename,
    parse_coordinates=None,
    chunks=None,
    cache=None,
    lock=None,
    masked=False,
    variable=None,
    group=None,
    default_name=None,
    **open_kwargs,
):
    """Open a file with rasterio (experimental).

    This should work with any file that rasterio can open (most often:
    geoTIFF). The x and y coordinates are generated automatically from the
    file's geoinformation, shifted to the center of each pixel (see
    `"PixelIsArea" Raster Space
    <http://web.archive.org/web/20160326194152/http://remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2>`_
    for more information).

    You can generate 2D coordinates from the file's attributes with::

        from affine import Affine
        da = xr.open_rasterio('path_to_file.tif')
        transform = Affine.from_gdal(*da.attrs['transform'])
        nx, ny = da.sizes['x'], da.sizes['y']
        x, y = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5) * transform


    Parameters
    ----------
    filename: str, rasterio.DatasetReader, or rasterio.WarpedVRT
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
        ``dask.config.get("array.chunk-size").
    cache: bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False.
    lock: False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used to avoid issues with concurrent access to the same file when using
        dask's multithreaded backend.
    masked: bool, optional
        If True, read the mask and to set values to NaN. Defaults to False.
    variable: str or list or tuple, optional
        Variable name or names to use to filter loading.
    group: str or list or tuple, optional
        Group name or names to use to filter loading.
    default_name: str, optional
        The name of the data array if none exists. Default is None.
    **open_kwargs: kwargs, optional
        Optional keyword arguments to pass into rasterio.open().

    Returns
    -------
    data : DataArray
        The newly created DataArray.
    """
    parse_coordinates = True if parse_coordinates is None else parse_coordinates
    vrt_params = None
    if isinstance(filename, rasterio.io.DatasetReader):
        filename = filename.name
    elif isinstance(filename, rasterio.vrt.WarpedVRT):
        vrt = filename
        filename = vrt.src_dataset.name
        vrt_params = dict(
            crs=vrt.crs.to_string(),
            resampling=vrt.resampling,
            src_nodata=vrt.src_nodata,
            dst_nodata=vrt.dst_nodata,
            tolerance=vrt.tolerance,
            transform=vrt.transform,
            width=vrt.width,
            height=vrt.height,
            warp_extras=vrt.warp_extras,
        )

    if lock is None:
        lock = RASTERIO_LOCK

    # ensure default for sharing is False
    # ref https://github.com/mapbox/rasterio/issues/1504
    open_kwargs["sharing"] = open_kwargs.get("sharing", False)
    manager = CachingFileManager(
        rasterio.open, filename, lock=lock, mode="r", kwargs=open_kwargs
    )
    riods = manager.acquire()
    # open the subdatasets if they exist
    if riods.subdatasets:
        return _load_subdatasets(
            riods=riods,
            group=group,
            variable=variable,
            parse_coordinates=parse_coordinates,
            chunks=chunks,
            cache=cache,
            lock=lock,
            masked=masked,
        )

    if vrt_params is not None:
        riods = WarpedVRT(riods, **vrt_params)

    if cache is None:
        cache = chunks is None

    # Get bands
    if riods.count < 1:
        raise ValueError("Unknown dims")
    coords = OrderedDict()
    coords["band"] = np.asarray(riods.indexes)

    # parse tags
    attrs, encoding = _get_rasterio_attrs(riods=riods, masked=masked)
    _parse_driver_tags(riods=riods, attrs=attrs, coords=coords)

    # Get geospatial coordinates
    transform = _rio_transform(riods)
    if parse_coordinates and transform.is_rectilinear:
        # 1d coordinates
        coords.update(affine_to_coords(riods.transform, riods.width, riods.height))
    elif parse_coordinates:
        # 2d coordinates
        warnings.warn(
            "The file coordinates' transformation isn't "
            "rectilinear: xarray won't parse the coordinates "
            "in this case. Set `parse_coordinates=False` to "
            "suppress this warning.",
            RuntimeWarning,
            stacklevel=3,
        )

    data = indexing.LazilyOuterIndexedArray(
        RasterioArrayWrapper(manager, lock, vrt_params, masked=masked)
    )

    # this lets you write arrays loaded with rasterio
    data = indexing.CopyOnWriteArray(data)
    if cache and chunks is None:
        data = indexing.MemoryCachedArray(data)

    # create the output data array
    da_name = attrs.pop("NETCDF_VARNAME", default_name)
    result = DataArray(
        data=data, dims=("band", "y", "x"), coords=coords, attrs=attrs, name=da_name
    )
    result.encoding = encoding

    if hasattr(riods, "crs") and riods.crs:
        result.rio.write_crs(riods.crs, inplace=True)

    if chunks is not None:
        result = _prepare_dask(result, riods, filename, chunks)

    # Make the file closeable
    result._file_obj = manager

    return result
