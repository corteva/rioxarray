"""
This module allows you to merge xarray Datasets/DataArrays
geospatially with the `rasterio.merge` module.
"""

from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy
from rasterio.crs import CRS
from rasterio.merge import merge as _rio_merge
from xarray import DataArray, Dataset

from rioxarray.rioxarray import _get_nonspatial_coords, _make_coords


class RasterioDatasetDuck:
    """
    This class is to provide the attributes and methods necessary
    to make the :func:`rasterio.merge.merge` function think that
    the :obj:`xarray.DataArray` is a :obj:`rasterio.io.DatasetReader`.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, xds: DataArray):
        self._xds = xds
        self.crs = xds.rio.crs
        self.bounds = xds.rio.bounds(recalc=True)
        self.count = int(xds.rio.count)
        self.dtypes = [xds.dtype]
        self.name = xds.name
        self.nodatavals = [xds.rio.nodata]
        res = xds.rio.resolution(recalc=True)
        self.res = (abs(res[0]), abs(res[1]))
        self.transform = xds.rio.transform(recalc=True)
        try:
            rio_file = xds.rio._manager.acquire()
            self.profile = rio_file.profile
        except AttributeError:
            self.profile = {}
        self.profile.update(
            dtype=xds.dtype,
            crs=xds.rio.crs,
            nodata=xds.rio.nodata,
        )

    def colormap(self, *args, **kwargs):
        """
        Lazy load colormap through _manager.acquire()
        for the scenario many file handles are opened

        See: https://github.com/corteva/rioxarray/issues/479
        """
        try:
            rio_file = self.xds.rio._manager.acquire()
            return rio_file.colormap(*args, **kwargs)
        except AttributeError:
            return None

    def read(self, window, out_shape, *args, **kwargs) -> numpy.ma.MaskedArray:
        # pylint: disable=unused-argument
        """
        This method is meant to be used by the rasterio.merge.merge function.
        """
        data_window = self._xds.rio.isel_window(window)
        if data_window.shape != out_shape:
            # in this section, the data is geographically the same
            # however it is not the same dimensions as requested
            # so need to resample to the requested shape
            if len(out_shape) == 3:
                _, out_height, out_width = out_shape
            else:
                out_height, out_width = out_shape
            data_window = self._xds.rio.reproject(
                self._xds.rio.crs,
                transform=self.transform,
                shape=(out_height, out_width),
            )

        nodata = self.nodatavals[0]
        mask = False
        fill_value = None
        if nodata is not None and numpy.isnan(nodata):
            mask = numpy.isnan(data_window)
        elif nodata is not None:
            mask = data_window == nodata
            fill_value = nodata

        # make sure the returned shape matches
        # the expected shape. This can be the case
        # when the xarray dataset was squeezed to 2D beforehand
        if len(out_shape) == 3 and len(data_window.shape) == 2:
            data_window = data_window.values.reshape((1, out_height, out_width))

        return numpy.ma.array(
            data_window, mask=mask, fill_value=fill_value, dtype=self.dtypes[0]
        )


def merge_arrays(
    dataarrays: Sequence[DataArray],
    bounds: Optional[tuple] = None,
    res: Optional[tuple] = None,
    nodata: Optional[float] = None,
    precision: Optional[float] = None,
    method: Union[str, Callable, None] = None,
    crs: Optional[CRS] = None,
    parse_coordinates: bool = True,
) -> DataArray:
    """
    Merge data arrays geospatially.

    Uses :func:`rasterio.merge.merge`

    .. versionadded:: 0.2 crs

    Parameters
    ----------
    dataarrays: list[xarray.DataArray]
        List of xarray.DataArray's with all geo attributes.
        The first one is assumed to have the same
        CRS, dtype, and dimensions as the others in the array.
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).
        If not set, bounds are determined from bounds of input DataArrays.
    res: tuple, optional
        Output resolution in units of coordinate reference system.
        If not set, the resolution of the first DataArray is used.
        If a single value is passed, output pixels will be square.
    nodata: float, optional
        nodata value to use in output file.
        If not set, uses the nodata value in the first input DataArray.
    precision: float, optional
        Number of decimal points of precision when computing inverse transform.
    method: str or callable, optional
        See :func:`rasterio.merge.merge` for details.
    crs: rasterio.crs.CRS, optional
        Output CRS. If not set, the CRS of the first DataArray is used.
    parse_coordinates: bool, optional
        If False, it will disable loading spatial coordinates.

    Returns
    -------
    :obj:`xarray.DataArray`:
        The geospatially merged data.
    """
    input_kwargs = {
        "bounds": bounds,
        "res": res,
        "nodata": nodata,
        "precision": precision,
        "method": method,
    }

    if crs is None:
        crs = dataarrays[0].rio.crs
    if res is None:
        res = tuple(abs(res_val) for res_val in dataarrays[0].rio.resolution())

    # prepare the duck arrays
    rioduckarrays = []
    for dataarray in dataarrays:
        da_res = tuple(abs(res_val) for res_val in dataarray.rio.resolution())
        if da_res != res or dataarray.rio.crs != crs:
            rioduckarrays.append(
                RasterioDatasetDuck(
                    dataarray.rio.reproject(dst_crs=crs, resolution=res)
                )
            )
        else:
            rioduckarrays.append(RasterioDatasetDuck(dataarray))

    # use rasterio to merge
    merged_data, merged_transform = _rio_merge(
        rioduckarrays,
        **{key: val for key, val in input_kwargs.items() if val is not None},
    )
    # generate merged data array
    representative_array = rioduckarrays[0]._xds
    if parse_coordinates:
        coords = _make_coords(
            representative_array,
            merged_transform,
            merged_data.shape[-1],
            merged_data.shape[-2],
        )
    else:
        coords = _get_nonspatial_coords(representative_array)

    # make sure the output merged data shape is 2D if the
    # original data was 2D. this can happen if the
    # xarray datasarray was squeezed.
    if len(merged_data.shape) == 3 and len(representative_array.shape) == 2:
        merged_data = merged_data.squeeze()

    xda = DataArray(
        name=representative_array.name,
        data=merged_data,
        coords=coords,
        dims=tuple(representative_array.dims),
        attrs=representative_array.attrs,
    )
    xda.rio.write_nodata(
        nodata if nodata is not None else representative_array.rio.nodata, inplace=True
    )
    xda.rio.write_crs(representative_array.rio.crs, inplace=True)
    xda.rio.write_transform(merged_transform, inplace=True)
    return xda


def merge_datasets(
    datasets: Sequence[Dataset],
    bounds: Optional[tuple] = None,
    res: Optional[tuple] = None,
    nodata: Optional[float] = None,
    precision: Optional[float] = None,
    method: Union[str, Callable, None] = None,
    crs: Optional[CRS] = None,
) -> Dataset:
    """
    Merge datasets geospatially.

    Uses :func:`rasterio.merge.merge`

    .. versionadded:: 0.2 crs

    Parameters
    ----------
    datasets: list[xarray.Dataset]
        List of xarray.Dataset's with all geo attributes.
        The first one is assumed to have the same
        CRS, dtype, dimensions, and data_vars as the others in the array.
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).
        If not set, bounds are determined from bounds of input Dataset.
    res: tuple, optional
        Output resolution in units of coordinate reference system.
        If not set, the resolution of the first Dataset is used.
        If a single value is passed, output pixels will be square.
    nodata: float, optional
        nodata value to use in output file.
        If not set, uses the nodata value in the first input Dataset.
    precision: float, optional
        Number of decimal points of precision when computing inverse transform.
    method: str or callable, optional
        See rasterio docs.
    crs: rasterio.crs.CRS, optional
        Output CRS. If not set, the CRS of the first DataArray is used.

    Returns
    -------
    :obj:`xarray.Dataset`:
        The geospatially merged data.
    """

    representative_ds = datasets[0]
    merged_data = {}
    for data_var in representative_ds.data_vars:
        merged_data[data_var] = merge_arrays(
            [dataset[data_var] for dataset in datasets],
            bounds=bounds,
            res=res,
            nodata=nodata,
            precision=precision,
            method=method,
            crs=crs,
            parse_coordinates=False,
        )
    data_var = list(representative_ds.data_vars)[0]
    xds = Dataset(
        merged_data,
        coords=_make_coords(
            merged_data[data_var],
            merged_data[data_var].rio.transform(),
            merged_data[data_var].shape[-1],
            merged_data[data_var].shape[-2],
            force_generate=True,
        ),
        attrs=representative_ds.attrs,
    )
    xds.rio.write_crs(merged_data[data_var].rio.crs, inplace=True)
    return xds
