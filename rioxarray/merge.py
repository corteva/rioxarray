from typing import Callable, Iterable, Tuple, Union

import numpy
from rasterio.merge import merge as _rio_merge
from xarray import DataArray, Dataset

from rioxarray.rioxarray import _get_nonspatial_coords, _make_coords


class RasterioDatasetDuck:
    """
    This class is to provide the attributes and methods necessary
    to make the rasterio.merge.merge function think that
    the xarray.DataArray is a rasterio Dataset.
    """

    def __init__(self, xds: DataArray):
        self._xds = xds
        self.bounds = xds.rio.bounds(recalc=True)
        self.count = int(xds.rio.count)
        self.dtypes = [xds.dtype]
        self.name = xds.name
        self.nodatavals = [xds.rio.nodata]
        res = xds.rio.resolution(recalc=True)
        self.res = (abs(res[0]), abs(res[1]))
        self.transform = xds.rio.transform(recalc=True)

    def read(self, window, out_shape, *args, **kwargs) -> numpy.ma.array:
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

        return numpy.ma.array(
            data_window, mask=mask, fill_value=fill_value, dtype=self.dtypes[0]
        )


def merge_arrays(
    dataarrays: Iterable[DataArray],
    bounds: Union[Tuple, None] = None,
    res: Union[Tuple, None] = None,
    nodata: Union[float, None] = None,
    precision: Union[float, None] = None,
    method: Union[str, Callable, None] = None,
    parse_coordinates: bool = True,
) -> DataArray:
    """
    Merge data arrays geospatially.

    Uses rasterio.merge.merge:
        https://rasterio.readthedocs.io/en/stable/api/rasterio.merge.html#rasterio.merge.merge

    Parameters
    ----------
    dataarrays: list
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
        See rasterio docs.
    parse_coordinates: bool, optional
        If False, it will disable loading spatial coordinates.

    Returns
    -------
    :obj:`xarray.DataArray`:
        The geospatially merged data.
    """

    input_kwargs = dict(
        bounds=bounds, res=res, nodata=nodata, precision=precision, method=method
    )
    merged_data, merged_transform = _rio_merge(
        [RasterioDatasetDuck(dataarray) for dataarray in dataarrays],
        **{key: val for key, val in input_kwargs.items() if val is not None},
    )
    merged_shape = merged_data.shape
    representative_array = dataarrays[0]
    if parse_coordinates:
        coords = _make_coords(
            representative_array, merged_transform, merged_shape[-1], merged_shape[-2]
        )
    else:
        coords = _get_nonspatial_coords(representative_array)

    out_attrs = representative_array.attrs
    xda = DataArray(
        name=dataarrays[0].name,
        data=merged_data,
        coords=coords,
        dims=tuple(representative_array.dims),
        attrs=out_attrs,
    )
    out_nodata = nodata if nodata is not None else representative_array.rio.nodata
    xda.rio.write_nodata(out_nodata, inplace=True)
    xda.rio.write_crs(representative_array.rio.crs, inplace=True)
    xda.rio.write_transform(merged_transform, inplace=True)
    return xda


def merge_datasets(
    datasets: Iterable[Dataset],
    bounds: Union[Tuple, None] = None,
    res: Union[Tuple, None] = None,
    nodata: Union[float, None] = None,
    precision: Union[float, None] = None,
    method: Union[str, Callable, None] = None,
) -> DataArray:
    """
    Merge datasets geospatially.

    Uses rasterio.merge.merge:
        https://rasterio.readthedocs.io/en/stable/api/rasterio.merge.html#rasterio.merge.merge

    Parameters
    ----------
    datasets: list
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
        ),
        attrs=representative_ds.attrs,
    )
    xds.rio.write_crs(representative_ds.rio.crs, inplace=True)
    return xds
