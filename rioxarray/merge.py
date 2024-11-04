"""
This module allows you to merge xarray Datasets/DataArrays
geospatially with the `rasterio.merge` module.
"""
from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.merge import merge as _rio_merge
from xarray import DataArray, Dataset

from rioxarray._io import open_rasterio
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
        if xds.rio.encoded_nodata is not None:
            self.nodatavals = [xds.rio.encoded_nodata]
        else:
            self.nodatavals = [xds.rio.nodata]
        res = xds.rio.resolution(recalc=True)
        self.res = (abs(res[0]), abs(res[1]))
        self.transform = xds.rio.transform(recalc=True)
        self.profile: dict = {
            "crs": self.crs,
            "nodata": self.nodatavals[0],
        }
        valid_scale_factor = self._xds.encoding.get("scale_factor", 1) != 1 or any(
            scale != 1 for scale in self._xds.encoding.get("scales", (1,))
        )
        valid_offset = self._xds.encoding.get("add_offset", 0.0) != 0 or any(
            offset != 0 for offset in self._xds.encoding.get("offsets", (0,))
        )
        self._mask_and_scale = (
            self._xds.rio.encoded_nodata is not None
            or valid_scale_factor
            or valid_offset
            or self._xds.encoding.get("_Unsigned") is not None
        )

    def colormap(self, *args, **kwargs) -> None:
        """
        colormap is only used for writing to a file.
        This never happens with rioxarray merge.
        """
        # pylint: disable=unused-argument
        return None

    def read(self, *args, **kwargs) -> numpy.ma.MaskedArray:
        """
        This method is meant to be used by the rasterio.merge.merge function.
        """
        with MemoryFile() as memfile:
            self._xds.rio.to_raster(memfile.name)
            with memfile.open() as dataset:
                if self._mask_and_scale:
                    kwargs["masked"] = True
                out = dataset.read(*args, **kwargs)
                if self._mask_and_scale:
                    out = out.astype(self._xds.dtype)
                    for iii in range(self.count):
                        out[iii] = out[iii] * dataset.scales[iii] + dataset.offsets[iii]
                return out


def merge_arrays(
    dataarrays: Sequence[DataArray],
    *,
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
        List of multiple xarray.DataArray with all geo attributes.
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
    # generate merged data array
    representative_array = rioduckarrays[0]._xds
    with MemoryFile() as memfile:
        _rio_merge(
            rioduckarrays,
            **{key: val for key, val in input_kwargs.items() if val is not None},
            dst_path=memfile.name,
        )
        with open_rasterio(  # type: ignore
            memfile.name,
            parse_coordinates=parse_coordinates,
            mask_and_scale=rioduckarrays[0]._mask_and_scale,
        ) as xda:
            xda = xda.load()
        xda.coords.update(
            {
                coord: value
                for coord, value in _get_nonspatial_coords(representative_array).items()
                if coord not in xda.coords
            }
        )
    return xda  # type: ignore


def merge_datasets(
    datasets: Sequence[Dataset],
    *,
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
        List of multiple xarray.Dataset with all geo attributes.
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
            src_data_array=merged_data[data_var],
            dst_affine=merged_data[data_var].rio.transform(),
            dst_width=merged_data[data_var].shape[-1],
            dst_height=merged_data[data_var].shape[-2],
            force_generate=True,
        ),
        attrs=representative_ds.attrs,
    )
    xds.rio.write_crs(merged_data[data_var].rio.crs, inplace=True)
    return xds
