"""
This module allows for open_rasterio to be used with the xarray open methods
through a backend entrypoint.
"""
# pylint: disable=arguments-differ
import os.path

import xarray

from . import _io
from .exceptions import RioXarrayError

CAN_OPEN_EXTS = {
    "asc",
    "geotif",
    "geotiff",
    "img",
    "j2k",
    "jp2",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "vrt",
}


class RasterioBackend(xarray.backends.common.BackendEntrypoint):
    """
    Requires xarray 0.18+

    .. versionadded:: 0.4
    """

    def open_dataset(
        self,
        filename_or_obj,
        drop_variables=None,
        parse_coordinates=None,
        lock=None,
        masked=False,
        mask_and_scale=True,
        variable=None,
        group=None,
        default_name="band_data",
        decode_times=True,
        decode_timedelta=None,
        band_as_variable=False,
        open_kwargs=None,
    ):
        if open_kwargs is None:
            open_kwargs = {}
        rds = _io.open_rasterio(
            filename_or_obj,
            parse_coordinates=parse_coordinates,
            cache=False,
            lock=lock,
            masked=masked,
            mask_and_scale=mask_and_scale,
            variable=variable,
            group=group,
            default_name=default_name,
            decode_times=decode_times,
            decode_timedelta=decode_timedelta,
            band_as_variable=band_as_variable,
            **open_kwargs,
        )
        if isinstance(rds, xarray.DataArray):
            dataset = rds.to_dataset()
            dataset.set_close(rds._close)
            rds = dataset
        if isinstance(rds, list):
            for dataset in rds:
                dataset.close()
            raise RioXarrayError(
                "Multiple resolution sets found. "
                "Use 'variable' or 'group' to filter."
            )
        if drop_variables is not None:
            rds = rds.drop_vars(drop_variables)
        return rds

    def guess_can_open(self, filename_or_obj):  # pylint: disable=arguments-renamed
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext[1:].lower() in CAN_OPEN_EXTS
