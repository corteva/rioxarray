import os.path

import xarray as xr

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


class RasterioBackend(xr.backends.common.BackendEntrypoint):
    """
    Requires xarray 0.18+

    .. versionadded:: 0.4
    """

    def open_dataset(
        self,
        filename_or_obj,
        drop_variables=None,
        parse_coordinates=None,
        chunks=None,
        cache=None,
        lock=None,
        masked=False,
        mask_and_scale=True,
        variable=None,
        group=None,
        default_name="band_data",
        decode_times=True,
        decode_timedelta=None,
        open_kwargs=None,
    ):
        if open_kwargs is None:
            open_kwargs = {}
        ds = _io.open_rasterio(
            filename_or_obj,
            parse_coordinates=parse_coordinates,
            chunks=chunks,
            cache=cache,
            lock=lock,
            masked=masked,
            mask_and_scale=mask_and_scale,
            variable=variable,
            group=group,
            default_name=default_name,
            decode_times=decode_times,
            decode_timedelta=decode_timedelta,
            **open_kwargs,
        )
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
        if drop_variables is not None:
            ds = ds.drop_vars(drop_variables)
        if not isinstance(ds, xr.Dataset):
            raise RioXarrayError(
                "Multiple resolution sets found. "
                "Use 'variable' or 'group' to filter."
            )
        return ds

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext[1:].lower() in CAN_OPEN_EXTS
