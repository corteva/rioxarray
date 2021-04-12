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
    .. versionadded:: 0.4
    """

    def open_dataset(
        self,
        filename_or_obj,
        drop_variables=None,  # SKIP FROM XARRAY
        parse_coordinates=None,
        chunks=None,
        cache=None,
        lock=None,
        masked=False,
        mask_and_scale=True,
        variable=None,
        group=None,
        default_name="band_data",
        open_kwargs=None,
    ):
        if open_kwargs is None:
            open_kwargs = {}
        ds = _io.open_rasterio(
            filename_or_obj,
            mask_and_scale=mask_and_scale,
            parse_coordinates=parse_coordinates,
            lock=lock,
            masked=masked,
            variable=variable,
            group=group,
            default_name=default_name,
            **open_kwargs,
        )
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
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
