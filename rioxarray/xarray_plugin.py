import os.path

import xarray as xr

from . import _io

CAN_OPEN_EXT = {"geotif", "geotiff", "j2k", "jp2", "tif", "tiff", "vrt"}


class GdalBackend(xr.backends.common.BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        drop_variables=None,
        mask_and_scale=False,
        parse_coordinates=None,
        lock=None,
        masked=False,
        variable=None,
        group=None,
        default_name="band_data",
    ):
        ds = _io.open_rasterio(
            filename_or_obj,
            mask_and_scale=mask_and_scale,
            parse_coordinates=parse_coordinates,
            lock=lock,
            masked=masked,
            variable=variable,
            group=group,
            default_name=default_name,
        )
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
        ds = ds.reset_coords("spatial_ref")
        return ds

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext[1:].lower() in CAN_OPEN_EXT
