import os.path

import xarray as xr

from . import _io


class GdalBackend(xr.backends.common.BackendEntrypoint):
    def open_dataset(self, filename_or_obj, drop_variables=None):
        ds = _io.open_rasterio(filename_or_obj).to_dataset("band")
        ds = ds.rename({idx: f"band{idx}" for idx in ds.data_vars})
        ds = ds.reset_coords("spatial_ref")
        return ds

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".tif", ".geotif", ".tiff", ".geotiff"}
