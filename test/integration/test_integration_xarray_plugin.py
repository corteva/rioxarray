import os.path

import pytest

from test.conftest import TEST_INPUT_DATA_DIR

# FIXME: change to the next xarray version after release
xr = pytest.importorskip("xarray", minversion="0.17.1.dev0")


def test_xarray_open_dataset():
    cog_file = os.path.join(TEST_INPUT_DATA_DIR, "cog.tif")

    ds = xr.open_dataset(cog_file, engine="rasterio")

    assert isinstance(ds, xr.Dataset)
    assert "band_data" in ds.data_vars
    assert ds.data_vars["band_data"].shape == (1, 500, 500)
    assert "spatial_ref" not in ds.data_vars
    assert "spatial_ref" in ds.coords
    assert "grid_mapping" not in ds.data_vars["band_data"].attrs
    assert "grid_mapping" in ds.data_vars["band_data"].encoding

    ds = xr.open_dataset(cog_file)

    assert isinstance(ds, xr.Dataset)
