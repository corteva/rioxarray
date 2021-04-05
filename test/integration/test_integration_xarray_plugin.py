import os.path

import xarray as xr

from test.conftest import TEST_INPUT_DATA_DIR


def test_xarray_open_dataset():
    cog_file = os.path.join(TEST_INPUT_DATA_DIR, "cog.tif")

    ds = xr.open_dataset(cog_file)

    assert isinstance(ds, xr.Dataset)
    assert "band1" in ds.data_vars
    assert ds.data_vars["band1"].shape == (500, 500)
