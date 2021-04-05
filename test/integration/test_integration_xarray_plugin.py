import os.path

import pytest

from test.conftest import TEST_INPUT_DATA_DIR

xr = pytest.importorskip("xarray", minversion="0.17.1.dev0")


def test_xarray_open_dataset():
    cog_file = os.path.join(TEST_INPUT_DATA_DIR, "cog.tif")

    ds = xr.open_dataset(cog_file, engine="gdal")

    assert isinstance(ds, xr.Dataset)
    assert "band1" in ds.data_vars
    assert ds.data_vars["band1"].shape == (500, 500)

    ds = xr.open_dataset(cog_file)

    assert isinstance(ds, xr.Dataset)
