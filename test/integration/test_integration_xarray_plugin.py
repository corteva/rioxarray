import os.path

import pytest

from rioxarray.exceptions import RioXarrayError
from test.conftest import TEST_INPUT_DATA_DIR

xr = pytest.importorskip("xarray", minversion="0.18")


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


def test_xarray_open_dataset__drop_variables():
    input_file = os.path.join(
        TEST_INPUT_DATA_DIR, "MOD09GA.A2008296.h14v17.006.2015181011753.hdf"
    )

    rds = xr.open_dataset(
        input_file,
        engine="rasterio",
        group="MODIS_Grid_500m_2D",
        drop_variables=[
            "sur_refl_b01_1",
            "sur_refl_b02_1",
            "sur_refl_b03_1",
            "sur_refl_b04_1",
            "sur_refl_b05_1",
            "sur_refl_b06_1",
            "sur_refl_b07_1",
        ],
    )

    assert sorted(rds.data_vars) == [
        "QC_500m_1",
        "iobs_res_1",
        "num_observations_500m",
        "obscov_500m_1",
    ]


def test_open_multiple_resolution():
    with pytest.raises(
        RioXarrayError,
        match="Multiple resolution sets found. Use 'variable' or 'group' to filter.",
    ):
        xr.open_dataset(
            os.path.join(
                TEST_INPUT_DATA_DIR, "MOD09GA.A2008296.h14v17.006.2015181011753.hdf"
            ),
            engine="rasterio",
            drop_variables="QC_500m_1",
        )
