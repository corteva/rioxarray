import os

import pytest
import xarray
from numpy import nansum
from numpy.testing import assert_almost_equal

from rioxarray import open_rasterio
from rioxarray.merge import merge_arrays, merge_datasets
from test.conftest import RASTERIO_GE_14, TEST_INPUT_DATA_DIR


@pytest.mark.parametrize("squeeze", [True, False])
def test_merge_arrays(squeeze):
    dem_test = os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")
    with open_rasterio(dem_test) as rds:
        rds.attrs = {
            "_FillValue": rds.rio.nodata,
        }
        arrays = [
            rds.isel(x=slice(100), y=slice(100)),
            rds.isel(x=slice(100, 200), y=slice(100, 200)),
            rds.isel(x=slice(100), y=slice(100, 200)),
            rds.isel(x=slice(100, 200), y=slice(100)),
        ]
        if squeeze:
            arrays = [array.squeeze() for array in arrays]
        merged = merge_arrays(arrays)

        assert_almost_equal(
            merged.rio.bounds(),
            (-7274009.6494863, 5003777.3385, -7227678.3778335, 5050108.6101528),
        )
        assert merged.rio.shape == (200, 200)
        assert_almost_equal(merged.sum(), 22865733)

        assert_almost_equal(
            tuple(merged.rio.transform()),
            (
                231.6563582639536,
                0.0,
                -7274009.649486291,
                0.0,
                -231.65635826374404,
                5050108.61015275,
                0.0,
                0.0,
                1.0,
            ),
        )
        assert merged.rio._cached_transform() == merged.rio.transform()
        assert sorted(merged.coords) == sorted(rds.coords)
        assert merged.coords["band"].values == [1]
        assert merged.rio.crs == rds.rio.crs
        assert merged.attrs == {
            "AREA_OR_POINT": "Area",
            "add_offset": 0.0,
            "scale_factor": 1.0,
            **rds.attrs,
        }
        assert merged.encoding["grid_mapping"] == "spatial_ref"


@pytest.mark.parametrize("dataset", [True, False])
def test_merge__different_crs(dataset):
    dem_test = os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")
    with (
        xarray.open_dataset(dem_test, mask_and_scale=False)
        if dataset
        else xarray.open_dataarray(dem_test, mask_and_scale=False)
    ) as rds:
        crs = rds.rio.crs
        arrays = [
            rds.isel(x=slice(100), y=slice(100)).rio.reproject("EPSG:3857"),
            rds.isel(x=slice(100, 200), y=slice(100, 200)),
            rds.isel(x=slice(100), y=slice(100, 200)),
            rds.isel(x=slice(100, 200), y=slice(100)),
        ]
        if dataset:
            merged = merge_datasets(arrays, crs=crs)
        else:
            merged = merge_arrays(arrays, crs=crs)

        if dataset:
            test_sum = merged[merged.rio.vars[0]].sum()
        else:
            test_sum = merged.sum()
        assert_almost_equal(
            merged.rio.bounds(),
            (-7300984.0238134, 5003618.5908794, -7224054.1109682, 5050108.6101528),
        )
        assert merged.rio.shape == (84, 139)
        if RASTERIO_GE_14:
            assert_almost_equal(test_sum, -126821853)
        else:
            assert_almost_equal(test_sum, -131734881)

        assert_almost_equal(
            tuple(merged.rio.transform()),
            (
                553.4526103969893,
                0.0,
                -7300984.023813409,
                0.0,
                -553.4526103969796,
                5050108.610152751,
                0.0,
                0.0,
                1.0,
            ),
        )
        assert sorted(merged.coords) == sorted(list(rds.coords) + ["spatial_ref"])
        assert merged.rio.crs == rds.rio.crs
        if not dataset:
            assert merged.attrs == {
                "AREA_OR_POINT": "Area",
                "_FillValue": -28672,
                "add_offset": 0.0,
                "scale_factor": 1.0,
            }
        assert merged.encoding["grid_mapping"] == "spatial_ref"


def test_merge_arrays__res():
    dem_test = os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")
    with open_rasterio(dem_test, masked=True) as rds:
        arrays = [
            rds.isel(x=slice(100), y=slice(100)),
            rds.isel(x=slice(100, 200), y=slice(100, 200)),
            rds.isel(x=slice(100), y=slice(100, 200)),
            rds.isel(x=slice(100, 200), y=slice(100)),
        ]
        merged = merge_arrays(arrays, res=(300, 300))

        assert_almost_equal(
            merged.rio.bounds(),
            (-7274009.6494863, 5003608.6101528, -7227509.6494863, 5050108.6101528),
        )
        assert merged.rio.shape == (155, 155)

        assert_almost_equal(
            tuple(merged.rio.transform()),
            (
                300.0,
                0.0,
                -7274009.649486291,
                0.0,
                -300.0,
                5050108.61015275,
                0.0,
                0.0,
                1.0,
            ),
        )
        assert merged.rio._cached_transform() == merged.rio.transform()
        assert sorted(merged.coords) == sorted(rds.coords)
        assert merged.coords["band"].values == [1]
        assert merged.rio.crs == rds.rio.crs
        assert_almost_equal(merged.rio.nodata, rds.rio.nodata)
        assert_almost_equal(merged.rio.encoded_nodata, rds.rio.encoded_nodata)
        assert merged.encoding["grid_mapping"] == "spatial_ref"
        assert_almost_equal(nansum(merged), 13760565)


@pytest.mark.xfail(os.name == "nt", reason="On windows the merged data is different.")
def test_merge_datasets():
    dem_test = os.path.join(
        TEST_INPUT_DATA_DIR, "MOD09GA.A2008296.h14v17.006.2015181011753.hdf"
    )
    with open_rasterio(dem_test, group="MODIS_Grid_500m_2D") as rds:
        datasets = [
            rds.isel(x=slice(600), y=slice(600)),
            rds.isel(x=slice(600, None), y=slice(600, None)),
            rds.isel(x=slice(600), y=slice(600, None)),
            rds.isel(x=slice(600, None), y=slice(600)),
        ]
        merged = merge_datasets(datasets)
        data_vars = sorted(merged.data_vars)
        assert data_vars == [
            "QC_500m_1",
            "iobs_res_1",
            "num_observations_500m",
            "obscov_500m_1",
            "sur_refl_b01_1",
            "sur_refl_b02_1",
            "sur_refl_b03_1",
            "sur_refl_b04_1",
            "sur_refl_b05_1",
            "sur_refl_b06_1",
            "sur_refl_b07_1",
        ]
        data_var = data_vars[0]
        assert_almost_equal(
            merged[data_var].rio.bounds(),
            (-4447802.078667, -10007554.677, -3335851.559, -8895604.157333),
        )
        assert merged.rio.shape == (2400, 2400)
        assert_almost_equal(
            tuple(merged[data_var].rio.transform()),
            (
                463.3127165279158,
                0.0,
                -4447802.078667,
                0.0,
                -463.3127165279151,
                -8895604.157333,
                0.0,
                0.0,
                1.0,
            ),
        )
        assert sorted(merged.coords) == sorted(rds.coords)
        assert merged.coords["band"].values == [1]
        assert merged.rio.crs == rds.rio.crs
        assert merged.attrs == rds.attrs
        assert merged.encoding["grid_mapping"] == "spatial_ref"
        assert_almost_equal(merged[data_var].sum(), 4539666606551516)


@pytest.mark.xfail(os.name == "nt", reason="On windows the merged data is different.")
def test_merge_datasets__res():
    dem_test = os.path.join(
        TEST_INPUT_DATA_DIR, "MOD09GA.A2008296.h14v17.006.2015181011753.hdf"
    )
    with open_rasterio(dem_test, group="MODIS_Grid_500m_2D") as rds:
        datasets = [
            rds.isel(x=slice(1200), y=slice(1200)),
            rds.isel(x=slice(1200, None), y=slice(1200, None)),
            rds.isel(x=slice(1200), y=slice(1200, None)),
            rds.isel(x=slice(1200, None), y=slice(1200)),
        ]
        merged = merge_datasets(datasets, res=1000)
        data_vars = sorted(merged.data_vars)
        assert data_vars == [
            "QC_500m_1",
            "iobs_res_1",
            "num_observations_500m",
            "obscov_500m_1",
            "sur_refl_b01_1",
            "sur_refl_b02_1",
            "sur_refl_b03_1",
            "sur_refl_b04_1",
            "sur_refl_b05_1",
            "sur_refl_b06_1",
            "sur_refl_b07_1",
        ]
        data_var = data_vars[0]
        assert_almost_equal(
            merged[data_var].rio.bounds(),
            (-4447802.078667, -10007604.157333, -3335802.078667, -8895604.157333),
        )
        assert_almost_equal(
            tuple(merged[data_var].rio.transform()),
            (
                1000.0,
                0.0,
                -4447802.078667,
                0.0,
                -1000.0,
                -8895604.157333,
                0.0,
                0.0,
                1.0,
            ),
        )
        assert merged.rio.shape == (1112, 1112)
        assert merged.coords["band"].values == [1]
        assert sorted(merged.coords) == sorted(rds.coords)
        assert merged.rio.crs == rds.rio.crs
        assert merged.attrs == rds.attrs
        assert merged.encoding["grid_mapping"] == "spatial_ref"
        assert_almost_equal(merged[data_var].sum(), 974566547463955)


@pytest.mark.parametrize("mask_and_scale", [True, False])
def test_merge_datasets__mask_and_scale(mask_and_scale):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc")
    with open_rasterio(test_file, mask_and_scale=mask_and_scale) as rds:
        rds = rds.to_dataset()
        datasets = [
            rds.isel(x=slice(100), y=slice(100)),
            rds.isel(x=slice(100, None), y=slice(100, None)),
            rds.isel(x=slice(100), y=slice(100, None)),
            rds.isel(x=slice(100, None), y=slice(100)),
        ]
        merged = merge_datasets(datasets)
        assert sorted(merged.coords) == sorted(list(rds.coords) + ["spatial_ref"])
        total = merged.air_temperature.sum()
        if mask_and_scale:
            assert_almost_equal(total, 133376696)
        else:
            assert_almost_equal(total, 10981781386)


def test_merge_datasets__preserve_dimension_names():
    sentinel_2_geographic = os.path.join(
        TEST_INPUT_DATA_DIR, "sentinel_2_L1C_geographic.nc"
    )
    with xarray.open_dataset(sentinel_2_geographic) as mda:
        merged = merge_datasets([mda])
        assert sorted(merged.coords) == sorted(mda.coords)
        for data_var in mda.data_vars:
            assert_almost_equal(merged[data_var].sum(), mda[data_var].sum())
        assert merged.rio.crs == mda.rio.crs
