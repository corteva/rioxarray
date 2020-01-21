import os
from distutils.version import LooseVersion

import pytest
from numpy import nansum
from numpy.testing import assert_almost_equal
from rasterio import gdal_version

from rioxarray import open_rasterio
from rioxarray.merge import merge_arrays, merge_datasets
from test.conftest import TEST_INPUT_DATA_DIR


def test_merge_arrays():
    dem_test = os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")
    with open_rasterio(dem_test) as rds:
        rds.attrs = {
            "_FillValue": rds.rio.nodata,
            "grid_mapping": "spatial_ref",
            "crs": rds.attrs["crs"],
        }
        arrays = [
            rds.isel(x=slice(100), y=slice(100)),
            rds.isel(x=slice(100, 200), y=slice(100, 200)),
            rds.isel(x=slice(100), y=slice(100, 200)),
            rds.isel(x=slice(100, 200), y=slice(100)),
        ]
        merged = merge_arrays(arrays)

    assert_almost_equal(
        merged.rio.bounds(),
        (-7274009.649486291, 5003545.682141737, -7227446.721475236, 5050108.61015275),
    )
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
    assert_almost_equal(
        merged.attrs.pop("transform"), tuple(merged.rio.transform())[:6]
    )
    assert merged.rio.shape == (201, 201)
    assert merged.coords["band"].values == [1]
    assert sorted(merged.coords) == ["band", "spatial_ref", "x", "y"]
    assert merged.rio.crs == rds.rio.crs
    assert merged.attrs == rds.attrs
    assert_almost_equal(merged.sum(), 11368261)


def test_merge_arrays__res():
    dem_test = os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")
    with open_rasterio(dem_test, masked=True) as rds:
        rds.attrs = {
            "_FillValue": rds.rio.nodata,
            "grid_mapping": "spatial_ref",
            "crs": rds.attrs["crs"],
        }
        arrays = [
            rds.isel(x=slice(100), y=slice(100)),
            rds.isel(x=slice(100, 200), y=slice(100, 200)),
            rds.isel(x=slice(100), y=slice(100, 200)),
            rds.isel(x=slice(100, 200), y=slice(100)),
        ]
        merged = merge_arrays(arrays, res=(300, 300))

    assert_almost_equal(
        merged.rio.bounds(),
        (-7274009.649486291, 5003608.61015275, -7227509.649486291, 5050108.61015275),
    )
    assert_almost_equal(
        tuple(merged.rio.transform()),
        (300.0, 0.0, -7274009.649486291, 0.0, -300.0, 5050108.61015275, 0.0, 0.0, 1.0),
    )
    assert_almost_equal(
        merged.attrs.pop("transform"), tuple(merged.rio.transform())[:6]
    )
    assert merged.rio.shape == (155, 155)
    assert merged.coords["band"].values == [1]
    assert sorted(merged.coords) == ["band", "spatial_ref", "x", "y"]
    assert merged.rio.crs == rds.rio.crs
    assert merged.attrs == rds.attrs
    if LooseVersion(gdal_version()) >= LooseVersion("2.4.4"):
        assert_almost_equal(nansum(merged), 13754521.430030823)
    else:
        assert_almost_equal(nansum(merged), 13767944)


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
        (-4447802.078667, -10008017.989716524, -3335388.246283474, -8895604.157333),
    )
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
    assert merged.rio.shape == (2401, 2401)
    assert merged.coords["band"].values == [1]
    assert sorted(merged.coords) == ["band", "spatial_ref", "x", "y"]
    assert merged.rio.crs == rds.rio.crs
    base_attrs = dict(rds.attrs)
    base_attrs["grid_mapping"] = "spatial_ref"
    assert merged.attrs == base_attrs
    assert_almost_equal(merged[data_var].sum(), 4539265823591471)


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
        (1000.0, 0.0, -4447802.078667, 0.0, -1000.0, -8895604.157333, 0.0, 0.0, 1.0),
    )
    assert merged.rio.shape == (1112, 1112)
    assert merged.coords["band"].values == [1]
    assert sorted(merged.coords) == ["band", "spatial_ref", "x", "y"]
    assert merged.rio.crs == rds.rio.crs
    base_attrs = dict(rds.attrs)
    base_attrs["grid_mapping"] = "spatial_ref"
    assert merged.attrs == base_attrs
    if LooseVersion(gdal_version()) >= LooseVersion("2.4.4"):
        assert_almost_equal(merged[data_var].sum(), 974565505482489)
    else:
        assert_almost_equal(merged[data_var].sum(), 974565970607345)
