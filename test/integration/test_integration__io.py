import contextlib
import io
import itertools
import logging
import os
import pickle
import shutil
import sys
import tempfile
import warnings
from unittest.mock import patch

import dask.array
import numpy
import pytest
import rasterio
import xarray
from affine import Affine
from numpy.testing import assert_almost_equal, assert_array_equal
from packaging import version
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform
from xarray import DataArray
from xarray.testing import assert_allclose, assert_equal, assert_identical

import rioxarray
from rioxarray._io import build_subdataset_filter
from rioxarray.rioxarray import DEFAULT_GRID_MAP
from test.conftest import (
    GDAL_GE_3_11,
    GDAL_GE_36,
    GDAL_GE_364,
    TEST_COMPARE_DATA_DIR,
    TEST_INPUT_DATA_DIR,
    _assert_xarrays_equal,
    _ensure_dataset,
)
from test.integration.test_integration_rioxarray import (
    _check_rio_gcps,
    _create_gdal_gcps,
)


def _assert_tmmx_source(source):
    # https://github.com/OSGeo/gdal/issues/7695
    if GDAL_GE_364:
        assert source.endswith("tmmx_20190121.nc")
    else:
        assert source.startswith("netcdf:") and source.endswith(
            "tmmx_20190121.nc:air_temperature"
        )


@pytest.mark.parametrize(
    "subdataset, variable, group, match",
    [
        (
            "netcdf:../../test/test_data/input/PLANET_SCOPE_3D.nc:blue",
            "green",
            None,
            False,
        ),
        (
            "netcdf:../../test/test_data/input/PLANET_SCOPE_3D.nc:blue",
            "blue",
            None,
            True,
        ),
        (
            "netcdf:../../test/test_data/input/PLANET_SCOPE_3D.nc:blue1",
            "blue",
            None,
            False,
        ),
        (
            "netcdf:../../test/test_data/input/PLANET_SCOPE_3D.nc:1blue",
            "blue",
            None,
            False,
        ),
        (
            "netcdf:../../test/test_data/input/PLANET_SCOPE_3D.nc:blue",
            "blue",
            "gr",
            False,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            ":MODIS_Grid_2D:sur_refl_b01_1",
            ["sur_refl_b01_1"],
            None,
            True,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            ":MODIS_Grid_2D:sur_refl_b01_1",
            None,
            ["MODIS_Grid_2D"],
            True,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            ":MODIS_Grid_2D:sur_refl_b01_1",
            ("sur_refl_b01_1",),
            ("MODIS_Grid_2D",),
            True,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            ":MODIS_Grid_2D:sur_refl_b01_1",
            "blue",
            "gr",
            False,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            ":MODIS_Grid_2D:sur_refl_b01_1",
            "sur_refl_b01_1",
            "gr",
            False,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            ":MODIS_Grid_2D:sur_refl_b01_1",
            None,
            "gr",
            False,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            "://MODIS_Grid_2D://sur_refl_b01_1",
            "sur_refl_b01_1",
            None,
            True,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            "://MODIS_Grid_2D://sur_refl_b01_1",
            None,
            "MODIS_Grid_2D",
            True,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            "://MODIS_Grid_2D://sur_refl_b01_1",
            "sur_refl_b01_1",
            "MODIS_Grid_2D",
            True,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            "://MODIS_Grid_2D://sur_refl_b01_1",
            "blue",
            "gr",
            False,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            "://MODIS_Grid_2D://sur_refl_b01_1",
            "sur_refl_b01_1",
            "gr",
            False,
        ),
        (
            'HDF4_EOS:EOS_GRID:"./modis/MOD09GQ.A2017290.h11v04.006.NRT.hdf"'
            "://MODIS_Grid_2D://sur_refl_b01_1",
            None,
            "gr",
            False,
        ),
        (
            "netcdf:S5P_NRTI_L2__NO2____20190513T181819_20190513T182319_08191_"
            "01_010301_20190513T185033.nc:/PRODUCT/tm5_constant_a",
            None,
            "PRODUCT",
            True,
        ),
        (
            "netcdf:S5P_NRTI_L2__NO2____20190513T181819_20190513T182319_08191_"
            "01_010301_20190513T185033.nc:/PRODUCT/tm5_constant_a",
            "tm5_constant_a",
            "PRODUCT",
            True,
        ),
        (
            "netcdf:S5P_NRTI_L2__NO2____20190513T181819_20190513T182319_08191_"
            "01_010301_20190513T185033.nc:/PRODUCT/tm5_constant_a",
            "tm5_constant_a",
            "/PRODUCT",
            True,
        ),
    ],
)
def test_build_subdataset_filter(subdataset, variable, group, match):
    assert (
        build_subdataset_filter(group, variable).search(subdataset) is not None
    ) == match


def test_open_variable_filter(open_rasterio):
    with open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"), variable=["blue"]
    ) as rds:
        assert list(rds.data_vars) == ["blue"]


def test_open_group_filter__missing(open_rasterio):
    with open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"),
        variable="blue",
        group=["non-existent"],
    ) as rds:
        assert list(rds.data_vars) == []


def test_open_multiple_resolution():
    rds_list = rioxarray.open_rasterio(
        os.path.join(
            TEST_INPUT_DATA_DIR, "MOD09GA.A2008296.h14v17.006.2015181011753.hdf"
        )
    )
    assert isinstance(rds_list, list)
    assert len(rds_list) == 2
    assert rds_list[0].sizes == {"y": 1200, "x": 1200, "band": 1}
    assert rds_list[1].sizes == {"y": 2400, "x": 2400, "band": 1}
    for rds in rds_list:
        assert rds.attrs["SHORTNAME"] == "MOD09GA"
        rds.close()


def test_open_group_filter(open_rasterio):
    with open_rasterio(
        os.path.join(
            TEST_INPUT_DATA_DIR, "MOD09GA.A2008296.h14v17.006.2015181011753.hdf"
        ),
        group="MODIS_Grid_500m_2D",
    ) as rds:
        assert sorted(rds.data_vars) == [
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


def test_open_group_load_attrs(open_rasterio):
    with open_rasterio(
        os.path.join(
            TEST_INPUT_DATA_DIR, "MOD09GA.A2008296.h14v17.006.2015181011753.hdf"
        ),
        mask_and_scale=False,
        variable="sur_refl_b05_1",
    ) as rds:
        attrs = rds["sur_refl_b05_1"].attrs
        assert sorted(attrs) == [
            "Nadir Data Resolution",
            "_FillValue",
            "add_offset",
            "add_offset_err",
            "calibrated_nt",
            "long_name",
            "scale_factor",
            "scale_factor_err",
            "units",
            "valid_range",
        ]
        assert attrs["long_name"] == "500m Surface Reflectance Band 5 - first layer"
        assert attrs["units"] == "reflectance"
        assert attrs["_FillValue"] == -28672.0
        assert rds["sur_refl_b05_1"].encoding["grid_mapping"] == "spatial_ref"


def test_open_rasterio_mask_chunk_clip():
    path = os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif")
    with rasterio.open(path) as src:
        profile = src.profile

    with rioxarray.open_rasterio(
        path,
        masked=True,
        chunks=True,
        default_name="dem",
    ) as xdi:
        if isinstance(xdi, xarray.Dataset):
            xdi = xdi.dem
        assert xdi.name == "dem"
        assert str(xdi.dtype) == "float32"
        assert str(xdi.data.dtype) == "float32"
        assert str(type(xdi.data)) == "<class 'dask.array.core.Array'>"
        assert xdi.chunks == ((1,), (245,), (574,))
        assert numpy.isnan(xdi.values).sum() == 52119
        test_encoding = dict(xdi.encoding)
        assert test_encoding.pop("source").endswith("small_dem_3m_merged.tif")
        assert test_encoding == {
            "_FillValue": 0.0,
            "grid_mapping": "spatial_ref",
            "dtype": "uint16",
            "rasterio_dtype": "uint16",
            "profile": profile,
        }
        attrs = dict(xdi.attrs)
        assert_almost_equal(
            tuple(xdi.rio._cached_transform())[:6],
            (3.0, 0.0, 425047.68381405267, 0.0, -3.0, 4615780.040546387),
        )
        assert attrs == {
            "AREA_OR_POINT": "Area",
            "add_offset": 0.0,
            "scale_factor": 1.0,
        }

        # get subset for testing
        subset = xdi.isel(x=slice(150, 160), y=slice(100, 150))
        comp_subset = subset.isel(x=slice(1, None), y=slice(1, None))
        # add transform for test
        comp_subset.rio.write_transform()

        geometries = [
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [subset.x.values[0], subset.y.values[-1]],
                        [subset.x.values[0], subset.y.values[0]],
                        [subset.x.values[-1], subset.y.values[0]],
                        [subset.x.values[-1], subset.y.values[-1]],
                        [subset.x.values[0], subset.y.values[-1]],
                    ]
                ],
            }
        ]

        # test data array
        clipped = xdi.rio.clip(geometries, comp_subset.rio.crs)
        _assert_xarrays_equal(clipped, comp_subset)
        test_encoding = dict(clipped.encoding)
        assert test_encoding.pop("source").endswith("small_dem_3m_merged.tif")
        assert test_encoding == {
            "_FillValue": 0.0,
            "grid_mapping": "spatial_ref",
            "dtype": "uint16",
            "rasterio_dtype": "uint16",
            "profile": profile,
        }

        # test dataset
        clipped_ds = xdi.to_dataset(name="test_data").rio.clip(
            geometries, subset.rio.crs
        )
        comp_subset_ds = comp_subset.to_dataset(name="test_data")
        _assert_xarrays_equal(clipped_ds, comp_subset_ds)
        test_encoding = dict(clipped.encoding)
        assert test_encoding.pop("source").endswith("small_dem_3m_merged.tif")
        assert test_encoding == {
            "_FillValue": 0.0,
            "grid_mapping": "spatial_ref",
            "dtype": "uint16",
            "rasterio_dtype": "uint16",
            "profile": profile,
        }


##############################################################################
# From xarray tests
##############################################################################
ON_WINDOWS = sys.platform == "win32"
_counter = itertools.count()


@contextlib.contextmanager
def create_tmp_file(suffix=".nc", allow_cleanup_failure=False):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "temp-{}{}".format(next(_counter), suffix))
    try:
        yield path
    finally:
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            if not allow_cleanup_failure:
                raise


@contextlib.contextmanager
def create_tmp_geotiff(
    nx=4,
    ny=3,
    nz=3,
    transform=None,
    transform_args=[5000, 80000, 1000, 2000.0],
    crs="EPSG:32618",
    open_kwargs=None,
    additional_attrs=None,
):
    # yields a temporary geotiff file and a corresponding expected DataArray
    if open_kwargs is None:
        open_kwargs = {}

    with create_tmp_file(suffix=".tif", allow_cleanup_failure=ON_WINDOWS) as tmp_file:
        # allow 2d or 3d shapes
        if nz == 1:
            data_shape = ny, nx
            write_kwargs = {"indexes": 1}
        else:
            data_shape = nz, ny, nx
            write_kwargs = {}
        data = numpy.arange(nz * ny * nx, dtype=rasterio.float32).reshape(*data_shape)
        if transform is None and transform_args is not None:
            transform = from_origin(*transform_args)
        if additional_attrs is None:
            additional_attrs = {
                "descriptions": tuple(f"d{n + 1}" for n in range(nz)),
                "units": tuple(f"u{n + 1}" for n in range(nz)),
            }
        with rasterio.open(
            tmp_file,
            "w",
            driver="GTiff",
            height=ny,
            width=nx,
            count=nz,
            crs=crs,
            transform=transform,
            dtype=rasterio.float32,
            **open_kwargs,
        ) as s:
            for attr, val in additional_attrs.items():
                setattr(s, attr, val)
            for band in range(1, nz + 1):
                s.update_tags(band, BAND=band)
            s.write(data, **write_kwargs)
            dx, dy = s.res[0], -s.res[1]
            tt = s.transform
            crs_wkt = s.crs.to_wkt() if s.crs else None

        if not transform_args:
            a, b, c, d = tt.c, tt.f, -tt.e, tt.a
        else:
            a, b, c, d = transform_args
        data = data[numpy.newaxis, ...] if nz == 1 else data
        expected = DataArray(
            data,
            dims=("band", "y", "x"),
            coords={
                "band": numpy.arange(nz) + 1,
                "y": -numpy.arange(ny) * d + b + dy / 2,
                "x": numpy.arange(nx) * c + a + dx / 2,
            },
        )
        if crs_wkt is not None:
            expected.coords[DEFAULT_GRID_MAP] = xarray.Variable((), 0)
            expected.coords[DEFAULT_GRID_MAP].attrs["spatial_ref"] = crs_wkt
        yield tmp_file, expected


def test_serialization():
    with create_tmp_geotiff(additional_attrs={}) as (tmp_file, expected):
        # Write it to a netcdf and read again (roundtrip)
        with rioxarray.open_rasterio(tmp_file, mask_and_scale=True) as rioda:
            with create_tmp_file(suffix=".nc") as tmp_nc_file:
                rioda.to_netcdf(tmp_nc_file)
                with xarray.open_dataarray(tmp_nc_file, decode_coords="all") as ncds:
                    assert_identical(rioda, ncds)


def test_utm():
    with create_tmp_geotiff() as (tmp_file, expected):
        with rioxarray.open_rasterio(tmp_file) as rioda:
            assert_allclose(rioda, expected)
            assert rioda.attrs["scale_factor"] == 1.0
            assert rioda.attrs["add_offset"] == 0.0
            assert rioda.attrs["long_name"] == ("d1", "d2", "d3")
            assert rioda.attrs["units"] == ("u1", "u2", "u3")
            assert rioda.rio.crs == expected.rio.crs
            assert_array_equal(rioda.rio.resolution(), expected.rio.resolution())
            assert isinstance(rioda.rio._cached_transform(), Affine)
            assert rioda.rio.nodata is None

        # Check no parse coords
        with rioxarray.open_rasterio(tmp_file, parse_coordinates=False) as rioda:
            assert "x" not in rioda.coords
            assert "y" not in rioda.coords


@pytest.mark.parametrize("chunks", [True, None])
def test_band_as_variable(open_rasterio, chunks, tmp_path):
    test_raster = tmp_path / "test.tif"

    with create_tmp_geotiff() as (tmp_file, expected):
        with open_rasterio(
            tmp_file,
            band_as_variable=True,
            mask_and_scale=False,
            chunks=chunks,
        ) as riods:

            def _check_raster(raster_ds):
                for band in (1, 2, 3):
                    band_name = f"band_{band}"
                    assert_allclose(
                        raster_ds[band_name], expected.sel(band=band).drop("band")
                    )
                    assert raster_ds[band_name].attrs["BAND"] == band
                    assert raster_ds[band_name].attrs["scale_factor"] == 1.0
                    assert raster_ds[band_name].attrs["add_offset"] == 0.0
                    assert raster_ds[band_name].attrs["long_name"] == f"d{band}"
                    assert raster_ds[band_name].attrs["units"] == f"u{band}"
                    assert raster_ds[band_name].rio.crs == expected.rio.crs
                    assert_array_equal(
                        raster_ds[band_name].rio.resolution(), expected.rio.resolution()
                    )
                    assert isinstance(
                        raster_ds[band_name].rio._cached_transform(), Affine
                    )
                    assert raster_ds[band_name].rio.nodata is None

            _check_raster(riods)
            # test roundtrip
            riods.rio.to_raster(test_raster)
            with open_rasterio(
                test_raster,
                band_as_variable=True,
                mask_and_scale=False,
                chunks=chunks,
            ) as riods_round:
                _check_raster(riods_round)


def test_platecarree():
    with create_tmp_geotiff(
        8,
        10,
        1,
        transform_args=[1, 2, 0.5, 2.0],
        crs="EPSG:4326",
        open_kwargs={"nodata": -9765},
    ) as (tmp_file, expected):
        with rioxarray.open_rasterio(tmp_file) as rioda:
            assert_allclose(rioda, expected)
            assert rioda.attrs["scale_factor"] == 1.0
            assert rioda.attrs["add_offset"] == 0.0
            assert rioda.attrs["long_name"] == "d1"
            assert rioda.attrs["units"] == "u1"
            assert rioda.rio.crs == expected.rio.crs
            assert isinstance(rioda.rio.resolution(), tuple)
            assert isinstance(rioda.rio._cached_transform(), Affine)
            assert rioda.rio.nodata == -9765.0


def test_notransform():
    # regression test for https://github.com/pydata/xarray/issues/1686
    # Create a geotiff file
    with warnings.catch_warnings():
        # rasterio throws a NotGeoreferencedWarning here, which is
        # expected since we test rasterio's defaults in this case.
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Dataset has no geotransform set",
        )
        with create_tmp_file(suffix=".tif") as tmp_file:
            # data
            nx, ny, nz = 4, 3, 3
            data = numpy.arange(nx * ny * nz, dtype=rasterio.float32).reshape(
                nz, ny, nx
            )
            with rasterio.open(
                tmp_file,
                "w",
                driver="GTiff",
                height=ny,
                width=nx,
                count=nz,
                dtype=rasterio.float32,
            ) as s:
                s.descriptions = ("nx", "ny", "nz")
                s.units = ("cm", "m", "km")
                s.write(data)

            # Tests
            expected = DataArray(
                data,
                dims=("band", "y", "x"),
                coords={
                    "band": [1, 2, 3],
                    "y": [0.5, 1.5, 2.5],
                    "x": [0.5, 1.5, 2.5, 3.5],
                },
            )
            expected.coords[DEFAULT_GRID_MAP] = xarray.Variable((), 0)
            expected.coords[DEFAULT_GRID_MAP].attrs[
                "GeoTransform"
            ] = "0.0 1.0 0.0 0.0 0.0 1.0"

            with rioxarray.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert rioda.attrs["scale_factor"] == 1.0
                assert rioda.attrs["add_offset"] == 0.0
                assert rioda.attrs["long_name"] == ("nx", "ny", "nz")
                assert rioda.attrs["units"] == ("cm", "m", "km")
                assert isinstance(rioda.rio.resolution(), tuple)
                assert isinstance(rioda.rio._cached_transform(), Affine)


def test_indexing():
    with create_tmp_geotiff(
        8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="EPSG:4326"
    ) as (tmp_file, expected):
        with rioxarray.open_rasterio(tmp_file, cache=False) as actual:
            # tests
            # assert_allclose checks all data + coordinates
            assert_allclose(actual, expected)
            assert not actual.variable._in_memory

            # Basic indexer
            ind = {"x": slice(2, 5), "y": slice(5, 7)}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            ind = {"band": slice(1, 2), "x": slice(2, 5), "y": slice(5, 7)}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            ind = {"band": slice(1, 2), "x": slice(2, 5), "y": 0}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            # orthogonal indexer
            ind = {
                "band": numpy.array([2, 1, 0]),
                "x": numpy.array([1, 0]),
                "y": numpy.array([0, 2]),
            }
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            ind = {"band": numpy.array([2, 1, 0]), "x": numpy.array([1, 0]), "y": 0}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            ind = {"band": 0, "x": numpy.array([0, 0]), "y": numpy.array([1, 1, 1])}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            # minus-stepped slice
            ind = {"band": numpy.array([2, 1, 0]), "x": slice(-1, None, -1), "y": 0}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            ind = {"band": numpy.array([2, 1, 0]), "x": 1, "y": slice(-1, 1, -2)}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            # empty selection
            ind = {"band": numpy.array([2, 1, 0]), "x": 1, "y": slice(2, 2, 1)}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            ind = {"band": slice(0, 0), "x": 1, "y": 2}
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            # vectorized indexer
            ind = {
                "band": DataArray([2, 1, 0], dims="a"),
                "x": DataArray([1, 0, 0], dims="a"),
                "y": numpy.array([0, 2]),
            }
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            ind = {
                "band": DataArray([[2, 1, 0], [1, 0, 2]], dims=["a", "b"]),
                "x": DataArray([[1, 0, 0], [0, 1, 0]], dims=["a", "b"]),
                "y": 0,
            }
            assert_allclose(expected.isel(**ind), actual.isel(**ind))
            assert not actual.variable._in_memory

            # Selecting lists of bands is fine
            ex = expected.isel(band=[1, 2])
            ac = actual.isel(band=[1, 2])
            assert_allclose(ac, ex)
            ex = expected.isel(band=[0, 2])
            ac = actual.isel(band=[0, 2])
            assert_allclose(ac, ex)

            # Integer indexing
            ex = expected.isel(band=1)
            ac = actual.isel(band=1)
            assert_allclose(ac, ex)

            ex = expected.isel(x=1, y=2)
            ac = actual.isel(x=1, y=2)
            assert_allclose(ac, ex)

            ex = expected.isel(band=0, x=1, y=2)
            ac = actual.isel(band=0, x=1, y=2)
            assert_allclose(ac, ex)

            # Mixed
            ex = actual.isel(x=slice(2), y=slice(2))
            ac = actual.isel(x=[0, 1], y=[0, 1])
            assert_allclose(ac, ex)

            ex = expected.isel(band=0, x=1, y=slice(5, 7))
            ac = actual.isel(band=0, x=1, y=slice(5, 7))
            assert_allclose(ac, ex)

            ex = expected.isel(band=0, x=slice(2, 5), y=2)
            ac = actual.isel(band=0, x=slice(2, 5), y=2)
            assert_allclose(ac, ex)

            # One-element lists
            ex = expected.isel(band=[0], x=slice(2, 5), y=[2])
            ac = actual.isel(band=[0], x=slice(2, 5), y=[2])
            assert_allclose(ac, ex)


def test_caching():
    with create_tmp_geotiff(
        8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="EPSG:4326"
    ) as (tmp_file, expected):
        # Cache is the default
        with rioxarray.open_rasterio(tmp_file) as actual:
            # This should cache everything
            assert_allclose(actual, expected)

            # once cached, non-windowed indexing should become possible
            ac = actual.isel(x=[2, 4])
            ex = expected.isel(x=[2, 4])
            assert_allclose(ac, ex)


def test_chunks():
    with create_tmp_geotiff(
        8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="EPSG:4326"
    ) as (tmp_file, expected):
        # Chunk at open time
        with rioxarray.open_rasterio(tmp_file, chunks=(1, 2, 2)) as actual:
            assert isinstance(actual.data, dask.array.Array)
            assert "open_rasterio" in actual.data.name

            # do some arithmetic
            ac = actual.mean()
            ex = expected.mean()
            assert_allclose(ac, ex)

            ac = actual.sel(band=1).mean(dim="x")
            ex = expected.sel(band=1).mean(dim="x")
            assert_allclose(ac, ex)


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("chunks", [True, "auto"])
def test_auto_chunks_no_deprecation_warning(chunks):
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "cog.tif"), chunks=chunks
    ) as rds:
        rds.mean().compute()


def test_chunks_with_mask_and_scale():
    with create_tmp_geotiff(
        10, 10, 4, transform_args=[1, 2, 0.5, 2.0], crs="EPSG:4326"
    ) as (tmp_file, expected):
        # Chunk at open time
        with rioxarray.open_rasterio(
            tmp_file, mask_and_scale=True, chunks=(1, 2, 2)
        ) as actual:
            assert isinstance(actual.data, dask.array.Array)
            assert "open_rasterio" in actual.data.name

            # do some arithmetic
            ac = actual.mean().compute()
            ex = expected.mean()
            assert_allclose(ac, ex)


def test_pickle_rasterio():
    # regression test for https://github.com/pydata/xarray/issues/2121
    with create_tmp_geotiff() as (tmp_file, expected):
        with rioxarray.open_rasterio(tmp_file) as rioda:
            temp = pickle.dumps(rioda)
            with pickle.loads(temp) as actual:
                assert_equal(actual, rioda)


def test_ENVI_tags():
    # Create an ENVI file with some tags in the ENVI namespace
    # this test uses a custom driver, so we can't use create_tmp_geotiff
    with create_tmp_file(suffix=".dat") as tmp_file:
        # data
        nx, ny, nz = 4, 3, 3
        data = numpy.arange(nx * ny * nz, dtype=rasterio.float32).reshape(nz, ny, nx)
        transform = from_origin(5000, 80000, 1000, 2000.0)
        with rasterio.open(
            tmp_file,
            "w",
            driver="ENVI",
            height=ny,
            width=nx,
            count=nz,
            crs="EPSG:32618",
            transform=transform,
            dtype=rasterio.float32,
        ) as s:
            s.update_tags(
                ns="ENVI",
                description="{Tagged file}",
                wavelength="{123.000000, 234.234000, 345.345678}",
                fwhm="{1.000000, 0.234000, 0.000345}",
            )
            s.write(data)
            dx, dy = s.res[0], -s.res[1]
            crs_wkt = s.crs.to_wkt()

        # Tests
        coords = {
            "band": [1, 2, 3],
            "y": -numpy.arange(ny) * 2000 + 80000 + dy / 2,
            "x": numpy.arange(nx) * 1000 + 5000 + dx / 2,
            "wavelength": ("band", numpy.array([123, 234.234, 345.345678])),
            "fwhm": ("band", numpy.array([1, 0.234, 0.000345])),
        }
        expected = DataArray(data, dims=("band", "y", "x"), coords=coords)
        expected.coords[DEFAULT_GRID_MAP] = xarray.Variable((), 0)
        expected.coords[DEFAULT_GRID_MAP].attrs["crs_wkt"] = crs_wkt

        with rioxarray.open_rasterio(tmp_file) as rioda:
            assert_allclose(rioda, expected)
            assert rioda.rio.crs == crs_wkt
            assert isinstance(rioda.rio._cached_transform(), Affine)
            # from ENVI tags
            assert isinstance(rioda.attrs["description"], str)
            assert isinstance(rioda.attrs["map_info"], str)
            assert isinstance(rioda.attrs["samples"], str)


def test_no_mftime():
    # rasterio can accept "filename" arguments that are actually urls,
    # including paths to remote files.
    # In issue #1816, we found that these caused dask to break, because
    # the modification time was used to determine the dask token. This
    # tests ensure we can still chunk such files when reading with
    # rasterio.
    with create_tmp_geotiff(
        8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="EPSG:4326"
    ) as (tmp_file, expected):
        with patch("os.path.getmtime", side_effect=OSError):
            with rioxarray.open_rasterio(tmp_file, chunks=(1, 2, 2)) as actual:
                assert isinstance(actual.data, dask.array.Array)
                assert_allclose(actual, expected)


@pytest.mark.timeout(30)
@pytest.mark.xfail(
    error=rasterio.errors.RasterioIOError, reason="Network could be problematic"
)
def test_http_url():
    # more examples urls here
    # http://download.osgeo.org/geotiff/samples/
    url = (
        "https://github.com/corteva/rioxarray/blob/master/"
        "test/test_data/input/cog.tif?raw=true"
    )
    with rioxarray.open_rasterio(url) as actual:
        assert actual.shape == (1, 500, 500)
    # make sure chunking works
    with rioxarray.open_rasterio(url, chunks=(1, 256, 256)) as actual:
        assert isinstance(actual.data, dask.array.Array)


def test_rasterio_environment(tmp_path):
    log = logging.getLogger("rasterio._env")
    log.setLevel(logging.DEBUG)
    logfile = tmp_path / "file.log"
    fh = logging.FileHandler(logfile)
    log.addHandler(fh)
    with create_tmp_geotiff() as (tmp_file, expected):
        # Should fail with error since suffix not allowed
        with rasterio.Env(CPL_DEBUG=True):
            with rioxarray.open_rasterio(tmp_file) as actual:
                assert_allclose(actual.load(), expected)

    assert f"GDAL: GDALOpen({tmp_file}" in logfile.read_text()


@pytest.mark.parametrize("band_as_variable", [True, False])
def test_rasterio_vrt(band_as_variable):
    # tmp_file default crs is UTM: CRS({'init': 'epsg:32618'}
    with create_tmp_geotiff() as (tmp_file, expected):
        with rasterio.open(tmp_file) as src:
            with rasterio.vrt.WarpedVRT(src, crs="epsg:4326") as vrt:
                # Value of single pixel in center of image
                lon, lat = vrt.xy(vrt.width // 2, vrt.height // 2)
                expected_val = next(vrt.sample([(lon, lat)]))
                with rioxarray.open_rasterio(
                    vrt, band_as_variable=band_as_variable
                ) as rds:
                    if band_as_variable:
                        rds = rds.band_1

                    actual_val = rds.sel(dict(x=lon, y=lat), method="nearest").data

                    assert_array_equal(rds.rio.shape, (vrt.height, vrt.width))
                    assert rds.rio.crs == vrt.crs
                    assert_array_equal(
                        tuple(abs(val) for val in rds.rio.resolution()), vrt.res
                    )
                    assert expected_val.all() == actual_val.all()


def test_rasterio_vrt_with_transform_and_size():
    # Test open_rasterio() support of WarpedVRT with transform, width and
    # height (issue #2864)
    with create_tmp_geotiff() as (tmp_file, expected):
        with rasterio.open(tmp_file) as src:
            # Estimate the transform, width and height
            # for a change of resolution
            # tmp_file initial res is (1000,2000) (default values)
            trans, w, h = calculate_default_transform(
                src.crs, src.crs, src.width, src.height, resolution=500, *src.bounds
            )
            with rasterio.vrt.WarpedVRT(src, transform=trans, width=w, height=h) as vrt:
                with rioxarray.open_rasterio(vrt) as rds:
                    assert_array_equal(rds.rio.shape, (vrt.height, vrt.width))
                    assert rds.rio.crs == vrt.crs
                    assert_array_equal(
                        tuple(abs(val) for val in rds.rio.resolution()), vrt.res
                    )
                    assert rds.rio.transform() == vrt.transform


@pytest.mark.timeout(30)
@pytest.mark.xfail(
    error=rasterio.errors.RasterioIOError, reason="Network could be problematic"
)
def test_rasterio_vrt_network():
    url = "https://storage.googleapis.com/\
    gcp-public-data-landsat/LC08/01/047/027/\
    LC08_L1TP_047027_20130421_20170310_01_T1/\
    LC08_L1TP_047027_20130421_20170310_01_T1_B4.TIF"
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_USE_HEAD=False,
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF",
    )
    with env:
        with rasterio.open(url) as src:
            with rasterio.vrt.WarpedVRT(src, crs="epsg:4326") as vrt:
                # Value of single pixel in center of image
                lon, lat = vrt.xy(vrt.width // 2, vrt.height // 2)
                expected_val = next(vrt.sample([(lon, lat)]))
                with rioxarray.open_rasterio(vrt) as rds:
                    actual_val = rds.sel(dict(x=lon, y=lat), method="nearest").data
                    assert_array_equal(rds.rio.shape, (vrt.height, vrt.width))
                    assert rds.rio.crs == vrt.crs
                    assert_array_equal(
                        tuple(abs(val) for val in rds.rio.resolution()), vrt.res
                    )
                    assert_equal(expected_val, actual_val)


def test_rasterio_vrt_with_src_crs():
    # Test open_rasterio() support of WarpedVRT with specified src_crs
    # create geotiff with no CRS and specify it manually
    with create_tmp_geotiff(crs=None) as (tmp_file, expected):
        src_crs = rasterio.crs.CRS.from_epsg(32618)
        with rasterio.open(tmp_file) as src:
            assert src.crs is None
            with rasterio.vrt.WarpedVRT(src, src_crs=src_crs) as vrt:
                with rioxarray.open_rasterio(vrt) as rds:
                    assert rds.rio.crs == src_crs


@pytest.mark.parametrize(
    "affine_c_param",
    [
        pytest.param(
            115698.25,
            marks=pytest.mark.skipif(
                GDAL_GE_3_11,
                reason="GDAL 3.10 and earlier used METHOD=GCP_POLYNOMIAL by default",
            ),
        ),
        pytest.param(
            115698.0,
            marks=pytest.mark.skipif(
                not GDAL_GE_3_11,
                reason="GDAL 3.11+ uses METHOD=GCP_HOMOGRAPHY by default if 4 or 5 GCPs (https://github.com/OSGeo/gdal/pull/11949)",
            ),
        ),
    ],
)
def test_rasterio_vrt_gcps(tmp_path, affine_c_param):
    tiffname = tmp_path / "test.tif"
    src_gcps = [
        GroundControlPoint(row=0, col=0, x=156113, y=2818720, z=0),
        GroundControlPoint(row=0, col=800, x=338353, y=2785790, z=0),
        GroundControlPoint(row=800, col=800, x=297939, y=2618518, z=0),
        GroundControlPoint(row=800, col=0, x=115698, y=2651448, z=0),
    ]
    crs = CRS.from_epsg(32618)
    with rasterio.open(
        tiffname,
        mode="w",
        height=800,
        width=800,
        count=3,
        dtype=numpy.uint8,
        driver="GTiff",
    ) as source:
        source.gcps = (src_gcps, crs)

    with rasterio.open(tiffname) as src:
        # NOTE: Eventually src_crs will not need to be provided
        # https://github.com/mapbox/rasterio/pull/2193
        with rasterio.vrt.WarpedVRT(src, src_crs=crs) as vrt:
            with rioxarray.open_rasterio(vrt) as rds:
                assert rds.rio.height == 923
                assert rds.rio.width == 1027
                assert rds.rio.crs == crs
                assert rds.rio.transform().almost_equals(
                    Affine(
                        216.8587081056465,
                        0.0,
                        affine_c_param,
                        0.0,
                        -216.8587081056465,
                        2818720.0,
                    )
                )


def test_rasterio_vrt_warp_extras(tmp_path):
    tiffname = tmp_path / "test.tif"
    src_gcps = [
        GroundControlPoint(
            row=2015.0,
            col=0.0,
            x=100.61835695597287,
            y=-0.19173548698662005,
            z=685.0004720482975,
            id="22",
            info="",
        ),
        GroundControlPoint(
            row=8060.0,
            col=18990.0,
            x=98.84559009470779,
            y=-0.3783665839371584,
            z=-0.00012378208339214325,
            id="100",
            info="",
        ),
        GroundControlPoint(
            row=20150.0,
            col=7596.0,
            x=99.61705472917613,
            y=-1.6823719472066612,
            z=-7.35744833946228e-08,
            id="217",
            info="",
        ),
        GroundControlPoint(
            row=22165.0,
            col=22788.0,
            x=98.2441590762303,
            y=-1.5732331941915954,
            z=-4.936009645462036e-08,
            id="250",
            info="",
        ),
        GroundControlPoint(
            row=12090.0,
            col=17724.0,
            x=98.88075494473509,
            y=-0.7646707270388453,
            z=-0.0003558387979865074,
            id="141",
            info="",
        ),
    ]

    crs = CRS.from_epsg(4326)

    with rasterio.open(
        tiffname,
        mode="w",
        height=23063,
        width=25313,
        count=1,
        dtype=numpy.uint8,
        driver="GTiff",
    ) as source:
        source.gcps = (src_gcps, crs)

    with rasterio.open(tiffname) as src:
        warp_extras = {"SRC_METHOD": "GCP_TPS"}
        with rasterio.vrt.WarpedVRT(
            src,
            **warp_extras,
        ) as vrt:
            assert vrt.crs == "EPSG:4326"
            assert vrt.shape == (28286, 29338)


def test_rasterio_vrt_gcps__data_exists():
    # https://github.com/corteva/rioxarray/issues/515
    vrt_file = os.path.join(TEST_INPUT_DATA_DIR, "cint16.tif")
    with rasterio.open(vrt_file) as src:
        crs = src.gcps[1]
        # NOTE: Eventually src_crs will not need to be provided
        # https://github.com/mapbox/rasterio/pull/2193
        with rasterio.vrt.WarpedVRT(src, src_crs=crs) as vrt:
            with rioxarray.open_rasterio(vrt) as rds:
                assert rds.values.any()


@pytest.mark.parametrize("lock", [True, False])
def test_open_cog(lock):
    cog_file = os.path.join(TEST_INPUT_DATA_DIR, "cog.tif")
    with rioxarray.open_rasterio(cog_file) as rdsm:
        assert rdsm.shape == (1, 500, 500)
    with rioxarray.open_rasterio(cog_file, lock=lock, overview_level=0) as rdso:
        assert rdso.shape == (1, 250, 250)


def test_mask_and_scale(open_rasterio):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc")
    with rasterio.open(test_file) as src:
        profile = src.profile

    with open_rasterio(test_file, mask_and_scale=True) as rds:
        rds = _ensure_dataset(rds)
        assert numpy.nanmin(rds.air_temperature.values) == numpy.float32(248.7)
        assert numpy.nanmax(rds.air_temperature.values) == numpy.float32(302.1)
        test_encoding = dict(rds.air_temperature.encoding)
        _assert_tmmx_source(test_encoding.pop("source"))
        assert test_encoding == {
            "_Unsigned": "true",
            "add_offset": 220.0,
            "scale_factor": 0.1,
            "_FillValue": 32767.0,
            "grid_mapping": "crs",
            "dtype": "uint16",
            "rasterio_dtype": "uint16",
            "preferred_chunks": dict(band=1, x=1386, y=585),
            "profile": profile,
        }
        attrs = rds.air_temperature.attrs
        assert "_Unsigned" not in attrs
        assert "add_offset" not in attrs
        assert "scale_factor" not in attrs
        assert "_FillValue" not in attrs


def test_no_mask_and_scale(open_rasterio):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc")
    with rasterio.open(test_file) as src:
        profile = src.profile

    with open_rasterio(
        test_file,
        mask_and_scale=False,
        masked=True,
    ) as rds:
        rds = _ensure_dataset(rds)
        assert numpy.nanmin(rds.air_temperature.values) == 287
        assert numpy.nanmax(rds.air_temperature.values) == 821
        test_encoding = dict(rds.air_temperature.encoding)
        source = test_encoding.pop("source")
        _assert_tmmx_source(source)
        assert test_encoding == {
            "_FillValue": 32767.0,
            "grid_mapping": "crs",
            "dtype": "uint16",
            "rasterio_dtype": "uint16",
            "preferred_chunks": {"band": 1, "x": 1386, "y": 585},
            "profile": profile,
        }
        attrs = rds.air_temperature.attrs
        assert attrs["_Unsigned"] == "true"
        assert attrs["add_offset"] == 220.0
        assert attrs["scale_factor"] == 0.1
        assert "_FillValue" not in attrs


def test_mask_and_scale__select_band_values(open_rasterio, tmp_path):
    # https://github.com/corteva/rioxarray/issues/580
    output_nc = tmp_path / "test.nc"
    data = numpy.hypot(
        *numpy.meshgrid(numpy.linspace(-100, 500, 50), numpy.linspace(-150, 700, 60))
    )
    xds = xarray.Dataset(
        {"var": (["y", "x"], data)},
        coords={"x": numpy.linspace(40, 51, 50), "y": numpy.linspace(55, 62, 60)},
    )
    xds.rio.write_crs(4326, inplace=True)
    xds.rio.write_coordinate_system(inplace=True)
    encoding = {"var": {"scale_factor": 0.01, "_FillValue": -9999, "dtype": "int32"}}
    xds.to_netcdf(output_nc, encoding=encoding)
    with open_rasterio(output_nc) as rds:
        if isinstance(rds, xarray.Dataset):
            rds = rds["var"]
        assert_array_equal(rds.values[0], rds.isel(band=0).values)


def test_mask_and_scale__to_raster(open_rasterio, tmp_path):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc")
    tmp_output = tmp_path / "tmmx_20190121.tif"
    with open_rasterio(test_file, mask_and_scale=True) as rds:
        _ensure_dataset(rds).air_temperature.rio.to_raster(str(tmp_output))
        with rasterio.open(str(tmp_output)) as riofh:
            assert riofh.scales == (0.1,)
            assert riofh.offsets == (220.0,)
            assert riofh.nodata == 32767.0
            data = riofh.read(1, masked=True)
            assert data.min() == 287
            assert data.max() == 821


def test_mask_and_scale__unicode(open_rasterio):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "unicode.nc")
    with open_rasterio(test_file, mask_and_scale=True) as rds:
        rds = _ensure_dataset(rds)
        assert numpy.nanmin(rds.LST.values) == numpy.float32(270.4925)
        assert numpy.nanmax(rds.LST.values) == numpy.float32(276.6025)
        test_encoding = dict(rds.LST.encoding)
        assert test_encoding["_Unsigned"] == "true"
        assert test_encoding["add_offset"] == 190
        assert test_encoding["scale_factor"] == pytest.approx(0.0025)
        assert test_encoding["_FillValue"] == 65535
        expected_dtype = "int16"
        if GDAL_GE_36:
            # https://github.com/OSGeo/gdal/pull/6369
            expected_dtype = "uint16"
        assert test_encoding["dtype"] == expected_dtype
        assert test_encoding["rasterio_dtype"] == expected_dtype


def test_mask_and_scale__unicode__to_raster(open_rasterio, tmp_path):
    tmp_output = tmp_path / "unicode.tif"
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "unicode.nc")
    with open_rasterio(test_file, mask_and_scale=True) as rds:
        _ensure_dataset(rds).LST.rio.to_raster(str(tmp_output))
        with rasterio.open(str(tmp_output)) as riofh:
            assert riofh.scales == (pytest.approx(0.0025),)
            assert riofh.offsets == (190,)
            assert riofh.nodata == 65535
            data = riofh.read(1, masked=True)
            assert data.min() == 32197
            assert data.max() == 34641
            assert riofh.dtypes == ("uint16",)


def test_notgeoreferenced_warning(open_rasterio):
    with create_tmp_geotiff(transform_args=None) as (tmp_file, expected):
        with pytest.warns(NotGeoreferencedWarning):
            open_rasterio(tmp_file)


@pytest.mark.xfail(
    version.parse(rasterio.__gdal_version__) < version.parse("3.0.4"),
    reason="This was fixed in GDAL 3.0.4",
)
def test_nc_attr_loading(open_rasterio):
    with open_rasterio(os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc")) as rds:
        assert rds.sizes == {"y": 10, "x": 10, "time": 2}
        assert rds.attrs == {"coordinates": "spatial_ref"}
        assert rds.y.attrs["units"] == "metre"
        assert rds.x.attrs["units"] == "metre"
        assert rds.time.encoding == {
            "units": "seconds since 2016-12-19T10:27:29.687763",
            "calendar": "proleptic_gregorian",
        }
        assert str(rds.time.values[0]) == "2016-12-19 10:27:29.687763"
        assert str(rds.time.values[1]) == "2016-12-29 12:52:42.347451"


def test_nc_attr_loading__disable_decode_times(open_rasterio):
    with open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"), decode_times=False
    ) as rds:
        assert rds.sizes == {"y": 10, "x": 10, "time": 2}
        assert rds.attrs == {"coordinates": "spatial_ref"}
        assert rds.y.attrs["units"] == "metre"
        assert rds.x.attrs["units"] == "metre"
        assert rds.time.encoding == {}
        assert numpy.isnan(rds.time.attrs.pop("_FillValue"))
        assert rds.time.attrs == {
            "units": "seconds since 2016-12-19T10:27:29.687763",
            "calendar": "proleptic_gregorian",
        }
        assert_array_equal(rds.time.values, [0, 872712.659688])


def test_lockless():
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "cog.tif"), lock=False, chunks=True
    ) as rds:
        rds.mean().compute()


def test_lock_true():
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "cog.tif"), lock=True, chunks=True
    ) as rds:
        rds.mean().compute()


def test_non_rectilinear():
    # Create a geotiff file with 2d coordinates
    with create_tmp_geotiff(
        transform=from_origin(0, 3, 1, 1).rotation(45), crs=None
    ) as (tmp_file, _):
        with rasterio.open(tmp_file) as rds:
            with rioxarray.open_rasterio(tmp_file) as rioda:
                for xi, yi in itertools.product(range(rds.width), range(rds.height)):
                    subset = rioda.isel(x=xi, y=yi)
                    assert_almost_equal(
                        rds.xy(yi, xi), (subset.xc.item(), subset.yc.item())
                    )
                assert "x" not in rioda.coords
                assert "y" not in rioda.coords
                assert "xc" in rioda.coords
                assert "yc" in rioda.coords
                assert rioda.rio.crs is None
                assert rioda.attrs["scale_factor"] == 1.0
                assert rioda.attrs["add_offset"] == 0.0
                assert rioda.attrs["long_name"] == ("d1", "d2", "d3")
                assert rioda.attrs["units"] == ("u1", "u2", "u3")
                assert isinstance(rioda.rio.resolution(), tuple)
                assert isinstance(rioda.rio._cached_transform(), Affine)


def test_non_rectilinear__load_coords(open_rasterio):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "2d_test.tif")
    with open_rasterio(test_file) as xds:
        assert xds.rio.shape == (10, 10)
        with rasterio.open(test_file) as rds:
            assert rds.transform == xds.rio.transform()
            for xi, yi in itertools.product(range(rds.width), range(rds.height)):
                subset = xds.isel(x=xi, y=yi)
                assert_almost_equal(
                    rds.xy(yi, xi), (subset.xc.item(), subset.yc.item())
                )


def test_non_rectilinear__skip_parse_coordinates(open_rasterio):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "2d_test.tif")
    with open_rasterio(test_file, parse_coordinates=False) as xds:
        assert "xc" not in xds.coords
        assert "yc" not in xds.coords
        assert xds.rio.shape == (10, 10)
        with rasterio.open(test_file) as rds:
            assert rds.transform == xds.rio.transform()


def test_rotation_affine():
    with create_tmp_geotiff(
        transform=Affine(0.0, -10.0, 1743817.4113815, -10.0, -0.0, 5435582.5113815),
        crs=None,
    ) as (tmp_file, _):
        with rasterio.open(tmp_file) as rds:
            with rioxarray.open_rasterio(tmp_file) as rioda:
                for xi, yi in itertools.product(range(rds.width), range(rds.height)):
                    subset = rioda.isel(x=xi, y=yi)
                    assert_almost_equal(
                        rds.xy(yi, xi), (subset.xc.item(), subset.yc.item())
                    )
            assert rioda.rio.transform() == rds.transform
            with pytest.warns(
                UserWarning,
                match=r"Transform that is non\-rectilinear or with rotation found",
            ):
                assert rioda.rio.transform(recalc=True) == rds.transform
            assert rioda.rio.resolution() == (10, 10)


@pytest.mark.parametrize("dtype", [None, "complex_int16"])
def test_cint16_dtype(dtype, tmp_path):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "cint16.tif")
    with rioxarray.open_rasterio(test_file) as xds:
        assert xds.rio.shape == (100, 100)
        assert xds.dtype == "complex64"
        assert xds.encoding["rasterio_dtype"] == "complex_int16"

        tmp_output = tmp_path / "tmp_cint16.tif"
        with pytest.warns(NotGeoreferencedWarning):
            xds.rio.to_raster(str(tmp_output), dtype=dtype)
    with rasterio.open(str(tmp_output)) as riofh:
        data = riofh.read()
        assert "complex_int16" in riofh.dtypes
        assert data.dtype == "complex64"


def test_cint16_dtype_nodata(tmp_path):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "cint16.tif")
    with rioxarray.open_rasterio(test_file) as xds:
        assert xds.rio.nodata == 0

        tmp_output = tmp_path / "tmp_cint16.tif"
        with pytest.warns(NotGeoreferencedWarning):
            xds.rio.to_raster(str(tmp_output), dtype="complex_int16")
        with rasterio.open(str(tmp_output)) as riofh:
            assert riofh.nodata == 0

        # Assign nodata=None
        tmp_output = tmp_path / "tmp_cint16_nodata.tif"
        xds.rio.write_nodata(None, inplace=True)
        with pytest.warns(NotGeoreferencedWarning):
            xds.rio.to_raster(str(tmp_output), dtype="complex_int16")
    with rasterio.open(str(tmp_output)) as riofh:
        assert riofh.nodata is None


@pytest.mark.parametrize("dtype", [None, "complex_int16"])
def test_cint16_dtype_masked(dtype, tmp_path):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "cint16.tif")
    with rioxarray.open_rasterio(test_file, masked=True) as xds:
        assert xds.rio.shape == (100, 100)
        assert xds.dtype == "complex64"
        assert xds.rio.encoded_nodata == 0
        assert numpy.isnan(xds.rio.nodata)

        tmp_output = tmp_path / "tmp_cint16.tif"
        with pytest.warns(NotGeoreferencedWarning):
            xds.rio.to_raster(str(tmp_output), dtype=dtype)
    with rasterio.open(str(tmp_output)) as riofh:
        data = riofh.read()
        assert "complex_int16" in riofh.dtypes
        assert riofh.nodata == 0
        assert data.dtype == "complex64"


def test_cint16_promote_dtype(tmp_path):
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "cint16.tif")
    with rioxarray.open_rasterio(test_file) as xds:
        tmp_output = tmp_path / "tmp_cfloat64.tif"
        with pytest.warns(NotGeoreferencedWarning):
            xds.rio.to_raster(str(tmp_output), dtype="complex64")
        with rasterio.open(str(tmp_output)) as riofh:
            data = riofh.read()
            assert "complex64" in riofh.dtypes
            assert riofh.nodata == 0
            assert data.dtype == "complex64"

        tmp_output = tmp_path / "tmp_cfloat128.tif"
        with pytest.warns(NotGeoreferencedWarning):
            xds.rio.to_raster(str(tmp_output), dtype="complex128")
    with rasterio.open(str(tmp_output)) as riofh:
        data = riofh.read()
        assert "complex128" in riofh.dtypes
        assert riofh.nodata == 0
        assert data.dtype == "complex128"


def test_reading_gcps(tmp_path):
    """
    Test reading gcps from a tiff file.
    """
    tiffname = tmp_path / "test.tif"

    gdal_gcps = _create_gdal_gcps()

    with rasterio.open(
        tiffname,
        mode="w",
        height=800,
        width=800,
        count=3,
        dtype=numpy.uint8,
        driver="GTiff",
    ) as source:
        source.gcps = gdal_gcps

    with rioxarray.open_rasterio(tiffname) as darr:
        _check_rio_gcps(darr, *gdal_gcps)


def test_writing_gcps(tmp_path):
    """
    Test writing gcps to a tiff file.
    """
    tiffname = tmp_path / "test.tif"
    tiffname2 = tmp_path / "test_written.tif"

    gdal_gcps = _create_gdal_gcps()

    with rasterio.open(
        tiffname,
        mode="w",
        height=800,
        width=800,
        count=3,
        dtype=numpy.uint8,
        driver="GTiff",
    ) as source:
        source.gcps = gdal_gcps

    with rioxarray.open_rasterio(tiffname) as darr:
        darr.rio.to_raster(tiffname2, driver="GTIFF")

    with rioxarray.open_rasterio(tiffname2) as darr:
        assert "gcps" in darr.coords["spatial_ref"].attrs
        _check_rio_gcps(darr, *gdal_gcps)


def test_writing_gcps__to_netcdf(tmp_path):
    """
    Test writing gcps to a netCDF file.
    """
    tiffname = tmp_path / "test.tif"
    nc_name = tmp_path / "test_written.nc"

    src_gcps, crs = _create_gdal_gcps()

    with rasterio.open(
        tiffname,
        mode="w",
        height=800,
        width=800,
        count=3,
        dtype=numpy.uint8,
        driver="GTiff",
    ) as source:
        source.gcps = (src_gcps, crs)

    with rioxarray.open_rasterio(tiffname) as darr:
        darr.to_netcdf(nc_name)

    with xarray.open_dataset(nc_name, decode_coords="all") as darr:
        assert "gcps" in darr.coords["spatial_ref"].attrs
        _check_rio_gcps(darr, src_gcps=src_gcps, crs=crs)


def test_read_file_handle_with_dask():
    with open(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif"), "rb"
    ) as src:
        with rioxarray.open_rasterio(src, chunks=2048):
            pass


def test_read_cint16_with_dask():
    test_file = os.path.join(TEST_INPUT_DATA_DIR, "cint16.tif")
    with rioxarray.open_rasterio(test_file, chunks=True):
        pass


def test_read_ascii__from_bytesio():
    ascii_raster_string = """ncols        5
    nrows        5
    xllcorner    440720.000000000000
    yllcorner    3750120.000000000000
    cellsize     60.000000000000
    nodata_value -99999
        107    123    132    115    132
        115    132    107    123    148
        115    132    140    132    123
        148    132    123    123    115
        132    156    132    140    132
    """
    ascii_raster_io = io.BytesIO(ascii_raster_string.encode("utf-8"))
    with rioxarray.open_rasterio(ascii_raster_io) as rds:
        assert rds.rio.bounds() == (440720.0, 3750120.0, 441020.0, 3750420.0)


def test_reading_writing_rpcs(tmp_path):
    """Read and Write RPCs"""
    out = tmp_path / "out.tif"
    in_rpc_ext_file = os.path.join(TEST_INPUT_DATA_DIR, "test_rpcs_ext_file.tif")

    with rasterio.open(in_rpc_ext_file) as src:
        assert (
            src.tags(ns="RPC") is not None
        ), "Existing RPCs in src raster (through tag check)"
        assert (
            src.rpcs is not None
        ), "Existing RPCs in src raster (through rpc attribute)"

        # Check rioxarray consistency vs rasterio
        in_rpc_file_da = rioxarray.open_rasterio(in_rpc_ext_file)
        in_rpc_file_da_rpcs = in_rpc_file_da.rio.get_rpcs()
        assert in_rpc_file_da_rpcs is not None, "Rioxarray RPCs are not existing"
        assert (
            in_rpc_file_da_rpcs == src.rpcs
        ), "Rioxarray RPCs are not similar to rasterio's"

        # Write on disk the opened raster
        in_rpc_file_da.rio.to_raster(out)

        # Read the output and check its integrity
        dst = rioxarray.open_rasterio(out)
        dst_rpcs = dst.rio.get_rpcs()

        # Assert RPCs are saved and can be loaded
        assert dst_rpcs is not None, "Rioxarray dst RPCs are not existing"

        # WARNING: For an unknown reason, RPC written in GeoTiff are rounded!
        # This behavior is not controlled by rioxarray so don't fail the whole test if this particular check fails
        try:
            assert (
                dst_rpcs == in_rpc_file_da_rpcs
            ), "Rioxarray dst RPCs are not similar to src's"
        except AssertionError as ex:
            print(f"For an unknown reason, RPC written in GeoTiff are rounded!\n\n{ex}")
            # Expected: RPC(height_off=1046.8700801726143, height_scale=962.0000000000239, lat_off=39.81995259980312, lat_scale=0.076441044582353, line_den_coeff=[1.0, 0.0001288248348598096, ...
            # Actual:   RPC(height_off=1046.87008017261,   height_scale=962.000000000024,  lat_off=39.8199525998031,  lat_scale=0.076441044582353, line_den_coeff=[1.0, 0.00012882483485981,   ...

    # Check that rasterio can also read the RPCs
    with rasterio.open(out) as dst:
        assert (
            dst.tags(ns="RPC") is not None
        ), "Existing RPCs in dst raster (through tag check)"
        assert (
            dst.rpcs is not None
        ), "Existing RPCs in dst raster (through rpc attribute)"
