import contextlib
import itertools
import os
import pickle
import shutil
import sys
import tempfile
import warnings

import dask.array as da
import mock
import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine
from numpy.testing import assert_almost_equal
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform
from xarray import DataArray
from xarray.coding.variables import SerializationWarning
from xarray.testing import assert_allclose, assert_equal, assert_identical

import rioxarray
from rioxarray._io import build_subdataset_filter
from test.conftest import (
    TEST_COMPARE_DATA_DIR,
    TEST_INPUT_DATA_DIR,
    _assert_xarrays_equal,
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


def test_open_variable_filter():
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"), variable=["blue"]
    ) as rds:
        assert list(rds.data_vars) == ["blue"]


def test_open_group_filter__missing():
    with rioxarray.open_rasterio(
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
    for rds in rds_list:
        assert rds.attrs["SHORTNAME"] == "MOD09GA"
    assert rds_list[0].dims == {"y": 1200, "x": 1200, "band": 1}
    assert rds_list[1].dims == {"y": 2400, "x": 2400, "band": 1}


def test_open_group_filter():
    with rioxarray.open_rasterio(
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


def test_open_group_load_attrs():
    with rioxarray.open_rasterio(
        os.path.join(
            TEST_INPUT_DATA_DIR, "MOD09GA.A2008296.h14v17.006.2015181011753.hdf"
        ),
        variable="sur_refl_b05_1",
    ) as rds:
        attrs = rds["sur_refl_b05_1"].attrs
        assert sorted(attrs) == [
            "_FillValue",
            "add_offset",
            "grid_mapping",
            "long_name",
            "scale_factor",
            "transform",
            "units",
        ]
        assert attrs["long_name"] == "500m Surface Reflectance Band 5 - first layer"
        assert attrs["units"] == "reflectance"
        assert attrs["_FillValue"] == -28672.0
        assert attrs["grid_mapping"] == "spatial_ref"


def test_open_rasterio_mask_chunk_clip():
    with rioxarray.open_rasterio(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif"),
        masked=True,
        chunks=True,
        default_name="dem",
    ) as xdi:
        assert xdi.name == "dem"
        assert str(xdi.dtype) == "float64"
        assert str(xdi.data.dtype) == "float64"
        assert str(type(xdi.data)) == "<class 'dask.array.core.Array'>"
        assert xdi.chunks == ((1,), (245,), (574,))
        assert np.isnan(xdi.values).sum() == 52119
        assert xdi.encoding == {"_FillValue": 0.0}
        attrs = dict(xdi.attrs)
        assert_almost_equal(
            attrs.pop("transform"),
            (3.0, 0.0, 425047.68381405267, 0.0, -3.0, 4615780.040546387),
        )
        assert attrs == {
            "grid_mapping": "spatial_ref",
            "add_offset": 0.0,
            "scale_factor": 1.0,
        }

        # get subset for testing
        subset = xdi.isel(x=slice(150, 160), y=slice(100, 150))
        comp_subset = subset.isel(x=slice(1, None), y=slice(1, None))
        # add transform for test
        comp_subset.attrs["transform"] = tuple(comp_subset.rio.transform(recalc=True))

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
        assert clipped.encoding == {"_FillValue": 0.0}

        # test dataset
        clipped_ds = xdi.to_dataset(name="test_data").rio.clip(
            geometries, subset.rio.crs
        )
        comp_subset_ds = comp_subset.to_dataset(name="test_data")
        _assert_xarrays_equal(clipped_ds, comp_subset_ds)
        assert clipped_ds.test_data.encoding == {"_FillValue": 0.0}


##############################################################################
# From xarray tests
##############################################################################
ON_WINDOWS = sys.platform == "win32"
_counter = itertools.count()


@contextlib.contextmanager
def create_tmp_file(suffix=".nc", allow_cleanup_failure=False):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "temp-%s%s" % (next(_counter), suffix))
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
    crs={"units": "m", "no_defs": True, "ellps": "WGS84", "proj": "utm", "zone": 18},
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
        data = np.arange(nz * ny * nx, dtype=rasterio.float32).reshape(*data_shape)
        if transform is None and transform_args is not None:
            transform = from_origin(*transform_args)
        if additional_attrs is None:
            additional_attrs = {
                "descriptions": tuple("d{}".format(n + 1) for n in range(nz)),
                "units": tuple("u{}".format(n + 1) for n in range(nz)),
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
            s.write(data, **write_kwargs)
            dx, dy = s.res[0], -s.res[1]
            tt = s.transform

        if not transform_args:
            a, b, c, d = tt.c, tt.f, -tt.e, tt.a
        else:
            a, b, c, d = transform_args
        data = data[np.newaxis, ...] if nz == 1 else data
        expected = DataArray(
            data,
            dims=("band", "y", "x"),
            coords={
                "band": np.arange(nz) + 1,
                "y": -np.arange(ny) * d + b + dy / 2,
                "x": np.arange(nx) * c + a + dx / 2,
            },
        )
        yield tmp_file, expected


class TestRasterio:
    def test_serialization(self):
        with create_tmp_geotiff(additional_attrs={}) as (tmp_file, expected):
            # Write it to a netcdf and read again (roundtrip)
            with xr.open_rasterio(tmp_file) as rioda:
                with create_tmp_file(suffix=".nc") as tmp_nc_file:
                    rioda.to_netcdf(tmp_nc_file)
                    with xr.open_dataarray(tmp_nc_file) as ncds:
                        assert_identical(rioda, ncds)

    def test_utm(self):
        with create_tmp_geotiff() as (tmp_file, expected):
            with xr.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert rioda.attrs["scales"] == (1.0, 1.0, 1.0)
                assert rioda.attrs["offsets"] == (0.0, 0.0, 0.0)
                assert rioda.attrs["descriptions"] == ("d1", "d2", "d3")
                assert rioda.attrs["units"] == ("u1", "u2", "u3")
                assert isinstance(rioda.attrs["crs"], str)
                assert isinstance(rioda.attrs["res"], tuple)
                assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                assert isinstance(rioda.attrs["transform"], tuple)
                assert len(rioda.attrs["transform"]) == 6
                np.testing.assert_array_equal(
                    rioda.attrs["nodatavals"], [np.NaN, np.NaN, np.NaN]
                )

            # Check no parse coords
            with xr.open_rasterio(tmp_file, parse_coordinates=False) as rioda:
                assert "x" not in rioda.coords
                assert "y" not in rioda.coords

    def test_non_rectilinear(self):
        # Create a geotiff file with 2d coordinates
        with create_tmp_geotiff(
            transform=from_origin(0, 3, 1, 1).rotation(45), crs=None
        ) as (tmp_file, _):
            # Default is to not parse coords
            with xr.open_rasterio(tmp_file) as rioda:
                assert "x" not in rioda.coords
                assert "y" not in rioda.coords
                assert "crs" not in rioda.attrs
                assert rioda.attrs["scales"] == (1.0, 1.0, 1.0)
                assert rioda.attrs["offsets"] == (0.0, 0.0, 0.0)
                assert rioda.attrs["descriptions"] == ("d1", "d2", "d3")
                assert rioda.attrs["units"] == ("u1", "u2", "u3")
                assert isinstance(rioda.attrs["res"], tuple)
                assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                assert isinstance(rioda.attrs["transform"], tuple)
                assert len(rioda.attrs["transform"]) == 6

            # See if a warning is raised if we force it
            with pytest.warns(Warning, match="transformation isn't rectilinear"):
                with xr.open_rasterio(tmp_file, parse_coordinates=True) as rioda:
                    assert "x" not in rioda.coords
                    assert "y" not in rioda.coords

    def test_platecarree(self):
        with create_tmp_geotiff(
            8,
            10,
            1,
            transform_args=[1, 2, 0.5, 2.0],
            crs="+proj=latlong",
            open_kwargs={"nodata": -9765},
        ) as (tmp_file, expected):
            with xr.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert rioda.attrs["scales"] == (1.0,)
                assert rioda.attrs["offsets"] == (0.0,)
                assert isinstance(rioda.attrs["descriptions"], tuple)
                assert isinstance(rioda.attrs["units"], tuple)
                assert isinstance(rioda.attrs["crs"], str)
                assert isinstance(rioda.attrs["res"], tuple)
                assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                assert isinstance(rioda.attrs["transform"], tuple)
                assert len(rioda.attrs["transform"]) == 6
                np.testing.assert_array_equal(rioda.attrs["nodatavals"], [-9765.0])

    def test_notransform(self):
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
                data = np.arange(nx * ny * nz, dtype=rasterio.float32).reshape(
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
                with xr.open_rasterio(tmp_file) as rioda:
                    assert_allclose(rioda, expected)
                    assert rioda.attrs["scales"] == (1.0, 1.0, 1.0)
                    assert rioda.attrs["offsets"] == (0.0, 0.0, 0.0)
                    assert rioda.attrs["descriptions"] == ("nx", "ny", "nz")
                    assert rioda.attrs["units"] == ("cm", "m", "km")
                    assert isinstance(rioda.attrs["res"], tuple)
                    assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                    assert isinstance(rioda.attrs["transform"], tuple)
                    assert len(rioda.attrs["transform"]) == 6

    def test_indexing(self):
        with create_tmp_geotiff(
            8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="+proj=latlong"
        ) as (tmp_file, expected):
            with xr.open_rasterio(tmp_file, cache=False) as actual:

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
                    "band": np.array([2, 1, 0]),
                    "x": np.array([1, 0]),
                    "y": np.array([0, 2]),
                }
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": np.array([2, 1, 0]), "x": np.array([1, 0]), "y": 0}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": 0, "x": np.array([0, 0]), "y": np.array([1, 1, 1])}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                # minus-stepped slice
                ind = {"band": np.array([2, 1, 0]), "x": slice(-1, None, -1), "y": 0}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": np.array([2, 1, 0]), "x": 1, "y": slice(-1, 1, -2)}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                # empty selection
                ind = {"band": np.array([2, 1, 0]), "x": 1, "y": slice(2, 2, 1)}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": slice(0, 0), "x": 1, "y": 2}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                # vectorized indexer
                ind = {
                    "band": DataArray([2, 1, 0], dims="a"),
                    "x": DataArray([1, 0, 0], dims="a"),
                    "y": np.array([0, 2]),
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

    def test_caching(self):
        with create_tmp_geotiff(
            8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="+proj=latlong"
        ) as (tmp_file, expected):
            # Cache is the default
            with xr.open_rasterio(tmp_file) as actual:

                # This should cache everything
                assert_allclose(actual, expected)

                # once cached, non-windowed indexing should become possible
                ac = actual.isel(x=[2, 4])
                ex = expected.isel(x=[2, 4])
                assert_allclose(ac, ex)

    def test_chunks(self):
        with create_tmp_geotiff(
            8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="+proj=latlong"
        ) as (tmp_file, expected):
            # Chunk at open time
            with xr.open_rasterio(tmp_file, chunks=(1, 2, 2)) as actual:
                assert isinstance(actual.data, da.Array)
                assert "open_rasterio" in actual.data.name

                # do some arithmetic
                ac = actual.mean()
                ex = expected.mean()
                assert_allclose(ac, ex)

                ac = actual.sel(band=1).mean(dim="x")
                ex = expected.sel(band=1).mean(dim="x")
                assert_allclose(ac, ex)

    def test_pickle_rasterio(self):
        # regression test for https://github.com/pydata/xarray/issues/2121
        with create_tmp_geotiff() as (tmp_file, expected):
            with xr.open_rasterio(tmp_file) as rioda:
                temp = pickle.dumps(rioda)
                with pickle.loads(temp) as actual:
                    assert_equal(actual, rioda)

    def test_ENVI_tags(self):
        # Create an ENVI file with some tags in the ENVI namespace
        # this test uses a custom driver, so we can't use create_tmp_geotiff
        with create_tmp_file(suffix=".dat") as tmp_file:
            # data
            nx, ny, nz = 4, 3, 3
            data = np.arange(nx * ny * nz, dtype=rasterio.float32).reshape(nz, ny, nx)
            transform = from_origin(5000, 80000, 1000, 2000.0)
            with rasterio.open(
                tmp_file,
                "w",
                driver="ENVI",
                height=ny,
                width=nx,
                count=nz,
                crs={
                    "units": "m",
                    "no_defs": True,
                    "ellps": "WGS84",
                    "proj": "utm",
                    "zone": 18,
                },
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

            # Tests
            coords = {
                "band": [1, 2, 3],
                "y": -np.arange(ny) * 2000 + 80000 + dy / 2,
                "x": np.arange(nx) * 1000 + 5000 + dx / 2,
                "wavelength": ("band", np.array([123, 234.234, 345.345678])),
                "fwhm": ("band", np.array([1, 0.234, 0.000345])),
            }
            expected = DataArray(data, dims=("band", "y", "x"), coords=coords)

            with xr.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert isinstance(rioda.attrs["crs"], str)
                assert isinstance(rioda.attrs["res"], tuple)
                assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                assert isinstance(rioda.attrs["transform"], tuple)
                assert len(rioda.attrs["transform"]) == 6
                # from ENVI tags
                assert isinstance(rioda.attrs["description"], str)
                assert isinstance(rioda.attrs["map_info"], str)
                assert isinstance(rioda.attrs["samples"], str)

    def test_no_mftime(self):
        # rasterio can accept "filename" urguments that are actually urls,
        # including paths to remote files.
        # In issue #1816, we found that these caused dask to break, because
        # the modification time was used to determine the dask token. This
        # tests ensure we can still chunk such files when reading with
        # rasterio.
        with create_tmp_geotiff(
            8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="+proj=latlong"
        ) as (tmp_file, expected):
            with mock.patch("os.path.getmtime", side_effect=OSError):
                with xr.open_rasterio(tmp_file, chunks=(1, 2, 2)) as actual:
                    assert isinstance(actual.data, da.Array)
                    assert_allclose(actual, expected)

    @pytest.mark.xfail(reason="Network could be problematic")
    def test_http_url(self):
        # more examples urls here
        # http://download.osgeo.org/geotiff/samples/
        url = "http://download.osgeo.org/geotiff/samples/made_up/ntf_nord.tif"
        with xr.open_rasterio(url) as actual:
            assert actual.shape == (1, 512, 512)
        # make sure chunking works
        with xr.open_rasterio(url, chunks=(1, 256, 256)) as actual:
            assert isinstance(actual.data, da.Array)

    def test_rasterio_environment(self):
        with create_tmp_geotiff() as (tmp_file, expected):
            # Should fail with error since suffix not allowed
            with pytest.raises(Exception):
                with rasterio.Env(GDAL_SKIP="GTiff"):
                    with xr.open_rasterio(tmp_file) as actual:
                        assert_allclose(actual, expected)

    @pytest.mark.xfail(
        rasterio.__version__ == "1.1.1",
        reason="https://github.com/mapbox/rasterio/issues/1833",
    )
    def test_rasterio_vrt(self):
        # tmp_file default crs is UTM: CRS({'init': 'epsg:32618'}
        with create_tmp_geotiff() as (tmp_file, expected):
            with rasterio.open(tmp_file) as src:
                with rasterio.vrt.WarpedVRT(src, crs="epsg:4326") as vrt:
                    expected_shape = (vrt.width, vrt.height)
                    expected_crs = vrt.crs
                    expected_res = vrt.res
                    # Value of single pixel in center of image
                    lon, lat = vrt.xy(vrt.width // 2, vrt.height // 2)
                    expected_val = next(vrt.sample([(lon, lat)]))
                    with xr.open_rasterio(vrt) as da:
                        actual_shape = (da.sizes["x"], da.sizes["y"])
                        actual_crs = da.crs
                        actual_res = da.res
                        actual_val = da.sel(dict(x=lon, y=lat), method="nearest").data

                        assert actual_crs == expected_crs
                        assert actual_res == expected_res
                        assert actual_shape == expected_shape
                        assert expected_val.all() == actual_val.all()

    def test_rasterio_vrt_with_transform_and_size(self):
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
                with rasterio.vrt.WarpedVRT(
                    src, transform=trans, width=w, height=h
                ) as vrt:
                    expected_shape = (vrt.width, vrt.height)
                    expected_res = vrt.res
                    expected_transform = vrt.transform
                    with xr.open_rasterio(vrt) as da:
                        actual_shape = (da.sizes["x"], da.sizes["y"])
                        actual_res = da.res
                        actual_transform = Affine(*da.transform)
                        assert actual_res == expected_res
                        assert actual_shape == expected_shape
                        assert actual_transform == expected_transform

    @pytest.mark.xfail(reason="Network could be problematic")
    def test_rasterio_vrt_network(self):
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
                    expected_shape = (vrt.width, vrt.height)
                    expected_crs = vrt.crs
                    expected_res = vrt.res
                    # Value of single pixel in center of image
                    lon, lat = vrt.xy(vrt.width // 2, vrt.height // 2)
                    expected_val = next(vrt.sample([(lon, lat)]))
                    with xr.open_rasterio(vrt) as da:
                        actual_shape = (da.sizes["x"], da.sizes["y"])
                        actual_crs = da.crs
                        actual_res = da.res
                        actual_val = da.sel(dict(x=lon, y=lat), method="nearest").data

                        assert_equal(actual_shape, expected_shape)
                        assert_equal(actual_crs, expected_crs)
                        assert_equal(actual_res, expected_res)
                        assert_equal(expected_val, actual_val)


def test_open_cog():
    cog_file = os.path.join(TEST_INPUT_DATA_DIR, "cog.tif")
    rdsm = rioxarray.open_rasterio(cog_file)
    assert rdsm.shape == (1, 500, 500)
    rdso = rioxarray.open_rasterio(cog_file, overview_level=0)
    assert rdso.shape == (1, 250, 250)


def test_mask_and_scale():
    with pytest.warns(SerializationWarning):
        with rioxarray.open_rasterio(
            os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc"), mask_and_scale=True
        ) as rds:
            assert np.nanmin(rds.air_temperature.values) == 248.7
            assert np.nanmax(rds.air_temperature.values) == 302.1
            assert rds.air_temperature.encoding == {
                "_Unsigned": "true",
                "add_offset": 220.0,
                "scale_factor": 0.1,
                "_FillValue": 32767.0,
                "missing_value": 32767,
            }
            attrs = rds.air_temperature.attrs
            attrs.pop("transform")
            assert attrs == {
                "coordinates": "day",
                "coordinate_system": "WGS84,EPSG:4326",
                "description": "Daily Maximum Temperature",
                "dimensions": "lon lat time",
                "grid_mapping": "spatial_ref",
                "long_name": "tmmx",
                "standard_name": "tmmx",
                "units": "K",
            }


def test_no_mask_and_scale():
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc"), masked=True
    ) as rds:
        assert np.nanmin(rds.air_temperature.values) == 287
        assert np.nanmax(rds.air_temperature.values) == 821
        assert rds.air_temperature.encoding == {
            "_FillValue": 32767.0,
            "missing_value": 32767,
        }
        attrs = rds.air_temperature.attrs
        attrs.pop("transform")
        assert attrs == {
            "_Unsigned": "true",
            "add_offset": 220.0,
            "coordinates": "day",
            "coordinate_system": "WGS84,EPSG:4326",
            "description": "Daily Maximum Temperature",
            "dimensions": "lon lat time",
            "grid_mapping": "spatial_ref",
            "long_name": "tmmx",
            "scale_factor": 0.1,
            "standard_name": "tmmx",
            "units": "K",
        }


def test_notgeoreferenced_warning():
    with create_tmp_geotiff(transform_args=None) as (tmp_file, expected):
        with pytest.warns(NotGeoreferencedWarning):
            rioxarray.open_rasterio(tmp_file)


def test_nc_attr_loading():
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc")
    ) as rds:
        assert rds.dims == {"y": 10, "x": 10, "time": 2}
        assert rds.attrs == {"coordinates": "spatial_ref"}
        assert rds.y.attrs["units"] == "metre"
        assert rds.x.attrs["units"] == "metre"
        assert rds.time.encoding == {
            "units": "seconds since 2016-12-19T10:27:29.687763",
            "calendar": "proleptic_gregorian",
        }
        assert str(rds.time.values[0]) == "2016-12-19 10:27:29"
        assert str(rds.time.values[1]) == "2016-12-29 12:52:41.659696"
