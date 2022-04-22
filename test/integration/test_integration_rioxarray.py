import json
import os
import platform
import threading
from functools import partial

import dask.array as da
import numpy
import pytest
import rasterio
import xarray
from affine import Affine
from dask.delayed import Delayed
from numpy.testing import assert_almost_equal, assert_array_equal
from packaging import version
from pyproj import CRS as pCRS
from pyproj import Transformer
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.windows import Window

import rioxarray
from rioxarray.exceptions import (
    DimensionMissingCoordinateError,
    MissingCRS,
    MissingSpatialDimensionError,
    NoDataInBounds,
    OneDimensionalRaster,
    RioXarrayError,
)
from rioxarray.rioxarray import _make_coords
from test.conftest import (
    PYPROJ_LT_3,
    RASTERIO_EQ_122,
    RASTERIO_LT_122,
    TEST_COMPARE_DATA_DIR,
    TEST_INPUT_DATA_DIR,
    _assert_xarrays_equal,
    open_rasterio_engine,
)

try:
    import scipy

    SCIPY_VERSION = scipy.__version__
    SCIPY_INSTALLED = True
except ModuleNotFoundError:
    SCIPY_VERSION = "0.0.0"
    SCIPY_INSTALLED = False


@pytest.fixture(
    params=[
        xarray.open_dataset,
        xarray.open_dataarray,
        open_rasterio_engine,
    ]
)
def modis_reproject(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_UTM.nc"),
        to_proj="+datum=WGS84 +no_defs +proj=utm +units=m +zone=15",
        open=request.param,
    )


@pytest.fixture
def modis_reproject_3d():
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "PLANET_SCOPE_WGS84.nc"),
        to_proj="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
    )


@pytest.fixture(
    params=[
        xarray.open_dataset,
        xarray.open_dataarray,
        open_rasterio_engine,
    ]
)
def interpolate_na(request):
    pytest.importorskip("scipy")
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_INTERPOLATE.nc"),
        open=request.param,
    )


@pytest.fixture
def interpolate_na_3d():
    pytest.importorskip("scipy")
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "PLANET_SCOPE_3D_INTERPOLATE.nc"),
    )


@pytest.fixture(
    params=[
        xarray.open_dataset,
        xarray.open_dataarray,
        open_rasterio_engine,
    ]
)
def interpolate_na_filled(request):
    pytest.importorskip("scipy")
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(
            TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_INTERPOLATE_FILLED.nc"
        ),
        open=request.param,
    )


@pytest.fixture
def interpolate_na_veris():
    pytest.importorskip("scipy")
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "veris.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "veris_interpolate.nc"),
    )


@pytest.fixture(
    params=[
        xarray.open_dataarray,
        rioxarray.open_rasterio,
        xarray.open_dataset,
        open_rasterio_engine,
    ]
)
def interpolate_na_nan(request):
    pytest.importorskip("scipy")
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_INTERPOLATE_NAN.nc"),
        open=request.param,
    )


@pytest.fixture(
    params=[
        xarray.open_dataset,
        xarray.open_dataarray,
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
        open_rasterio_engine,
    ]
)
def modis_reproject_match(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_MATCH_UTM.nc"),
        match=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY_MATCH.nc"),
        open=request.param,
    )


@pytest.fixture(
    params=[
        xarray.open_dataset,
        xarray.open_dataarray,
        rioxarray.open_rasterio,
        open_rasterio_engine,
    ]
)
def modis_reproject_match_coords(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_MATCH_UTM.nc"),
        match=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY_MATCH.nc"),
        open=request.param,
    )


@pytest.fixture(
    params=[
        xarray.open_dataset,
        xarray.open_dataarray,
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
        open_rasterio_engine,
    ]
)
def modis_reproject_match__passed_nodata(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(
            TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_MATCH_PASSED_NODATA.nc"
        ),
        match=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY_MATCH.nc"),
        open=request.param,
    )


def _mod_attr(input_xr, attr, val=None, remove=False):
    if hasattr(input_xr, "variables"):
        for var in input_xr.rio.vars:
            _mod_attr(input_xr[var], attr, val=val, remove=remove)
    else:
        if remove:
            input_xr.attrs.pop(attr, None)
        else:
            input_xr.attrs[attr] = val


def _get_attr(input_xr, attr):
    if hasattr(input_xr, "variables"):
        return input_xr[input_xr.rio.vars.pop()].attrs[attr]
    return input_xr.attrs[attr]


def _del_attr(input_xr, attr):
    _mod_attr(input_xr, attr, remove=True)


@pytest.fixture(
    params=[
        xarray.open_dataset,
        xarray.open_dataarray,
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
        open_rasterio_engine,
    ]
)
def modis_clip(request, tmpdir):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_CLIP.nc"),
        compare_expand=os.path.join(
            TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_CLIP_EXPAND.nc"
        ),
        open=request.param,
        output=str(tmpdir.join("MODIS_CLIP_DUMP.nc")),
    )


def test_pad_box(modis_clip):
    if isinstance(modis_clip["open"], partial):
        # SKIP: parse_coodinates=False is not supported
        return
    with modis_clip["open"](modis_clip["input"]) as xdi:
        # first, clip
        clipped_ds = xdi.rio.clip_box(
            minx=xdi.x[4].values,
            miny=xdi.y[6].values,
            maxx=xdi.x[6].values,
            maxy=xdi.y[4].values,
        )
        # then, extend back to original
        padded_ds = clipped_ds.rio.pad_box(
            minx=xdi.x[0].values,
            miny=xdi.y[-1].values,
            maxx=xdi.x[-1].values,
            maxy=xdi.y[0].values,
        )
        # check the nodata value
        try:
            nodata = padded_ds[padded_ds.rio.vars[0]].rio.nodata
            if nodata is not None and not numpy.isnan(nodata):
                assert all(
                    [padded_ds[padded_ds.rio.vars[0]].isel(x=0, y=0).values == nodata]
                )
            else:
                assert all(
                    numpy.isnan(
                        [padded_ds[padded_ds.rio.vars[0]].isel(x=0, y=0).values]
                    )
                )
        except AttributeError:
            if padded_ds.rio.nodata is not None and not numpy.isnan(
                padded_ds.rio.nodata
            ):
                assert all([padded_ds.isel(x=0, y=0).values == padded_ds.rio.nodata])
            else:
                assert all(numpy.isnan([padded_ds.isel(x=0, y=0).values]))
        # finally, clip again
        clipped_ds2 = padded_ds.rio.clip_box(
            minx=xdi.x[4].values,
            miny=xdi.y[6].values,
            maxx=xdi.x[6].values,
            maxy=xdi.y[4].values,
        )
        _assert_xarrays_equal(clipped_ds, clipped_ds2)
        # padded data should have the same size as original data
        if hasattr(xdi, "variables"):
            for var in xdi.rio.vars:
                assert_almost_equal(
                    xdi[var].rio._cached_transform(),
                    padded_ds[var].rio._cached_transform(),
                )
                for padded_size, original_size in zip(
                    padded_ds[var].shape, xdi[var].shape
                ):
                    assert padded_size == original_size
        else:
            assert_almost_equal(
                xdi.rio._cached_transform(), padded_ds.rio._cached_transform()
            )
            for padded_size, original_size in zip(padded_ds.shape, xdi.shape):
                assert padded_size == original_size
        # make sure it safely writes to netcdf
        padded_ds.to_netcdf(modis_clip["output"])


def test_clip_box(modis_clip):
    with modis_clip["open"](modis_clip["input"]) as xdi, modis_clip["open"](
        modis_clip["compare"]
    ) as xdc:
        clipped_ds = xdi.rio.clip_box(
            minx=-7272967.195874103,  # xdi.x[4].values,
            miny=5048602.8438240355,  # xdi.y[6].values,
            maxx=-7272503.8831575755,  # xdi.x[6].values,
            maxy=5049066.156540562,  # xdi.y[4].values,
        )
        assert xdi.rio._cached_transform() != clipped_ds.rio._cached_transform()
        var = "__xarray_dataarray_variable__"
        try:
            clipped_ds_values = clipped_ds[var].values
        except KeyError:
            clipped_ds_values = clipped_ds.values
        try:
            xdc_values = xdc[var].values
        except KeyError:
            xdc_values = xdc.values
        assert_almost_equal(clipped_ds_values, xdc_values)
        assert_almost_equal(clipped_ds.rio.transform(), xdc.rio.transform())
        # make sure it safely writes to netcdf
        clipped_ds.to_netcdf(modis_clip["output"])


def test_clip_box__auto_expand(modis_clip):
    with modis_clip["open"](modis_clip["input"]) as xdi, modis_clip["open"](
        modis_clip["compare_expand"]
    ) as xdc:
        clipped_ds = xdi.rio.clip_box(
            minx=-7272735.53951584,  # xdi.x[5].values
            miny=5048834.500182299,  # xdi.y[5].values
            maxx=-7272735.53951584,  # xdi.x[5].values
            maxy=5048834.500182299,  # xdi.y[5].values
            auto_expand=True,
        )
        assert xdi.rio._cached_transform() != clipped_ds.rio._cached_transform()
        var = "__xarray_dataarray_variable__"
        try:
            clipped_ds_values = clipped_ds[var].values
        except KeyError:
            clipped_ds_values = clipped_ds.values
        try:
            xdc_values = xdc[var].values
        except KeyError:
            xdc_values = xdc.values
        assert_almost_equal(clipped_ds_values, xdc_values)
        assert_almost_equal(clipped_ds.rio.transform(), xdc.rio.transform())
        # make sure it safely writes to netcdf
        clipped_ds.to_netcdf(modis_clip["output"])


def test_clip_box__nodata_error(modis_clip):
    with modis_clip["open"](modis_clip["input"]) as xdi:
        var_match = ""
        if hasattr(xdi, "name") and xdi.name:
            var_match = " Data variable: __xarray_dataarray_variable__"
        if RASTERIO_LT_122:
            expected_exception = NoDataInBounds
        else:
            expected_exception = rasterio.errors.WindowError
            var_match = "Bounds and transform are inconsistent"

        with pytest.raises(expected_exception, match=var_match):
            xdi.rio.clip_box(
                minx=-8272735.53951584,  # xdi.x[5].values
                miny=8048371.187465771,  # xdi.y[7].values
                maxx=-8272967.195874103,  # xdi.x[4].values
                maxy=8048834.500182299,  # xdi.y[5].values
            )


def test_clip_box__one_dimension_error(modis_clip):
    with modis_clip["open"](modis_clip["input"]) as xdi:
        var_match = ""
        if hasattr(xdi, "name") and xdi.name:
            var_match = " Data variable: __xarray_dataarray_variable__"
        if RASTERIO_EQ_122:
            expected_exception = rasterio.errors.WindowError
            var_match = "Bounds and transform are inconsistent"
        else:
            expected_exception = OneDimensionalRaster
            var_match = (
                "At least one of the clipped raster x,y coordinates has "
                f"only one point.{var_match}"
            )
        # test exception after raster clipped
        with pytest.raises(
            expected_exception,
            match=var_match,
        ):
            xdi.rio.clip_box(
                minx=-7272735.53951584,  # xdi.x[5].values
                miny=5048834.500182299,  # xdi.y[5].values
                maxx=-7272735.53951584,  # xdi.x[5].values
                maxy=5048834.500182299,  # xdi.y[5].values
            )
        # test exception before raster clipped
        with pytest.raises(OneDimensionalRaster):
            xdi.isel(x=slice(5, 6), y=slice(5, 6)).rio.clip_box(
                minx=-7272735.53951584,  # xdi.x[5].values
                miny=5048371.187465771,  # xdi.y[7].values
                maxx=-7272272.226799311,  # xdi.x[7].values
                maxy=5048834.500182299,  # xdi.y[5].values
            )


def test_clip_box__nodata_in_bounds():
    xds = rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "clip_bbox__out_of_bounds.tif")
    )
    with pytest.raises(NoDataInBounds):
        xds.rio.clip_box(
            *Transformer.from_crs(
                xds.rio.crs, "EPSG:4326", always_xy=True
            ).transform_bounds(
                135.7821043660001123,
                -17.1608065079999506,
                135.7849362810001139,
                -17.1580839999999739,
            )
        )

    with pytest.raises(NoDataInBounds):
        xds.rio.reproject("EPSG:4326").rio.clip_box(
            135.7821043660001123,
            -17.1608065079999506,
            135.7849362810001139,
            -17.1580839999999739,
        )


def test_slice_xy(modis_clip):
    if isinstance(modis_clip["open"], partial):
        # SKIP: parse_coodinates=False is not supported
        return
    with modis_clip["open"](modis_clip["input"]) as xdi, modis_clip["open"](
        modis_clip["compare"]
    ) as xdc:
        clipped_ds = xdi.rio.slice_xy(
            minx=-7272967.195874103,  # xdi.x[4].values,
            miny=5048602.8438240355,  # xdi.y[6].values,
            maxx=-7272503.8831575755,  # xdi.x[6].values,
            maxy=5049297.812898826,  # xdi.y[4].values - resolution_y,
        )
        assert xdi.rio._cached_transform() != clipped_ds.rio._cached_transform()
        var = "__xarray_dataarray_variable__"
        try:
            clipped_ds_values = clipped_ds[var].values
        except KeyError:
            clipped_ds_values = clipped_ds.values
        try:
            xdc_values = xdc[var].values
        except KeyError:
            xdc_values = xdc.values
        assert_almost_equal(clipped_ds_values, xdc_values)
        assert_almost_equal(clipped_ds.rio.transform(), xdc.rio.transform())
        # make sure it safely writes to netcdf
        clipped_ds.to_netcdf(modis_clip["output"])


@pytest.mark.parametrize("from_disk", [True, False])
@pytest.mark.parametrize(
    "open_func",
    [
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
        partial(rioxarray.open_rasterio, parse_coordinates=False, masked=True),
        open_rasterio_engine,
        partial(open_rasterio_engine, parse_coordinates=False),
        partial(open_rasterio_engine, parse_coordinates=False, mask_and_scale=False),
    ],
)
def test_clip_geojson(open_func, from_disk):
    with open_func(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif"),
    ) as xdi:
        # get subset for testing
        subset = xdi.isel(x=slice(150, 160), y=slice(100, 150))
        comp_subset = subset.isel(x=slice(1, None), y=slice(1, None))
        if isinstance(comp_subset, xarray.Dataset):
            comp_subset = comp_subset.band_data
        # add transform for test
        comp_subset.rio.write_transform(inplace=True)
        # add grid mapping for test
        comp_subset.rio.write_crs(subset.rio.crs, inplace=True)
        if comp_subset.rio.encoded_nodata is None:
            comp_subset.rio.write_nodata(comp_subset.rio.nodata, inplace=True)

        geometries = [
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [425499.18381405267, 4615331.540546387],
                        [425499.18381405267, 4615478.540546387],
                        [425526.18381405267, 4615478.540546387],
                        [425526.18381405267, 4615331.540546387],
                        [425499.18381405267, 4615331.540546387],
                    ]
                ],
            }
        ]
        # test data array
        clipped = xdi.rio.clip(geometries, from_disk=from_disk)
        if from_disk and not isinstance(clipped, xarray.Dataset):
            _assert_xarrays_equal(clipped[:, 1:, 1:], comp_subset)
            if comp_subset.rio.encoded_nodata is not None:
                assert numpy.isnan(clipped.values[:, 0, :]).all()
                assert numpy.isnan(clipped.values[:, :, 0]).all()
            else:
                assert (clipped.values[:, 0, :] == comp_subset.rio.nodata).all()
                assert (clipped.values[:, :, 0] == comp_subset.rio.nodata).all()
        else:
            if isinstance(clipped, xarray.Dataset):
                clipped = clipped.band_data
            _assert_xarrays_equal(clipped, comp_subset)

        # test dataset
        if isinstance(xdi, xarray.DataArray):
            clipped_ds = xdi.to_dataset(name="band_data").rio.clip(geometries)
        else:
            clipped_ds = xdi.rio.clip(geometries)
        comp_subset_ds = comp_subset.to_dataset(name="band_data")
        # This coordinate checking is skipped when parse_coordinates=False
        # as the auto-generated coordinates differ and can be ignored
        _assert_xarrays_equal(
            clipped_ds, comp_subset_ds, skip_xy_check=isinstance(open_func, partial)
        )
        # check the transform
        assert_almost_equal(
            clipped_ds.rio.transform(),
            (3.0, 0.0, 425500.68381405267, 0.0, -3.0, 4615477.040546387, 0.0, 0.0, 1.0),
        )


@pytest.mark.parametrize(
    "invert, from_disk, expected_sum",
    [
        (False, False, 2150837592),
        (True, False, 535691205),
        (False, True, 2150837592),
        (True, True, 535691205),
    ],
)
@pytest.mark.parametrize(
    "open_func",
    [
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
    ],
)
def test_clip_geojson__no_drop(open_func, invert, from_disk, expected_sum):
    with open_func(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif")
    ) as xdi:
        geometries = [
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-93.880889448126, 41.68465068553298],
                        [-93.89966980835203, 41.68465068553298],
                        [-93.89966980835203, 41.689430423525266],
                        [-93.880889448126, 41.689430423525266],
                        [-93.880889448126, 41.68465068553298],
                    ]
                ],
            }
        ]
        # test data array
        clipped = xdi.rio.clip(
            geometries, "epsg:4326", drop=False, invert=invert, from_disk=from_disk
        )
        assert clipped.rio.crs == xdi.rio.crs
        assert clipped.shape == xdi.shape
        assert clipped.sum().item() == expected_sum
        assert clipped.rio.nodata == 0.0
        assert clipped.rio.nodata == xdi.rio.nodata

        # test dataset
        clipped_ds = xdi.to_dataset(name="test_data").rio.clip(
            geometries, "epsg:4326", drop=False, invert=invert
        )
        assert clipped_ds.rio.crs == xdi.rio.crs
        assert clipped_ds.test_data.shape == xdi.shape
        assert clipped_ds.test_data.sum().item() == expected_sum
        assert clipped_ds.test_data.rio.nodata == xdi.rio.nodata


@pytest.fixture
def dummy_dataset_non_geospatial():
    ds = xarray.Dataset(
        {
            "stuff": xarray.DataArray(
                data=numpy.zeros((6, 6, 6), dtype=float),
                dims=["time", "y", "x"],
                coords={
                    "time": numpy.arange(6),
                    "y": numpy.linspace(4615514.54054639, 4615295.54054639, num=6),
                    "x": numpy.linspace(425493.18381405, 425532.18381405, num=6),
                },
                attrs={
                    "_FillValue": -1,
                },
            ),
            "non_geo_stuff": xarray.DataArray(
                data=numpy.zeros((6, 6), dtype=float),
                dims=["time", "bb"],
                coords={
                    "time": numpy.arange(6),
                    "bb": numpy.arange(6),
                },
            ),
            "meta": ("time", numpy.random.random(6)),
        }
    )
    ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    ds.rio.write_crs("EPSG:26915", inplace=True)
    return ds


def test_clip__non_geospatial(dummy_dataset_non_geospatial):
    # check for variables without spatial dims
    geometries = [
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [425499.18381405267, 4615331.540546387],
                    [425499.18381405267, 4615478.540546387],
                    [425526.18381405267, 4615478.540546387],
                    [425526.18381405267, 4615331.540546387],
                    [425499.18381405267, 4615331.540546387],
                ]
            ],
        }
    ]
    ds = dummy_dataset_non_geospatial
    assert ds.stuff.shape == (6, 6, 6)
    with pytest.raises(MissingSpatialDimensionError):
        ds.rio.clip(geometries)

    ds_clip = ds[["stuff", "meta"]].rio.clip(geometries)
    assert ds_clip.stuff.shape == (6, 4, 4)
    assert "meta" in ds_clip.data_vars

    with rioxarray.set_options(skip_missing_spatial_dims=True):
        ds_clip = ds.rio.clip(geometries)
        assert ds_clip.stuff.shape == (6, 4, 4)
        assert "meta" in ds_clip.data_vars
        assert "non_geo_stuff" in ds_clip.data_vars


def test_clip_box__non_geospatial(dummy_dataset_non_geospatial):
    ds = dummy_dataset_non_geospatial
    assert ds.stuff.shape == (6, 6, 6)
    with pytest.raises(MissingSpatialDimensionError):
        ds.rio.clip_box(
            minx=425500.18381405,
            miny=4615395.54054639,
            maxx=425520.18381405,
            maxy=4615414.54054639,
        )

    ds_clip_box = ds[["stuff", "meta"]].rio.clip_box(
        minx=425500.18381405,
        miny=4615395.54054639,
        maxx=425520.18381405,
        maxy=4615414.54054639,
    )
    assert ds_clip_box.stuff.shape == (6, 2, 3)
    assert "meta" in ds_clip_box.data_vars

    with rioxarray.set_options(skip_missing_spatial_dims=True):
        ds_clip_box = ds.rio.clip_box(
            minx=425500.18381405,
            miny=4615395.54054639,
            maxx=425520.18381405,
            maxy=4615414.54054639,
        )
        assert ds_clip_box.stuff.shape == (6, 2, 3)
        assert "meta" in ds_clip_box.data_vars
        assert "non_geo_stuff" in ds_clip_box.data_vars


def test_reproject__non_geospatial(dummy_dataset_non_geospatial):
    ds = dummy_dataset_non_geospatial
    assert ds.stuff.shape == (6, 6, 6)
    with pytest.raises(MissingSpatialDimensionError):
        ds.rio.reproject("EPSG:4326")

    ds_reproject = ds[["stuff", "meta"]].rio.reproject("EPSG:4326")
    assert ds_reproject.stuff.shape == (6, 8, 2)
    assert "meta" in ds_reproject.data_vars

    with rioxarray.set_options(skip_missing_spatial_dims=True):
        ds_reproject = ds.rio.reproject("EPSG:4326")
        assert ds_reproject.stuff.shape == (6, 8, 2)
        assert "meta" in ds_reproject.data_vars
        assert "non_geo_stuff" in ds_reproject.data_vars


def test_reproject_match__non_geospatial(dummy_dataset_non_geospatial):
    ds = dummy_dataset_non_geospatial
    assert ds.stuff.shape == (6, 6, 6)
    with pytest.raises(MissingSpatialDimensionError):
        ds.rio.reproject_match(ds)

    ds_reproject = ds[["stuff", "meta"]].rio.reproject_match(ds)
    assert ds_reproject.stuff.shape == (6, 6, 6)
    assert "meta" in ds_reproject.data_vars

    with rioxarray.set_options(skip_missing_spatial_dims=True):
        ds_reproject = ds.rio.reproject_match(ds)
        assert ds_reproject.stuff.shape == (6, 6, 6)
        assert "meta" in ds_reproject.data_vars
        assert "non_geo_stuff" in ds_reproject.data_vars


def test_interpolate_na__non_geospatial(dummy_dataset_non_geospatial):
    pytest.importorskip("scipy")
    ds = dummy_dataset_non_geospatial
    assert ds.stuff.shape == (6, 6, 6)
    with pytest.raises(MissingSpatialDimensionError):
        ds.rio.interpolate_na()

    ds_interp = ds[["stuff", "meta"]].rio.interpolate_na()
    assert ds_interp.stuff.shape == (6, 6, 6)
    assert "meta" in ds_interp.data_vars

    with rioxarray.set_options(skip_missing_spatial_dims=True):
        ds_interp = ds.rio.interpolate_na()
        assert ds_interp.stuff.shape == (6, 6, 6)
        assert "meta" in ds_interp.data_vars
        assert "non_geo_stuff" in ds_interp.data_vars


def test_pad_box__non_geospatial(dummy_dataset_non_geospatial):
    ds = dummy_dataset_non_geospatial
    assert ds.stuff.shape == (6, 6, 6)
    with pytest.raises(MissingSpatialDimensionError):
        ds.rio.pad_box(*ds.rio.bounds())

    ds_pad_box = (
        ds[["stuff", "meta"]]
        .rio.clip_box(
            minx=425500.18381405,
            miny=4615395.54054639,
            maxx=425520.18381405,
            maxy=4615414.54054639,
        )
        .rio.pad_box(*ds.rio.bounds())
    )
    assert ds_pad_box.stuff.shape == (6, 6, 6)
    assert "meta" in ds_pad_box.data_vars

    with rioxarray.set_options(skip_missing_spatial_dims=True):
        ds_pad_box = ds.rio.clip_box(
            minx=425500.18381405,
            miny=4615395.54054639,
            maxx=425520.18381405,
            maxy=4615414.54054639,
        ).rio.pad_box(*ds.rio.bounds())
        assert ds_pad_box.stuff.shape == (6, 6, 6)
        assert "meta" in ds_pad_box.data_vars
        assert "non_geo_stuff" in ds_pad_box.data_vars


@pytest.mark.parametrize(
    "open_func",
    [
        xarray.open_dataset,
        xarray.open_dataarray,
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
        open_rasterio_engine,
        partial(open_rasterio_engine, parse_coordinates=False),
    ],
)
def test_transform_bounds(open_func):
    with open_func(os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")) as xdi:
        bounds = xdi.rio.transform_bounds(
            "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84"
            " +datum=WGS84 +units=m +no_defs",
            densify_pts=100,
        )
        assert_almost_equal(
            bounds,
            (
                -10374232.525903117,
                5591295.917919335,
                -10232919.684719983,
                5656912.314724255,
            ),
        )


def test_reproject_with_shape(modis_reproject):
    new_shape = (9, 10)
    mask_args = (
        dict(masked=False, mask_and_scale=False)
        if "rasterio" in str(modis_reproject["open"])
        else dict(mask_and_scale=False)
    )
    with modis_reproject["open"](modis_reproject["input"], **mask_args) as mda:
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"], shape=new_shape)
        # test
        if hasattr(mds_repr, "variables"):
            for var in mds_repr.rio.vars:
                assert mds_repr[var].rio.shape == new_shape
        else:
            assert mds_repr.rio.shape == new_shape


def test_reproject(modis_reproject):
    mask_args = (
        dict(masked=False, mask_and_scale=False)
        if "rasterio" in str(modis_reproject["open"])
        else dict(mask_and_scale=False)
    )
    with modis_reproject["open"](
        modis_reproject["input"], **mask_args
    ) as mda, modis_reproject["open"](modis_reproject["compare"], **mask_args) as mdc:
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)
        assert mds_repr.coords[mds_repr.rio.x_dim].attrs == {
            "axis": "X",
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
            "units": "metre",
        }
        assert mds_repr.coords[mds_repr.rio.y_dim].attrs == {
            "axis": "Y",
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
            "units": "metre",
        }


@pytest.mark.parametrize(
    "open_func",
    [
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
        open_rasterio_engine,
        partial(open_rasterio_engine, parse_coordinates=False),
    ],
)
def test_reproject_3d(open_func, modis_reproject_3d):
    with open_func(modis_reproject_3d["input"]) as mda, open_func(
        modis_reproject_3d["compare"]
    ) as mdc:
        mds_repr = mda.rio.reproject(modis_reproject_3d["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)
        if mdc.rio.x_dim in mdc.coords:
            assert mds_repr.coords[mds_repr.rio.x_dim].attrs == {
                "long_name": "longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "axis": "X",
            }
            assert mds_repr.coords[mds_repr.rio.y_dim].attrs == {
                "long_name": "latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "axis": "Y",
            }


def test_reproject__grid_mapping(modis_reproject):
    mask_args = (
        dict(masked=False, mask_and_scale=False)
        if "rasterio" in str(modis_reproject["open"])
        else dict(mask_and_scale=False)
    )
    with modis_reproject["open"](
        modis_reproject["input"], **mask_args
    ) as mda, modis_reproject["open"](modis_reproject["compare"], **mask_args) as mdc:

        # remove 'crs' attribute and add grid mapping
        # keep a copy of the crs before setting the spatial_ref to 0 as that will
        # set `rio.crs` to None.
        mda_rio_crs = mda.rio.crs
        mda.coords["spatial_ref"] = 0
        mda.coords["spatial_ref"].attrs["spatial_ref"] = CRS.from_user_input(
            mda_rio_crs
        ).wkt
        _mod_attr(mda, "grid_mapping", val="spatial_ref")
        # keep a copy of the crs before setting the spatial_ref to 0 as that will
        # set `rio.crs` to None.
        mdc_rio_crs = mdc.rio.crs
        mdc.coords["spatial_ref"] = 0
        mdc.coords["spatial_ref"].attrs["spatial_ref"] = CRS.from_user_input(
            mdc_rio_crs
        ).wkt
        _mod_attr(mdc, "grid_mapping", val="spatial_ref")

        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject__masked(modis_reproject):
    with modis_reproject["open"](modis_reproject["input"]) as mda, modis_reproject[
        "open"
    ](modis_reproject["compare"]) as mdc:
        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject__no_transform(modis_reproject):
    with modis_reproject["open"](modis_reproject["input"]) as mda, modis_reproject[
        "open"
    ](modis_reproject["compare"]) as mdc:
        orig_trans = mda.rio.transform()
        _del_attr(mda, "transform")
        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        if hasattr(mds_repr, "variables"):
            for var in mds_repr.rio.vars:
                assert_array_equal(orig_trans, tuple(mda[var].rio.transform()))
        else:
            assert_array_equal(orig_trans, tuple(mda.rio.transform()))
        _assert_xarrays_equal(mds_repr, mdc)


@pytest.mark.parametrize("nodata", [None, -9999])
def test_reproject__no_nodata(nodata, modis_reproject):
    mask_args = (
        dict(masked=False, mask_and_scale=False)
        if "rasterio" in str(modis_reproject["open"])
        else dict(mask_and_scale=False)
    )
    with modis_reproject["open"](
        modis_reproject["input"], **mask_args
    ) as mda, modis_reproject["open"](modis_reproject["compare"], **mask_args) as mdc:
        orig_fill = _get_attr(mda, "_FillValue")
        _del_attr(mda, "_FillValue")
        _del_attr(mda, "nodata")
        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"], nodata=nodata)

        # overwrite test dataset
        # if isinstance(modis_reproject['open'], xarray.DataArray):
        #    mds_repr.to_netcdf(modis_reproject['compare'])

        # replace -9999 with original _FillValue for testing
        fill_nodata = -32768 if nodata is None else nodata
        if hasattr(mds_repr, "variables"):
            for var in mds_repr.rio.vars:
                mds_repr[var].values[mds_repr[var].values == fill_nodata] = orig_fill
        else:
            mds_repr.values[mds_repr.values == fill_nodata] = orig_fill
        _mod_attr(mdc, "_FillValue", val=fill_nodata)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


@pytest.mark.parametrize("open_func", [rioxarray.open_rasterio, open_rasterio_engine])
def test_reproject__scalar_coord(open_func):
    with open_func(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif")
    ) as xdi:
        xdi_repr = xdi.squeeze().rio.reproject("epsg:3395")
        for coord in xdi.coords:
            assert coord in xdi_repr.coords


def test_reproject__no_nodata_masked(modis_reproject):
    with modis_reproject["open"](modis_reproject["input"]) as mda, modis_reproject[
        "open"
    ](modis_reproject["compare"]) as mdc:
        _del_attr(mda, "nodata")
        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject__gcps_kwargs(tmp_path):
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

    rds = rioxarray.open_rasterio(tiffname)
    rds.rio.write_crs(crs, inplace=True)
    rds = rds.rio.reproject(
        crs,
        gcps=src_gcps,
    )
    assert rds.rio.height == 923
    assert rds.rio.width == 1027
    assert rds.rio.crs == crs
    assert rds.rio.transform().almost_equals(
        Affine(
            216.8587081056465,
            0.0,
            115698.25,
            0.0,
            -216.8587081056465,
            2818720.0,
        )
    )
    assert (rds.coords["x"].values > 11000).all()
    assert (rds.coords["y"].values > 281700).all()


def test_reproject_match(modis_reproject_match):
    mask_args = (
        dict(masked=False, mask_and_scale=False)
        if "rasterio" in str(modis_reproject_match["open"])
        else dict(mask_and_scale=False)
    )
    with modis_reproject_match["open"](
        modis_reproject_match["input"], **mask_args
    ) as mda, modis_reproject_match["open"](
        modis_reproject_match["compare"], **mask_args
    ) as mdc, xarray.open_dataarray(
        modis_reproject_match["match"]
    ) as mdm:
        # reproject
        mds_repr = mda.rio.reproject_match(mdm)
        # test
        _assert_xarrays_equal(mds_repr, mdc)
        if mdc.rio.x_dim in mdc.coords:
            assert mds_repr.coords[mds_repr.rio.x_dim].attrs == {
                "axis": "X",
                "long_name": "x coordinate of projection",
                "standard_name": "projection_x_coordinate",
                "units": "metre",
            }
            assert mds_repr.coords[mds_repr.rio.y_dim].attrs == {
                "axis": "Y",
                "long_name": "y coordinate of projection",
                "standard_name": "projection_y_coordinate",
                "units": "metre",
            }


def test_reproject_match__masked(modis_reproject_match):
    mask_args = (
        dict(masked=True)
        if "rasterio" in str(modis_reproject_match["open"])
        else dict(mask_and_scale=True)
    )
    with modis_reproject_match["open"](
        modis_reproject_match["input"], **mask_args
    ) as mda, modis_reproject_match["open"](
        modis_reproject_match["compare"], **mask_args
    ) as mdc, xarray.open_dataarray(
        modis_reproject_match["match"]
    ) as mdm:
        # reproject
        mds_repr = mda.rio.reproject_match(mdm)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject_match__no_transform_nodata(modis_reproject_match_coords):
    mask_args = (
        dict(masked=True)
        if "rasterio" in str(modis_reproject_match_coords["open"])
        else dict(mask_and_scale=True)
    )
    with modis_reproject_match_coords["open"](
        modis_reproject_match_coords["input"], **mask_args
    ) as mda, modis_reproject_match_coords["open"](
        modis_reproject_match_coords["compare"], **mask_args
    ) as mdc, xarray.open_dataarray(
        modis_reproject_match_coords["match"]
    ) as mdm:
        _del_attr(mda, "transform")
        _del_attr(mda, "nodata")
        # reproject
        mds_repr = mda.rio.reproject_match(mdm)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject_match__pass_nodata(modis_reproject_match__passed_nodata):
    mask_args = (
        dict(masked=False, mask_and_scale=False)
        if "rasterio" in str(modis_reproject_match__passed_nodata["open"])
        else dict(mask_and_scale=False, decode_coords="all")
    )
    with modis_reproject_match__passed_nodata["open"](
        modis_reproject_match__passed_nodata["input"], **mask_args
    ) as mda, modis_reproject_match__passed_nodata["open"](
        modis_reproject_match__passed_nodata["compare"], **mask_args
    ) as mdc, xarray.open_dataarray(
        modis_reproject_match__passed_nodata["match"]
    ) as mdm:
        # reproject
        mds_repr = mda.rio.reproject_match(mdm, nodata=-9999)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


@pytest.mark.parametrize("open_func", [rioxarray.open_rasterio, open_rasterio_engine])
def test_make_src_affine(open_func, modis_reproject):
    with xarray.open_dataarray(modis_reproject["input"]) as xdi, open_func(
        modis_reproject["input"]
    ) as xri:

        # check the transform
        attribute_transform = tuple(xdi.attrs["transform"])
        attribute_transform_func = tuple(xdi.rio.transform())
        calculated_transform = tuple(xdi.rio.transform(recalc=True))
        # delete the transform to ensure it is not being used
        del xdi.attrs["transform"]
        calculated_transform_check = tuple(xdi.rio.transform())
        calculated_transform_check2 = tuple(xdi.rio.transform())
        rio_transform = tuple(xri.rio._cached_transform())

        assert_array_equal(attribute_transform, attribute_transform_func)
        assert_array_equal(calculated_transform, calculated_transform_check)
        assert_array_equal(calculated_transform, calculated_transform_check2)
        assert_array_equal(attribute_transform, calculated_transform)
        assert_array_equal(calculated_transform, rio_transform)


def test_make_src_affine__single_point():
    point_input = os.path.join(TEST_INPUT_DATA_DIR, "MODIS_POINT.nc")
    with xarray.open_dataarray(point_input) as xdi:
        # check the transform
        attribute_transform = tuple(xdi.attrs["transform"])
        attribute_transform_func = tuple(xdi.rio.transform())
        calculated_transform = tuple(xdi.rio.transform(recalc=True))
        # delete the transform to ensure it is not being used
        del xdi.attrs["transform"]
        assert xdi.rio.transform(recalc=True) == Affine.identity()
        assert xdi.rio.transform() == Affine.identity()

        assert_array_equal(attribute_transform, attribute_transform_func)
        assert_array_equal(attribute_transform, calculated_transform)


@pytest.mark.parametrize(
    "open_func",
    [
        xarray.open_dataset,
        open_rasterio_engine,
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
        partial(open_rasterio_engine, parse_coordinates=False),
    ],
)
def test_make_coords__calc_trans(open_func, modis_reproject):
    with xarray.open_dataarray(modis_reproject["input"]) as xdi, open_func(
        modis_reproject["input"]
    ) as xri:
        # calculate coordinates from the calculated transform
        width, height = xdi.rio.shape
        calculated_transform = xdi.rio.transform(recalc=True)
        calc_coords_calc_trans = _make_coords(xdi, calculated_transform, width, height)
        widthr, heightr = xri.rio.shape
        calculated_transformr = xri.rio.transform(recalc=True)
        calc_coords_calc_transr = _make_coords(
            xri, calculated_transformr, widthr, heightr
        )

        assert_almost_equal(calculated_transform, calculated_transformr)
        # check to see if they all match
        if not isinstance(open_func, partial):
            assert_almost_equal(
                xri.coords["x"].values, calc_coords_calc_trans["x"].values, decimal=9
            )
            assert_almost_equal(
                xri.coords["y"].values, calc_coords_calc_trans["y"].values, decimal=9
            )
            assert_almost_equal(
                xri.coords["x"].values, calc_coords_calc_transr["x"].values, decimal=9
            )
            assert_almost_equal(
                xri.coords["y"].values, calc_coords_calc_transr["y"].values, decimal=9
            )


@pytest.mark.parametrize(
    "open_func",
    [
        xarray.open_dataset,
        rioxarray.open_rasterio,
        partial(rioxarray.open_rasterio, parse_coordinates=False),
        open_rasterio_engine,
        partial(open_rasterio_engine, parse_coordinates=False),
    ],
)
def test_make_coords__attr_trans(open_func, modis_reproject):
    with xarray.open_dataarray(modis_reproject["input"]) as xdi, open_func(
        modis_reproject["input"]
    ) as xri:
        # calculate coordinates from the attribute transform
        width, height = xdi.rio.shape
        attr_transform = xdi.rio.transform()
        calc_coords_attr_trans = _make_coords(xdi, attr_transform, width, height)
        widthr, heightr = xri.rio.shape
        calculated_transformr = xri.rio.transform()
        calc_coords_calc_transr = _make_coords(
            xri, calculated_transformr, widthr, heightr
        )
        assert_almost_equal(attr_transform, calculated_transformr)
        # check to see if they all match
        if not isinstance(open_func, partial):
            assert_almost_equal(
                xri.coords["x"].values, calc_coords_calc_transr["x"].values, decimal=9
            )
            assert_almost_equal(
                xri.coords["y"].values, calc_coords_calc_transr["y"].values, decimal=9
            )
            assert_almost_equal(
                xri.coords["x"].values, calc_coords_attr_trans["x"].values, decimal=9
            )
            assert_almost_equal(
                xri.coords["y"].values, calc_coords_attr_trans["y"].values, decimal=9
            )
            assert_almost_equal(
                xdi.coords["x"].values, xri.coords["x"].values, decimal=9
            )
            assert_almost_equal(
                xdi.coords["y"].values, xri.coords["y"].values, decimal=9
            )


def test_interpolate_na(interpolate_na):
    mask_args = (
        dict(masked=False, mask_and_scale=False)
        if "rasterio" in str(interpolate_na["open"])
        else dict(mask_and_scale=False)
    )
    with interpolate_na["open"](
        interpolate_na["input"], **mask_args
    ) as mda, interpolate_na["open"](interpolate_na["compare"], **mask_args) as mdc:
        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


@pytest.mark.skipif(SCIPY_INSTALLED, reason="Only test if scipy is not installed.")
def test_interpolate_na__missing_scipy():
    xds = xarray.open_dataarray(os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"))
    with pytest.raises(ModuleNotFoundError, match=r"rioxarray\[interp\]"):
        xds.rio.interpolate_na()


@pytest.mark.xfail(
    version.parse(SCIPY_VERSION) < version.parse("1.7.0")
    or platform.system() != "Linux",
    reason="griddata behaves differently across versions and platforms",
)
def test_interpolate_na_veris(interpolate_na_veris):
    with xarray.open_dataset(interpolate_na_veris["input"]) as mda, xarray.open_dataset(
        interpolate_na_veris["compare"]
    ) as mdc:
        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_interpolate_na_3d(interpolate_na_3d):
    with xarray.open_dataset(interpolate_na_3d["input"]) as mda, xarray.open_dataset(
        interpolate_na_3d["compare"]
    ) as mdc:
        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_interpolate_na__nodata_filled(interpolate_na_filled):
    mask_args = (
        dict(masked=False, mask_and_scale=False)
        if "rasterio" in str(interpolate_na_filled["open"])
        else dict(mask_and_scale=False)
    )
    with interpolate_na_filled["open"](
        interpolate_na_filled["input"], **mask_args
    ) as mda, interpolate_na_filled["open"](
        interpolate_na_filled["compare"], **mask_args
    ) as mdc:
        if hasattr(mda, "variables"):
            for var in mda.rio.vars:
                mda[var].values[mda[var].values == mda[var].rio.nodata] = 400
        else:
            mda.values[mda.values == mda.rio.nodata] = 400

        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_interpolate_na__all_nodata(interpolate_na_nan):
    rio_opened = "open_rasterio " in str(interpolate_na_nan["open"])
    with interpolate_na_nan["open"](
        interpolate_na_nan["input"], mask_and_scale=True
    ) as mda, interpolate_na_nan["open"](
        interpolate_na_nan["compare"], mask_and_scale=True
    ) as mdc:
        if hasattr(mda, "variables"):
            for var in mda.rio.vars:
                mda[var].values[~numpy.isnan(mda[var].values)] = numpy.nan
        else:
            mda.values[~numpy.isnan(mda.values)] = numpy.nan

        interpolated_ds = mda.rio.interpolate_na()
        if rio_opened and "__xarray_dataarray_variable__" in mdc:
            mdc = mdc["__xarray_dataarray_variable__"]
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_load_in_geographic_dimensions():
    sentinel_2_geographic = os.path.join(
        TEST_INPUT_DATA_DIR, "sentinel_2_L1C_geographic.nc"
    )
    with xarray.open_dataset(sentinel_2_geographic) as mda:
        assert mda.rio.x_dim == "longitude"
        assert mda.rio.y_dim == "latitude"
        assert mda.rio.crs.to_epsg() == 4326
        assert mda.red.rio.x_dim == "longitude"
        assert mda.red.rio.y_dim == "latitude"
        assert mda.red.rio.crs.to_epsg() == 4326


def test_geographic_reproject():
    sentinel_2_geographic = os.path.join(
        TEST_INPUT_DATA_DIR, "sentinel_2_L1C_geographic.nc"
    )
    sentinel_2_utm = os.path.join(TEST_COMPARE_DATA_DIR, "sentinel_2_L1C_utm.nc")
    with xarray.open_dataset(sentinel_2_geographic) as mda, xarray.open_dataset(
        sentinel_2_utm
    ) as mdc:
        mds_repr = mda.rio.reproject("epsg:32721")
        # mds_repr.to_netcdf(sentinel_2_utm)
        # test
        _assert_xarrays_equal(mds_repr, mdc, precision=4)


def test_geographic_reproject__missing_nodata():
    sentinel_2_geographic = os.path.join(
        TEST_INPUT_DATA_DIR, "sentinel_2_L1C_geographic.nc"
    )
    sentinel_2_utm = os.path.join(
        TEST_COMPARE_DATA_DIR, "sentinel_2_L1C_utm__auto_nodata.nc"
    )
    with xarray.open_dataset(sentinel_2_geographic) as mda, xarray.open_dataset(
        sentinel_2_utm
    ) as mdc:
        mda.red.attrs.pop("nodata")
        mda.nir.attrs.pop("nodata")
        mds_repr = mda.rio.reproject("epsg:32721")
        # mds_repr.to_netcdf(sentinel_2_utm)
        # test
        _mod_attr(mdc, "_FillValue", val=65535)
        _assert_xarrays_equal(mds_repr, mdc, precision=4)


def test_geographic_resample_integer():
    pytest.importorskip("scipy")
    sentinel_2_geographic = os.path.join(
        TEST_INPUT_DATA_DIR, "sentinel_2_L1C_geographic.nc"
    )
    sentinel_2_interp = os.path.join(
        TEST_COMPARE_DATA_DIR, "sentinel_2_L1C_interpolate_na.nc"
    )
    with xarray.open_dataset(sentinel_2_geographic) as mda, xarray.open_dataset(
        sentinel_2_interp
    ) as mdc:
        mds_interp = mda.rio.interpolate_na()
        # mds_interp.to_netcdf(sentinel_2_interp)
        # test
        _assert_xarrays_equal(mds_interp, mdc)


@pytest.mark.parametrize(
    "open_method",
    [
        xarray.open_dataarray,
        partial(rioxarray.open_rasterio, masked=True),
        partial(rioxarray.open_rasterio, masked=True, chunks=True),
        partial(
            rioxarray.open_rasterio, masked=True, chunks=True, lock=threading.Lock()
        ),
        open_rasterio_engine,
    ],
)
@pytest.mark.parametrize(
    "windowed, recalc_transform",
    [
        (True, True),
        (False, False),
    ],
)
@pytest.mark.parametrize(
    "write_lock, compute",
    [
        (None, False),
        (threading.Lock(), False),
        (threading.Lock(), True),
    ],
)
def test_to_raster(
    open_method, windowed, recalc_transform, write_lock, compute, tmpdir
):
    tmp_raster = tmpdir.join("modis_raster.tif")
    test_tags = {"test": "1"}
    with open_method(os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")) as mda:
        if isinstance(mda, xarray.Dataset):
            mda = mda.__xarray_dataarray_variable__
        delayed = mda.rio.to_raster(
            str(tmp_raster),
            windowed=windowed,
            recalc_transform=recalc_transform,
            tags=test_tags,
            lock=write_lock,
            compute=compute,
        )
        xds = mda.copy().squeeze()
        xds_attrs = {
            key: str(value)
            for key, value in mda.attrs.items()
            if key
            not in (
                "add_offset",
                "crs",
                "is_tiled",
                "nodata",
                "res",
                "scale_factor",
                "transform",
                "grid_mapping",
            )
        }

    if write_lock is None or not isinstance(xds.data, da.Array) or compute:
        assert delayed is None
    else:
        assert isinstance(delayed, Delayed)
        delayed.compute()

    with rasterio.open(str(tmp_raster)) as rds:
        assert rds.count == 1
        assert rds.crs == xds.rio.crs
        assert_array_equal(rds.transform, xds.rio.transform())
        assert_array_equal(rds.nodata, xds.rio.encoded_nodata)
        assert_array_equal(rds.read(1), xds.fillna(xds.rio.encoded_nodata).values)
        assert rds.count == 1
        assert rds.tags() == {"AREA_OR_POINT": "Area", **test_tags, **xds_attrs}
        assert rds.dtypes == ("int16",)


@pytest.mark.parametrize(
    "open_method",
    [
        partial(xarray.open_dataset, decode_coords="all"),
        partial(rioxarray.open_rasterio, masked=True),
        partial(rioxarray.open_rasterio, masked=True, chunks=True),
        partial(
            rioxarray.open_rasterio, masked=True, chunks=True, lock=threading.Lock()
        ),
        open_rasterio_engine,
    ],
)
@pytest.mark.parametrize("windowed", [True, False])
@pytest.mark.parametrize(
    "write_lock, compute",
    [
        (None, False),
        (threading.Lock(), False),
        (threading.Lock(), True),
    ],
)
def test_to_raster_3d(open_method, windowed, write_lock, compute, tmpdir):
    tmp_raster = tmpdir.join("planet_3d_raster.tif")
    with open_method(os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc")) as mda:
        xds = mda.green.fillna(mda.green.rio.encoded_nodata)
        xds.rio._nodata = mda.green.rio.encoded_nodata
        delayed = xds.rio.to_raster(
            str(tmp_raster), windowed=windowed, lock=write_lock, compute=compute
        )
        xds_attrs = {
            key: str(value)
            for key, value in xds.attrs.items()
            if key
            not in ("add_offset", "nodata", "scale_factor", "transform", "grid_mapping")
        }

    if write_lock is None or not isinstance(xds.data, da.Array) or compute:
        assert delayed is None
    else:
        assert isinstance(delayed, Delayed)
        delayed.compute()

    with rasterio.open(str(tmp_raster)) as rds:
        assert rds.crs == xds.rio.crs
        assert_array_equal(rds.transform, xds.rio.transform())
        assert_array_equal(rds.nodata, xds.rio.nodata)
        assert_array_equal(rds.read(), xds.values)
        assert rds.tags() == {"AREA_OR_POINT": "Area", **xds_attrs}
        assert rds.descriptions == ("green", "green")

    # test roundtrip
    with rioxarray.open_rasterio(str(tmp_raster)) as rds:
        assert rds.attrs["long_name"] == "green"
        assert numpy.isnan(rds.rio.nodata)


def test_to_raster__custom_description(tmpdir):
    tmp_raster = tmpdir.join("planet_3d_raster.tif")
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc")
    ) as mda:
        xds = mda.green.fillna(mda.green.rio.encoded_nodata)
        xds.attrs["long_name"] = ("one", "two")
        xds.rio.to_raster(str(tmp_raster))
        xds_attrs = {
            key: str(value)
            for key, value in xds.attrs.items()
            if key not in ("nodata", "long_name", "grid_mapping")
        }

    with rasterio.open(str(tmp_raster)) as rds:
        assert rds.tags() == {"AREA_OR_POINT": "Area", **xds_attrs}
        assert rds.descriptions == ("one", "two")

    # test roundtrip
    with rioxarray.open_rasterio(str(tmp_raster)) as rds:
        assert rds.attrs["long_name"] == ("one", "two")
        assert rds.rio.nodata == 0.0


@pytest.mark.parametrize("chunks", [True, None])
def test_to_raster__scale_factor_and_add_offset(chunks, tmpdir):
    tmp_raster = tmpdir.join("air_temp_offset.tif")

    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc"), chunks=chunks
    ) as rds:
        assert rds.air_temperature.scale_factor == 0.1
        assert rds.air_temperature.add_offset == 220.0
        rds.air_temperature.rio.to_raster(str(tmp_raster))

    with rasterio.open(str(tmp_raster)) as rds:
        assert rds.scales == (0.1,)
        assert rds.offsets == (220.0,)

    # test roundtrip
    with rioxarray.open_rasterio(str(tmp_raster)) as rds:
        assert rds.scale_factor == 0.1
        assert rds.add_offset == 220.0
        assert rds.rio.nodata == 32767.0


@pytest.mark.parametrize("chunks", [True, None])
def test_to_raster__offsets_and_scales(chunks, tmpdir):
    tmp_raster = tmpdir.join("air_temp_offset.tif")

    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc"), chunks=chunks
    ) as rds:
        attrs = dict(rds.air_temperature.attrs)
        attrs["scales"] = [0.1]
        attrs["offsets"] = [220.0]
        attrs.pop("scale_factor")
        attrs.pop("add_offset")
        rds.air_temperature.attrs = attrs
        assert rds.air_temperature.scales == [0.1]
        assert rds.air_temperature.offsets == [220.0]
        rds.air_temperature.rio.to_raster(str(tmp_raster))

    with rasterio.open(str(tmp_raster)) as rds:
        assert rds.scales == (0.1,)
        assert rds.offsets == (220.0,)

    # test roundtrip
    with rioxarray.open_rasterio(str(tmp_raster)) as rds:
        assert rds.scale_factor == 0.1
        assert rds.add_offset == 220.0
        assert rds.rio.nodata == 32767.0


def test_to_raster__custom_description__wrong(tmpdir):
    tmp_raster = tmpdir.join("planet_3d_raster.tif")
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc")
    ) as mda:
        xds = mda.green.fillna(mda.green.rio.encoded_nodata)
        xds.attrs["long_name"] = ("one", "two", "three")
        with pytest.raises(RioXarrayError):
            xds.rio.to_raster(str(tmp_raster))


@pytest.mark.xfail(reason="Precision issues with windowed writing on python 3.6")
@pytest.mark.parametrize("windowed", [True, False])
def test_to_raster__preserve_profile__none_nodata(windowed, tmpdir):
    tmp_raster = tmpdir.join("output_profile.tif")
    input_raster = tmpdir.join("input_profile.tif")

    transform = Affine.from_gdal(0, 512, 0, 0, 0, 512)
    with rasterio.open(
        str(input_raster),
        "w",
        driver="GTiff",
        height=512,
        width=512,
        count=1,
        crs="epsg:4326",
        transform=transform,
        dtype=rasterio.float32,
        tiled=True,
        tilexsize=256,
        tileysize=256,
    ) as rds:
        rds.write(numpy.empty((1, 512, 512), dtype=numpy.float32))

    with rioxarray.open_rasterio(str(input_raster)) as mda:
        mda.rio.to_raster(str(tmp_raster), windowed=windowed)

    with rasterio.open(str(tmp_raster)) as rds, rasterio.open(str(input_raster)) as rdc:
        assert rds.count == rdc.count
        assert rds.crs == rdc.crs
        assert_array_equal(rds.transform, rdc.transform)
        assert_array_equal(rds.nodata, rdc.nodata)
        assert_almost_equal(rds.read(), rdc.read())
        assert rds.profile == rdc.profile
        assert rds.nodata is None


def test_to_raster__dataset(tmpdir):
    tmp_raster = tmpdir.join("planet_3d_raster.tif")
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc")
    ) as mda:
        mda.isel(time=0).rio.to_raster(str(tmp_raster))

    with rioxarray.open_rasterio(str(tmp_raster)) as rdscompare:
        assert rdscompare.scale_factor == 1.0
        assert rdscompare.add_offset == 0.0
        assert rdscompare.long_name == ("blue", "green")
        assert rdscompare.rio.crs == mda.rio.crs
        assert numpy.isnan(rdscompare.rio.nodata)


@pytest.mark.parametrize("chunks", [True, None])
def test_to_raster__dataset__mask_and_scale(chunks, tmpdir):
    output_raster = tmpdir.join("tmmx_20190121.tif")
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc"), chunks=chunks
    ) as rds:
        rds.isel(band=0).rio.to_raster(str(output_raster))

    with rioxarray.open_rasterio(str(output_raster)) as rdscompare:
        assert rdscompare.scale_factor == 0.1
        assert rdscompare.add_offset == 220.0
        assert rdscompare.long_name == "air_temperature"
        assert rdscompare.rio.crs == rds.rio.crs
        assert rdscompare.rio.nodata == rds.air_temperature.rio.nodata


def test_to_raster__dataset__different_crs(tmpdir):
    tmp_raster = tmpdir.join("planet_3d_raster.tif")
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc")
    ) as mda:
        rds = mda.isel(time=0)
        attrs = rds.green.attrs
        attrs["crs"] = "EPSG:4326"
        attrs.pop("grid_mapping")
        rds.green.attrs = attrs
        attrs = rds.blue.attrs
        attrs["crs"] = "EPSG:32722"
        attrs.pop("grid_mapping")
        rds.blue.attrs = attrs
        rds = rds.drop_vars("spatial_ref")
        with pytest.raises(
            RioXarrayError, match="CRS in DataArrays differ in the Dataset"
        ):
            rds.rio.to_raster(str(tmp_raster))


def test_to_raster__dataset__different_nodata(tmpdir):
    tmp_raster = tmpdir.join("planet_3d_raster.tif")
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"), mask_and_scale=False
    ) as mda:
        rds = mda.isel(time=0)
        rds.green.rio.write_nodata(1234, inplace=True)
        rds.blue.rio.write_nodata(2345, inplace=True)
        with pytest.raises(
            RioXarrayError,
            match="All nodata values must be the same when exporting to raster.",
        ):
            rds.rio.to_raster(str(tmp_raster))


@pytest.mark.parametrize("windowed", [True, False])
def test_to_raster__different_dtype(tmp_path, windowed):
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    test_da.values[1, 1] = -1.1
    test_nd = test_da.rio.write_nodata(-1.1)
    test_nd.rio.write_transform(
        Affine.from_gdal(425047, 3.0, 0.0, 4615780, 0.0, -3.0), inplace=True
    )
    test_nd.rio.write_crs("EPSG:4326", inplace=True)
    tmpfile = tmp_path / "dtype.tif"
    with pytest.warns(
        UserWarning,
        match=(
            r"The nodata value \(-1.1\) has been automatically changed to "
            r"\(255\) to match the dtype of the data."
        ),
    ):
        test_nd.rio.to_raster(tmpfile, dtype=numpy.uint8, windowed=windowed)
    xds = rioxarray.open_rasterio(tmpfile)
    assert str(xds.dtype) == "uint8"
    assert xds.attrs["_FillValue"] == 255
    assert xds.rio.nodata == 255
    assert xds.squeeze().values[1, 1] == 255


def test_missing_spatial_dimensions():
    test_ds = xarray.Dataset()
    with pytest.raises(MissingSpatialDimensionError):
        test_ds.rio.shape
    with pytest.raises(MissingSpatialDimensionError):
        test_ds.rio.width
    with pytest.raises(MissingSpatialDimensionError):
        test_ds.rio.height
    test_da = xarray.DataArray(1)
    with pytest.raises(MissingSpatialDimensionError):
        test_da.rio.shape
    with pytest.raises(MissingSpatialDimensionError):
        test_da.rio.width
    with pytest.raises(MissingSpatialDimensionError):
        test_da.rio.height


def test_set_spatial_dims():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("lat", "lon"),
        coords={"lat": numpy.arange(1, 6), "lon": numpy.arange(2, 7)},
    )
    test_da_copy = test_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    assert test_da_copy.rio.x_dim == "lon"
    assert test_da_copy.rio.y_dim == "lat"
    assert test_da_copy.rio.width == 5
    assert test_da_copy.rio.height == 5
    assert test_da_copy.rio.shape == (5, 5)
    with pytest.raises(MissingSpatialDimensionError):
        test_da.rio.shape
    with pytest.raises(MissingSpatialDimensionError):
        test_da.rio.width
    with pytest.raises(MissingSpatialDimensionError):
        test_da.rio.height

    test_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    assert test_da.rio.x_dim == "lon"
    assert test_da.rio.y_dim == "lat"
    assert test_da.rio.width == 5
    assert test_da.rio.height == 5
    assert test_da.rio.shape == (5, 5)


def test_set_spatial_dims__missing():
    test_ds = xarray.Dataset()
    with pytest.raises(MissingSpatialDimensionError):
        test_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("lat", "lon"),
        coords={"lat": numpy.arange(1, 6), "lon": numpy.arange(2, 7)},
    )
    with pytest.raises(MissingSpatialDimensionError):
        test_da.rio.set_spatial_dims(x_dim="long", y_dim="lati")


def test_crs_empty_dataset():
    assert xarray.Dataset().rio.crs is None


def test_crs_setter():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    assert test_da.rio.crs is None
    out_ds = test_da.rio.set_crs(4326)
    assert test_da.rio.crs.to_epsg() == 4326
    assert out_ds.rio.crs.to_epsg() == 4326
    test_ds = test_da.to_dataset(name="test")
    assert test_ds.rio.crs is None
    out_ds = test_ds.rio.set_crs(4326)
    assert test_ds.rio.crs.to_epsg() == 4326
    assert out_ds.rio.crs.to_epsg() == 4326


def test_crs_setter__copy():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    assert test_da.rio.crs is None
    out_ds = test_da.rio.set_crs(4326, inplace=False)
    assert test_da.rio.crs is None
    assert out_ds.rio.crs.to_epsg() == 4326
    test_ds = test_da.to_dataset(name="test")
    assert test_ds.rio.crs is None
    out_ds = test_ds.rio.set_crs(4326, inplace=False)
    assert test_ds.rio.crs is None
    assert out_ds.rio.crs.to_epsg() == 4326


def test_crs_writer__array__copy():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    assert test_da.rio.crs is None
    out_da = test_da.rio.write_crs(4326, grid_mapping_name="crs")
    assert "crs_wkt" in out_da.coords["crs"].attrs
    assert "spatial_ref" in out_da.coords["crs"].attrs
    out_da.rio._crs = None
    assert out_da.rio.crs.to_epsg() == 4326
    test_da.rio._crs = None
    assert test_da.rio.crs is None
    assert "crs" not in test_da.coords
    assert out_da.encoding["grid_mapping"] == "crs"


def test_crs_writer__array__inplace():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    assert test_da.rio.crs is None
    out_da = test_da.rio.write_crs(4326, inplace=True)
    assert "crs_wkt" in test_da.coords["spatial_ref"].attrs
    assert "spatial_ref" in test_da.coords["spatial_ref"].attrs
    assert out_da.coords["spatial_ref"] == test_da.coords["spatial_ref"]
    test_da.rio._crs = None
    assert test_da.rio.crs.to_epsg() == 4326
    assert test_da.encoding["grid_mapping"] == "spatial_ref"
    assert out_da.attrs == test_da.attrs
    out_da.rio._crs = None
    assert out_da.rio.crs.to_epsg() == 4326


def test_crs_writer__dataset__copy():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    test_da = test_da.to_dataset(name="test")
    assert test_da.rio.crs is None
    out_da = test_da.rio.write_crs(4326, grid_mapping_name="crs")
    assert "crs_wkt" in out_da.coords["crs"].attrs
    assert "spatial_ref" in out_da.coords["crs"].attrs
    out_da.test.rio._crs = None
    assert out_da.rio.crs.to_epsg() == 4326
    assert out_da.test.encoding["grid_mapping"] == "crs"
    # make sure input did not change the dataset
    test_da.test.rio._crs = None
    test_da.rio._crs = None
    assert test_da.rio.crs is None
    assert "crs" not in test_da.coords


def test_crs_writer__dataset__inplace():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    test_da = test_da.to_dataset(name="test")
    assert test_da.rio.crs is None
    out_da = test_da.rio.write_crs(4326, inplace=True)
    assert "crs_wkt" in test_da.coords["spatial_ref"].attrs
    assert "spatial_ref" in test_da.coords["spatial_ref"].attrs
    assert out_da.coords["spatial_ref"] == test_da.coords["spatial_ref"]
    out_da.test.rio._crs = None
    assert out_da.rio.crs.to_epsg() == 4326
    test_da.test.rio._crs = None
    test_da.rio._crs = None
    assert test_da.rio.crs.to_epsg() == 4326
    assert out_da.test.encoding["grid_mapping"] == "spatial_ref"
    assert out_da.test.attrs == test_da.test.attrs


def test_crs_writer__missing():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    with pytest.raises(MissingCRS):
        test_da.rio.write_crs()
    with pytest.raises(MissingCRS):
        test_da.to_dataset(name="test").rio.write_crs()


def test_crs__dataset__different_crs(tmpdir):
    green = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
        attrs={"crs": "EPSG:4326"},
    )
    blue = green.copy(deep=True)
    blue.attrs = {"crs": "EPSG:32722"}

    with pytest.raises(RioXarrayError, match="CRS in DataArrays differ in the Dataset"):
        xarray.Dataset({"green": green, "blue": blue}).rio.crs


def test_clip_missing_crs():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    with pytest.raises(MissingCRS):
        test_da.rio.clip({}, 4326)


def test_reproject_missing_crs():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    with pytest.raises(MissingCRS):
        test_da.rio.reproject(4326)


def test_reproject_resolution_and_shape_transform():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
        attrs={"crs": "epsg:3857"},
    )
    affine = Affine.from_gdal(0, 0.005, 0, 0, 0, 0.005)
    with pytest.raises(RioXarrayError):
        test_da.rio.reproject(4326, resolution=1, shape=(1, 1))
    with pytest.raises(RioXarrayError):
        test_da.rio.reproject(4326, resolution=1, transform=affine)
    with pytest.raises(RioXarrayError):
        test_da.rio.reproject(4326, resolution=1, shape=(1, 1), transform=affine)


def test_reproject_transform_missing_shape():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
        attrs={"crs": "epsg:3857"},
    )
    affine = Affine.from_gdal(0, 0.005, 0, 0, 0, 0.005)
    reprojected = test_da.rio.reproject(4326, transform=affine)
    assert reprojected.rio.shape == (5, 5)
    assert reprojected.rio.transform() == affine


class CustomCRS:
    @property
    def wkt(self):
        return CRS.from_epsg(4326).to_wkt()

    def __str__(self):
        return self.wkt


def test_crs_get_custom():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
        attrs={"crs": CustomCRS()},
    )
    assert test_da.rio.crs.to_epsg() == 4326
    test_ds = xarray.Dataset({"test": test_da})
    assert test_ds.rio.crs.to_epsg() == 4326


def test_get_crs_dataset():
    test_ds = xarray.Dataset()
    test_ds = test_ds.rio.write_crs(4326)
    assert test_ds.encoding["grid_mapping"] == "spatial_ref"
    assert test_ds.rio.crs.to_epsg() == 4326


def test_crs_is_removed():
    test_ds = xarray.Dataset(attrs=dict(crs="+init=epsg:4326"))
    test_ds = test_ds.rio.write_crs(4326)

    assert "crs" not in test_ds.attrs


def test_write_crs_cf():
    test_da = xarray.DataArray(1)
    test_da = test_da.rio.write_crs(4326)
    assert test_da.encoding["grid_mapping"] == "spatial_ref"
    assert test_da.rio.crs.to_epsg() == 4326
    assert "spatial_ref" in test_da.spatial_ref.attrs
    assert "crs_wkt" in test_da.spatial_ref.attrs
    assert test_da.spatial_ref.attrs["grid_mapping_name"] == "latitude_longitude"


def test_write_crs_cf__disable_grid_mapping():
    test_da = xarray.DataArray(1)
    with rioxarray.set_options(export_grid_mapping=False):
        test_da = test_da.rio.write_crs(4326)
    assert test_da.encoding["grid_mapping"] == "spatial_ref"
    assert test_da.rio.crs.to_epsg() == 4326
    assert "spatial_ref" in test_da.spatial_ref.attrs
    assert "crs_wkt" in test_da.spatial_ref.attrs
    assert "grid_mapping_name" not in test_da.spatial_ref.attrs


def test_write_crs__missing_geospatial_dims():
    test_da = xarray.DataArray(
        [1],
        name="data",
        dims=("time",),
        coords={"time": [1]},
    )
    assert test_da.copy().rio.write_crs(3857).rio.crs.to_epsg() == 3857
    assert test_da.to_dataset().rio.write_crs(3857).rio.crs.to_epsg() == 3857


def test_read_crs_cf():
    test_da = xarray.DataArray(1)
    test_da = test_da.rio.write_crs(4326)
    assert test_da.encoding["grid_mapping"] == "spatial_ref"
    attrs = test_da.spatial_ref.attrs
    attrs.pop("spatial_ref")
    attrs.pop("crs_wkt")
    assert test_da.rio.crs.is_geographic


def test_get_crs_dataset__nonstandard_grid_mapping():
    test_ds = xarray.Dataset()
    test_ds = test_ds.rio.write_crs(4326, grid_mapping_name="frank")
    assert test_ds.encoding["grid_mapping"] == "frank"
    assert test_ds.rio.crs.to_epsg() == 4326


def test_get_crs_dataset__missing_grid_mapping_default():
    test_ds = xarray.open_dataset(os.path.join(TEST_INPUT_DATA_DIR, "test_find_crs.nc"))
    assert test_ds.rio.crs.to_epsg() == 32614


def test_nodata_setter():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    assert test_da.rio.nodata is None
    out_ds = test_da.rio.set_nodata(-1)
    assert out_ds.rio.nodata == -1


def test_nodata_setter__copy():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    assert test_da.rio.nodata is None
    out_ds = test_da.rio.set_nodata(-1, inplace=False)
    assert test_da.rio.nodata is None
    assert out_ds.rio.nodata == -1


def test_write_nodata__array__copy():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    assert test_da.rio.nodata is None
    out_da = test_da.rio.write_nodata(-1)
    assert test_da.rio.nodata is None
    assert out_da.attrs["_FillValue"] == -1
    assert out_da.rio.nodata == -1
    out_da.rio._nodata = None
    assert out_da.rio.nodata == -1
    test_da.rio._nodata = None
    assert test_da.rio.nodata is None
    assert "_FillValue" not in test_da.attrs


def test_write_nodata__array__inplace():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    assert test_da.rio.nodata is None
    out_da = test_da.rio.write_nodata(-1, inplace=True)
    assert "_FillValue" in test_da.attrs
    assert out_da.attrs["_FillValue"] == test_da.attrs["_FillValue"]
    test_da.rio._nodata = None
    assert test_da.rio.nodata == -1
    assert out_da.attrs == test_da.attrs
    out_da.rio._nodata = None
    assert out_da.rio.nodata == -1


def test_write_nodata__missing():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    test_da.rio.write_nodata(None)
    assert not test_da.attrs
    assert not test_da.encoding


def test_write_nodata__remove():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    test_nd = test_da.rio.write_nodata(-1)
    assert not test_da.attrs
    assert test_nd.attrs["_FillValue"] == -1
    test_nd.rio.write_nodata(None, inplace=True)
    assert not test_nd.attrs


def test_write_nodata__encoded():
    test_da = xarray.DataArray(
        numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    test_nd = test_da.rio.write_nodata(-1, inplace=True)
    assert test_nd.attrs["_FillValue"] == -1
    test_nd = test_da.rio.write_nodata(-1, encoded=True, inplace=True)
    assert not test_nd.attrs
    assert test_nd.encoding["_FillValue"] == -1
    assert test_nd.rio.encoded_nodata == -1
    assert numpy.isnan(test_nd.rio.nodata)
    test_nd.rio.write_nodata(None, encoded=True, inplace=True)
    assert not test_nd.attrs
    assert not test_nd.encoding
    assert test_nd.rio.encoded_nodata is None
    assert test_nd.rio.nodata is None


@pytest.mark.parametrize("nodata", [-1.1, "-1.1"])
def test_write_nodata__different_dtype(nodata):
    test_da = xarray.DataArray(
        numpy.zeros((5, 5), dtype=int),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    with pytest.warns(
        UserWarning,
        match=(
            r"The nodata value \(-1.1\) has been automatically changed to "
            r"\(-1\) to match the dtype of the data."
        ),
    ):
        test_nd = test_da.rio.write_nodata(nodata)
    assert not test_da.attrs
    assert test_nd.attrs["_FillValue"] == -1
    assert test_nd.rio.nodata == -1


@pytest.mark.parametrize("nodata", [-1.1, "-1.1"])
def test_nodata_reader__different_dtype(nodata):
    test_da = xarray.DataArray(
        numpy.zeros((5, 5), dtype=numpy.uint8),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
        attrs={"_FillValue": nodata},
    )
    assert test_da.attrs["_FillValue"] == nodata
    with pytest.warns(
        UserWarning,
        match=(
            r"The nodata value \(-1.1\) has been automatically changed to "
            r"\(255\) to match the dtype of the data."
        ),
    ):
        assert test_da.rio.nodata == 255


def test_isel_window():
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")
    ) as mda:
        assert (
            mda.rio.isel_window(Window.from_slices(slice(9, 12), slice(10, 12)))
            == mda.isel(x=slice(10, 12), y=slice(9, 12))
        ).all()


def test_isel_window_wpad():
    with rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc")
    ) as mda:
        w1 = Window.from_slices(
            slice(-5, 10), slice(-5, 10), height=15, width=15, boundless=True
        )
        wb1 = rasterio.windows.bounds(w1, mda.rio.transform(recalc=True))
        res1 = mda.rio.isel_window(w1, pad=True)
        exp1 = mda.rio.pad_box(*wb1).isel(x=slice(0, 15), y=slice(0, 15))
        assert (res1 == exp1).all()
        w2 = Window.from_slices(
            slice(195, 210), slice(195, 210), height=15, width=15, boundless=True
        )
        wb2 = rasterio.windows.bounds(w2, mda.rio.transform(recalc=True))
        res2 = mda.rio.isel_window(w2, pad=True)
        exp2 = mda.rio.pad_box(*wb2).isel(x=slice(195, 210), y=slice(195, 210))
        assert (res2 == exp2).all()


def test_write_pyproj_crs_dataset():
    test_ds = xarray.Dataset()
    test_ds = test_ds.rio.write_crs(pCRS(4326))
    assert test_ds.encoding["grid_mapping"] == "spatial_ref"
    assert test_ds.rio.crs.to_epsg() == 4326


def test_nonstandard_dims_clip__dataset():
    with open(os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim_geom.json")) as ndj:
        geom = json.load(ndj)
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        clipped = xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.clip([geom])
        assert clipped.rio.width == 6
        assert clipped.rio.height == 5


def test_nonstandard_dims_clip__array():
    with open(os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim_geom.json")) as ndj:
        geom = json.load(ndj)
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        clipped = xds.analysed_sst.rio.set_spatial_dims(
            x_dim="lon", y_dim="lat"
        ).rio.clip([geom])
        assert clipped.rio.width == 6
        assert clipped.rio.height == 5


def test_nonstandard_dims_clip_box__dataset():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        clipped = xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.clip_box(
            -70.51367964678269,
            -23.780199727400767,
            -70.44589567737998,
            -23.71896017814794,
        )
        assert clipped.rio.width == 7
        assert clipped.rio.height == 7


def test_nonstandard_dims_clip_box_array():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        clipped = xds.analysed_sst.rio.set_spatial_dims(
            x_dim="lon", y_dim="lat"
        ).rio.clip_box(
            -70.51367964678269,
            -23.780199727400767,
            -70.44589567737998,
            -23.71896017814794,
        )
        assert clipped.rio.width == 7
        assert clipped.rio.height == 7


def test_nonstandard_dims_slice_xy_array():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        clipped = xds.analysed_sst.rio.set_spatial_dims(
            x_dim="lon", y_dim="lat"
        ).rio.slice_xy(
            -70.51367964678269,
            -23.780199727400767,
            -70.44589567737998,
            -23.71896017814794,
        )
        assert clipped.rio.width == 7
        assert clipped.rio.height == 7


def test_nonstandard_dims_reproject__dataset():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        xds = xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        reprojected = xds.rio.reproject("epsg:3857")
        assert reprojected.rio.width == 11
        assert reprojected.rio.height == 11
        assert reprojected.rio.crs.to_epsg() == 3857


def test_nonstandard_dims_reproject__array():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        reprojected = xds.analysed_sst.rio.set_spatial_dims(
            x_dim="lon", y_dim="lat"
        ).rio.reproject("epsg:3857")
        assert reprojected.rio.width == 11
        assert reprojected.rio.height == 11
        assert reprojected.rio.crs.to_epsg() == 3857


def test_nonstandard_dims_interpolate_na__dataset():
    pytest.importorskip("scipy")
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        reprojected = xds.rio.set_spatial_dims(
            x_dim="lon", y_dim="lat"
        ).rio.interpolate_na()
        assert reprojected.rio.width == 11
        assert reprojected.rio.height == 11


def test_nonstandard_dims_interpolate_na__array():
    pytest.importorskip("scipy")
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        reprojected = xds.analysed_sst.rio.set_spatial_dims(
            x_dim="lon", y_dim="lat"
        ).rio.interpolate_na()
        assert reprojected.rio.width == 11
        assert reprojected.rio.height == 11


def test_nonstandard_dims_write_nodata__array():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        reprojected = xds.analysed_sst.rio.set_spatial_dims(
            x_dim="lon", y_dim="lat"
        ).rio.write_nodata(-999)
        assert reprojected.rio.width == 11
        assert reprojected.rio.height == 11
        assert reprojected.rio.nodata == -999


def test_nonstandard_dims_isel_window():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        reprojected = xds.rio.set_spatial_dims(
            x_dim="lon", y_dim="lat"
        ).rio.isel_window(Window.from_slices(slice(4), slice(5)))
        assert reprojected.rio.width == 5
        assert reprojected.rio.height == 4


def test_nonstandard_dims_error_msg():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        with pytest.raises(MissingSpatialDimensionError, match="x dimension not found"):
            xds.rio.width
        with pytest.raises(
            MissingSpatialDimensionError, match="Data variable: analysed_sst"
        ):
            xds.analysed_sst.rio.width


def test_nonstandard_dims_find_dims():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        assert xds.rio.x_dim == "lon"
        assert xds.rio.y_dim == "lat"


def test_nonstandard_dims_find_dims__standard_name():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {"standard_name": "longitude"}
        xds.coords["lat"].attrs = {"standard_name": "latitude"}
        assert xds.rio.x_dim == "lon"
        assert xds.rio.y_dim == "lat"


def test_nonstandard_dims_find_dims__standard_name__projected():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {"standard_name": "projection_x_coordinate"}
        xds.coords["lat"].attrs = {"standard_name": "projection_y_coordinate"}
        assert xds.rio.x_dim == "lon"
        assert xds.rio.y_dim == "lat"


def test_nonstandard_dims_find_dims__axis():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds.coords["lon"].attrs = {"axis": "X"}
        xds.coords["lat"].attrs = {"axis": "Y"}
        assert xds.rio.x_dim == "lon"
        assert xds.rio.y_dim == "lat"


def test_nonstandard_dims_to_raster__dataset(tmp_path):
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc"), decode_coords="all"
    ) as xds:
        xds.attrs.pop("grid_mapping")
        xds.coords["lon"].attrs = {}
        xds.coords["lat"].attrs = {}
        xds.squeeze().rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.to_raster(
            tmp_path / "test.tif"
        )


def test_missing_crs_error_msg():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xds = xds.drop_vars("spatial_ref")
        xds.attrs.pop("grid_mapping")
        with pytest.raises(MissingCRS, match="Data variable: analysed_sst"):
            xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.reproject(
                "EPSG:4326"
            )
        with pytest.raises(MissingCRS, match="Data variable: analysed_sst"):
            xds.rio.set_spatial_dims(
                x_dim="lon", y_dim="lat"
            ).analysed_sst.rio.reproject("EPSG:4326")


def test_missing_transform_bounds():
    xds = rioxarray.open_rasterio(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif"),
        parse_coordinates=False,
    )
    xds.coords["spatial_ref"].attrs.pop("GeoTransform")
    with pytest.raises(DimensionMissingCoordinateError):
        xds.rio.bounds()


def test_missing_transform_resolution():
    xds = rioxarray.open_rasterio(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif"),
        parse_coordinates=False,
    )
    xds.coords["spatial_ref"].attrs.pop("GeoTransform")
    with pytest.raises(DimensionMissingCoordinateError):
        xds.rio.resolution()


def test_shape_order():
    rds = rioxarray.open_rasterio(os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc"))
    assert rds.air_temperature.rio.shape == (585, 1386)


def test_write_transform__from_read(tmp_path):
    xds = rioxarray.open_rasterio(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif"),
        parse_coordinates=False,
    )
    out_file = tmp_path / "test_geotransform.nc"
    xds.to_netcdf(out_file)
    xds2 = rioxarray.open_rasterio(out_file, parse_coordinates=False)
    assert_almost_equal(tuple(xds2.rio.transform()), tuple(xds.rio.transform()))
    assert xds.spatial_ref.GeoTransform == xds2.spatial_ref.GeoTransform


def test_write_transform():
    test_affine = Affine.from_gdal(
        *numpy.fromstring("425047 3.0 0.0 4615780 0.0 -3.0", sep=" ")
    )
    ds = xarray.Dataset()
    ds.rio.write_transform(test_affine, inplace=True)
    assert ds.spatial_ref.GeoTransform == "425047.0 3.0 0.0 4615780.0 0.0 -3.0"
    assert ds.rio._cached_transform() == test_affine
    assert ds.rio.transform() == test_affine
    assert ds.rio.grid_mapping == "spatial_ref"
    ds_transform = ds.rio.transform()
    assert all([isinstance(getattr(ds_transform, arg), float) for arg in "abcdef"])
    da = xarray.DataArray(1)
    da.rio.write_transform(test_affine, inplace=True)
    assert da.rio._cached_transform() == test_affine
    assert da.rio.transform() == test_affine
    assert da.spatial_ref.GeoTransform == "425047.0 3.0 0.0 4615780.0 0.0 -3.0"
    assert da.rio.grid_mapping == "spatial_ref"
    da_transform = da.rio.transform()
    assert all([isinstance(getattr(da_transform, arg), float) for arg in "abcdef"])


def test_write_read_transform__non_rectilinear():
    test_affine = Affine.from_gdal(305827, 14, 9, 5223236, 9, -14)
    ds = xarray.Dataset()
    ds.rio.write_transform(test_affine, inplace=True)
    assert ds.spatial_ref.GeoTransform == "305827.0 14.0 9.0 5223236.0 9.0 -14.0"
    assert ds.rio._cached_transform() == test_affine
    assert ds.rio.transform() == test_affine
    assert ds.rio.grid_mapping == "spatial_ref"
    da = xarray.DataArray(1)
    da.rio.write_transform(test_affine, inplace=True)
    assert ds.rio._cached_transform() == test_affine
    assert ds.rio.transform() == test_affine
    assert da.spatial_ref.GeoTransform == "305827.0 14.0 9.0 5223236.0 9.0 -14.0"
    assert da.rio.grid_mapping == "spatial_ref"


def test_write_read_transform__non_rectilinear__rotation__warning():
    test_affine = Affine.from_gdal(305827, 14, 9, 5223236, 9, -14)
    ds = xarray.Dataset()
    ds.rio.write_transform(test_affine, inplace=True)
    with pytest.warns(
        UserWarning, match=r"Transform that is non\-rectilinear or with rotation found"
    ):
        assert ds.rio.transform(recalc=True) == test_affine
    da = xarray.DataArray(1)
    da.rio.write_transform(test_affine, inplace=True)
    with pytest.warns(
        UserWarning, match=r"Transform that is non\-rectilinear or with rotation found"
    ):
        assert ds.rio.transform(recalc=True) == test_affine


def test_missing_transform():
    ds = xarray.Dataset()
    assert ds.rio.transform() == Affine.identity()
    da = xarray.DataArray(1)
    assert da.rio.transform() == Affine.identity()


def test_nonstandard_dims_write_coordinate_system__geographic():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xda = xds.analysed_sst.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        xda.coords[xda.rio.x_dim].attrs = {}
        xda.coords[xda.rio.y_dim].attrs = {}
        cs_array = xda.rio.write_crs("EPSG:4326").rio.write_coordinate_system()
        assert cs_array.coords[cs_array.rio.x_dim].attrs == {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        }
        assert cs_array.coords[cs_array.rio.y_dim].attrs == {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        }


def test_nonstandard_dims_write_coordinate_system__geographic__preserve_attrs():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        cs_array = (
            xds.analysed_sst.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            .rio.write_crs("EPSG:4326")
            .rio.write_coordinate_system()
        )
        assert cs_array.coords[cs_array.rio.x_dim].attrs == {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
            "comment": "geolocations inherited from the input data without correction",
            "valid_max": 180.0,
            "valid_min": -180.0,
        }
        assert cs_array.coords[cs_array.rio.y_dim].attrs == {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
            "comment": "geolocations inherited from the input data without correction",
            "valid_max": 90.0,
            "valid_min": -90.0,
        }


def test_nonstandard_dims_write_coordinate_system__projected_ft():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xda = xds.analysed_sst.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        xda.coords[xda.rio.x_dim].attrs = {}
        xda.coords[xda.rio.y_dim].attrs = {}
        cs_array = xda.rio.write_crs("EPSG:3418").rio.write_coordinate_system()
        assert cs_array.coords[cs_array.rio.x_dim].attrs == {
            "axis": "X",
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
            "units": "0.30480060960121924 metre",
        }
        assert cs_array.coords[cs_array.rio.y_dim].attrs == {
            "axis": "Y",
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
            "units": "0.30480060960121924 metre",
        }


def test_nonstandard_dims_write_coordinate_system__no_crs():
    with xarray.open_dataset(
        os.path.join(TEST_INPUT_DATA_DIR, "nonstandard_dim.nc")
    ) as xds:
        xda = xds.analysed_sst.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        xda.coords[xda.rio.x_dim].attrs = {}
        xda.coords[xda.rio.y_dim].attrs = {}
        xda.coords["spatial_ref"].attrs = {}
        cs_array = xda.rio.write_coordinate_system()
        assert cs_array.coords[cs_array.rio.x_dim].attrs == {
            "axis": "X",
        }
        assert cs_array.coords[cs_array.rio.y_dim].attrs == {
            "axis": "Y",
        }


@pytest.mark.parametrize(
    "open_func",
    [partial(xarray.open_dataset, mask_and_scale=False), rioxarray.open_rasterio],
)
def test_grid_mapping__pre_existing(open_func):
    with open_func(os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc")) as xdi:
        assert xdi.rio.grid_mapping == "crs"
        assert xdi.air_temperature.rio.grid_mapping == "crs"


@pytest.mark.parametrize(
    "open_func",
    [partial(xarray.open_dataset, mask_and_scale=False), rioxarray.open_rasterio],
)
def test_grid_mapping__change(open_func):
    with open_func(os.path.join(TEST_INPUT_DATA_DIR, "tmmx_20190121.nc")) as xdi:
        # part 1: check changing the data var grid mapping
        xdi["dummy"] = xdi.air_temperature.copy()
        xdi.dummy.rio.write_grid_mapping("different_crs", inplace=True)
        assert xdi.air_temperature.rio.grid_mapping == "crs"
        assert xdi.dummy.rio.grid_mapping == "different_crs"
        # part 2: ensure error raised when multiple exist
        with pytest.raises(RioXarrayError, match="Multiple grid mappings exist."):
            xdi.rio.grid_mapping
        # part 3: ensure that writing the grid mapping on the dataset fixes it
        xdi.rio.write_grid_mapping("final_crs", inplace=True)
        assert xdi.air_temperature.rio.grid_mapping == "final_crs"
        assert xdi.dummy.rio.grid_mapping == "final_crs"
        assert xdi.rio.grid_mapping == "final_crs"


def test_grid_mapping_default():
    xarray.Dataset().rio.grid_mapping == "spatial_ref"
    xarray.DataArray().rio.grid_mapping == "spatial_ref"


def test_estimate_utm_crs():
    xds = rioxarray.open_rasterio(
        os.path.join(TEST_INPUT_DATA_DIR, "cog.tif"),
    )
    if PYPROJ_LT_3:
        with pytest.raises(RuntimeError, match=r"pyproj 3\+ required"):
            xds.rio.estimate_utm_crs()
    else:
        assert xds.rio.estimate_utm_crs().to_epsg() in (32618, 32655)
        assert xds.rio.reproject("EPSG:4326").rio.estimate_utm_crs() == CRS.from_epsg(
            32618
        )
        assert xds.rio.estimate_utm_crs("WGS 72") in (32218, 32255)


@pytest.mark.skipif(PYPROJ_LT_3, reason="pyproj 3+ required")
def test_estimate_utm_crs__missing_crs():
    with pytest.raises(RuntimeError, match=r"crs must be set to estimate UTM CRS"):
        xarray.Dataset().rio.estimate_utm_crs("NAD83")


def test_estimate_utm_crs__out_of_bounds():
    xds = xarray.DataArray(
        numpy.zeros((2, 2)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": [-90.0, -90.0],
            "longitude": [-5.0, 5.0],
        },
    )
    xds.rio.write_crs("EPSG:4326", inplace=True)
    if PYPROJ_LT_3:
        with pytest.raises(RuntimeError, match=r"pyproj 3\+ required"):
            xds.rio.estimate_utm_crs()
    else:
        with pytest.raises(RuntimeError, match=r"Unable to determine UTM CRS"):
            xds.rio.estimate_utm_crs()


def test_interpolate_na_missing_nodata():
    pytest.importorskip("scipy")
    test_da = xarray.DataArray(
        name="missing",
        data=numpy.zeros((5, 5)),
        dims=("y", "x"),
        coords={"y": numpy.arange(1, 6), "x": numpy.arange(2, 7)},
    )
    match = (
        r"nodata not found\. Please set the nodata with "
        r"'rio\.write_nodata\(\)'\. Data variable: missing"
    )
    with pytest.raises(RioXarrayError, match=match):
        test_da.rio.interpolate_na()
    with pytest.raises(RioXarrayError, match=match):
        test_da.to_dataset().rio.interpolate_na()


def test_rio_write_gcps():
    """
    Test setting gcps in dataarray.
    """
    gdal_gcps, gcp_crs = _create_gdal_gcps()

    darr = xarray.DataArray(1)
    darr.rio.write_gcps(gdal_gcps, gcp_crs, inplace=True)

    _check_rio_gcps(darr, gdal_gcps, gcp_crs)


def _create_gdal_gcps():
    src_gcps = [
        GroundControlPoint(
            row=0, col=0, x=0.0, y=0.0, z=12.0, id="1", info="the first gcp"
        ),
        GroundControlPoint(
            row=0, col=800, x=10.0, y=0.0, z=1.0, id="2", info="the second gcp"
        ),
        GroundControlPoint(
            row=800, col=800, x=10.0, y=10.0, z=3.5, id="3", info="the third gcp"
        ),
        GroundControlPoint(
            row=800, col=0, x=0.0, y=10.0, z=5.5, id="4", info="the fourth gcp"
        ),
    ]
    crs = CRS.from_epsg(4326)
    gdal_gcps = (src_gcps, crs)
    return gdal_gcps


def _check_rio_gcps(darr, src_gcps, crs):
    assert "x" not in darr.coords
    assert "y" not in darr.coords
    assert darr.rio.crs == crs
    assert "gcps" in darr.spatial_ref.attrs
    gcps = darr.spatial_ref.attrs["gcps"]
    assert gcps["type"] == "FeatureCollection"
    assert len(gcps["features"]) == len(src_gcps)
    for feature, gcp in zip(gcps["features"], src_gcps):
        assert feature["type"] == "Feature"
        assert feature["properties"]["id"] == gcp.id
        # info seems to be lost when rasterio writes?
        # assert feature["properties"]["info"] == gcp.info
        assert feature["properties"]["row"] == gcp.row
        assert feature["properties"]["col"] == gcp.col
        assert feature["geometry"]["type"] == "Point"
        assert feature["geometry"]["coordinates"] == [gcp.x, gcp.y, gcp.z]


def test_rio_get_gcps():
    """
    Test setting gcps in dataarray.
    """
    gdal_gcps, gdal_crs = _create_gdal_gcps()

    darr = xarray.DataArray(1)
    darr.rio.write_gcps(gdal_gcps, gdal_crs, inplace=True)

    gcps = darr.rio.get_gcps()
    for gcp, gdal_gcp in zip(gcps, gdal_gcps):
        assert gcp.row == gdal_gcp.row
        assert gcp.col == gdal_gcp.col
        assert gcp.x == gdal_gcp.x
        assert gcp.y == gdal_gcp.y
        assert gcp.z == gdal_gcp.z
        assert gcp.id == gdal_gcp.id
        assert gcp.info == gdal_gcp.info


def test_reproject__gcps_file(tmp_path):
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

    rds = rioxarray.open_rasterio(tiffname)
    rds = rds.rio.reproject(
        crs,
    )
    assert rds.rio.height == 923
    assert rds.rio.width == 1027
    assert rds.rio.crs == crs
    assert rds.rio.transform().almost_equals(
        Affine(
            216.8587081056465,
            0.0,
            115698.25,
            0.0,
            -216.8587081056465,
            2818720.0,
        )
    )
