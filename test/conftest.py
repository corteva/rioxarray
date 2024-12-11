import os
from functools import partial

import pytest
import rasterio
import xarray
from numpy.testing import assert_almost_equal, assert_array_equal
from packaging import version

import rioxarray
from rioxarray.raster_array import UNWANTED_RIO_ATTRS

xarray.set_options(warn_for_unclosed_files=True)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_INPUT_DATA_DIR = os.path.join(TEST_DATA_DIR, "input")
TEST_COMPARE_DATA_DIR = os.path.join(TEST_DATA_DIR, "compare")
RASTERIO_GE_14 = version.parse(rasterio.__version__) >= version.parse("1.4.0")
RASTERIO_GE_143 = version.parse(rasterio.__version__) >= version.parse("1.4.3")
GDAL_GE_36 = version.parse(rasterio.__gdal_version__) >= version.parse("3.6.0")
GDAL_GE_361 = version.parse(rasterio.__gdal_version__) >= version.parse("3.6.1")
GDAL_GE_364 = version.parse(rasterio.__gdal_version__) >= version.parse("3.6.4")


# xarray.testing.assert_equal(input_xarray, compare_xarray)
def _assert_attrs_equal(input_xr, compare_xr, decimal_precision):
    """check attrubutes that matter"""
    for attr in compare_xr.attrs:
        if attr == "transform":
            assert_almost_equal(
                tuple(input_xr.rio._cached_transform())[:6],
                compare_xr.attrs[attr][:6],
                decimal=decimal_precision,
            )
        elif (
            attr != "_FillValue"
            and attr
            not in UNWANTED_RIO_ATTRS
            + (
                "creation_date",
                "grid_mapping",
                "coordinates",
                "crs",
            )
            and "#" not in attr
        ):
            try:
                assert_almost_equal(
                    input_xr.attrs[attr],
                    compare_xr.attrs[attr],
                    decimal=decimal_precision,
                )
            except (TypeError, ValueError):
                assert input_xr.attrs[attr] == compare_xr.attrs[attr]


def _assert_xarrays_equal(
    input_xarray, compare_xarray, precision=7, skip_xy_check=False
):
    _assert_attrs_equal(input_xarray, compare_xarray, precision)
    if hasattr(input_xarray, "variables"):
        # check coordinates
        for coord in input_xarray.coords:
            if coord in "xy":
                if not skip_xy_check:
                    assert_almost_equal(
                        input_xarray[coord].values,
                        compare_xarray[coord].values,
                        decimal=precision,
                    )
            else:
                assert (
                    input_xarray[coord].values == compare_xarray[coord].values
                ).all()

        for var in input_xarray.rio.vars:
            try:
                _assert_xarrays_equal(
                    input_xarray[var], compare_xarray[var], precision=precision
                )
            except AssertionError:
                print(f"Error with variable {var}")
                raise
    else:
        try:
            assert_almost_equal(
                input_xarray.values, compare_xarray.values, decimal=precision
            )
        except AssertionError:
            where_diff = input_xarray.values != compare_xarray.values
            print(input_xarray.values[where_diff])
            print(compare_xarray.values[where_diff])
            raise
        _assert_attrs_equal(input_xarray, compare_xarray, precision)

        compare_fill_value = compare_xarray.attrs.get(
            "_FillValue", compare_xarray.encoding.get("_FillValue")
        )
        input_fill_value = input_xarray.attrs.get(
            "_FillValue", input_xarray.encoding.get("_FillValue")
        )
        assert_array_equal([input_fill_value], [compare_fill_value])
        assert input_xarray.rio.grid_mapping == compare_xarray.rio.grid_mapping
        for unwanted_attr in UNWANTED_RIO_ATTRS + ("crs", "transform"):
            assert unwanted_attr not in input_xarray.attrs


open_rasterio_engine = partial(xarray.open_dataset, engine="rasterio")


@pytest.fixture(
    params=[
        rioxarray.open_rasterio,
        open_rasterio_engine,
    ]
)
def open_rasterio(request):
    return request.param


def _ensure_dataset(rds):
    # https://github.com/OSGeo/gdal/issues/7695
    if GDAL_GE_364 and isinstance(rds, xarray.DataArray):
        rds = rds.to_dataset()
    return rds
