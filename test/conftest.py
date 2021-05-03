import os
from distutils.version import LooseVersion

import pyproj
import pytest
import rasterio
from numpy.testing import assert_almost_equal, assert_array_equal

import rioxarray
from rioxarray.raster_array import UNWANTED_RIO_ATTRS

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_INPUT_DATA_DIR = os.path.join(TEST_DATA_DIR, "input")
TEST_COMPARE_DATA_DIR = os.path.join(TEST_DATA_DIR, "compare")
PYPROJ_LT_3 = LooseVersion(pyproj.__version__) < LooseVersion("3")
RASTERIO_LT_122 = LooseVersion(rasterio.__version__) < LooseVersion("1.2.2")
RASTERIO_EQ_122 = LooseVersion(rasterio.__version__) == LooseVersion("1.2.2")


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
            and attr not in UNWANTED_RIO_ATTRS
            and attr != "creation_date"
            and attr != "grid_mapping"
            and attr != "coordinates"
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
                print("Error with variable {}".format(var))
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
        for unwanted_attr in UNWANTED_RIO_ATTRS:
            assert unwanted_attr not in input_xarray.attrs


def open_rasterio_engine(file_name_or_object, **kwargs):
    # FIXME: change to the next xarray version after release
    xr = pytest.importorskip("xarray", minversion="0.17.1.dev0")
    return xr.open_dataset(file_name_or_object, engine="rasterio", **kwargs)


@pytest.fixture(
    params=[
        rioxarray.open_rasterio,
        open_rasterio_engine,
    ]
)
def open_rasterio(request):
    return request.param
