import os

from numpy.testing import assert_almost_equal, assert_array_equal

from rioxarray.rioxarray import UNWANTED_RIO_ATTRS

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_INPUT_DATA_DIR = os.path.join(TEST_DATA_DIR, "input")
TEST_COMPARE_DATA_DIR = os.path.join(TEST_DATA_DIR, "compare")


# xarray.testing.assert_equal(input_xarray, compare_xarray)
def _assert_attrs_equal(input_xr, compare_xr, decimal_precision):
    """check attrubutes that matter"""
    for attr in compare_xr.attrs:
        if attr == "transform":
            assert_almost_equal(
                input_xr.attrs[attr],
                compare_xr.attrs[attr][:6],
                decimal=decimal_precision,
            )
        elif (
            attr != "_FillValue"
            and attr not in UNWANTED_RIO_ATTRS
            and attr != "creation_date"
        ):
            try:
                assert_almost_equal(
                    input_xr.attrs[attr],
                    compare_xr.attrs[attr],
                    decimal=decimal_precision,
                )
            except (TypeError, ValueError):
                assert input_xr.attrs[attr] == compare_xr.attrs[attr]


def _assert_xarrays_equal(input_xarray, compare_xarray, precision=7):
    _assert_attrs_equal(input_xarray, compare_xarray, precision)
    if hasattr(input_xarray, "variables"):
        # check coordinates
        for coord in input_xarray.coords:
            if coord in "xy":
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
        assert "grid_mapping" in compare_xarray.attrs
        assert (
            input_xarray[input_xarray.attrs["grid_mapping"]]
            == compare_xarray[compare_xarray.attrs["grid_mapping"]]
        )
        for unwanted_attr in UNWANTED_RIO_ATTRS:
            assert unwanted_attr not in input_xarray.attrs
