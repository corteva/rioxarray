import numpy
import xarray

import rioxarray  # noqa: F401


def test_reproject_match__exact():
    """
    Based on: https://github.com/corteva/rioxarray/issues/298#issue-858511505
    """
    da1 = xarray.DataArray(
        numpy.random.rand(2, 3),
        [
            ("y", [-30.25, -30.3], {"id": 1, "random": "random"}),
            ("x", [149.8, 149.85, 149.9], {"id": 1, "random": "random"}),
        ],
    )
    da1.rio.write_crs(4326, inplace=True)
    da2 = xarray.DataArray(
        numpy.random.rand(4, 11),
        [
            (
                "longitude",
                [-30.25733604, -30.26550543, -30.27367483, -30.28184423],
                {"id": 2},
                {"rasterio_dtype": "int"},
            ),
            (
                "latitude",
                [
                    149.82193392,
                    149.83010332,
                    149.83827272,
                    149.84644211,
                    149.85461151,
                    149.86278091,
                    149.87095031,
                    149.87911971,
                    149.8872891,
                    149.8954585,
                    149.9036279,
                ],
                {"id": 2},
                {"rasterio_dtype": "int"},
            ),
        ],
    )
    da2.rio.write_crs(4326, inplace=True)
    resampled = da1.rio.reproject_match(da2)
    assert resampled.x.attrs == {
        "axis": "X",
        "long_name": "longitude",
        "standard_name": "longitude",
        "units": "degrees_east",
    }
    assert resampled.y.attrs == {
        "axis": "Y",
        "long_name": "latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
    }
    numpy.testing.assert_array_equal(resampled.x, da2.longitude)
    numpy.testing.assert_array_equal(resampled.y, da2.latitude)
