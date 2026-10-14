import numpy
import pytest
import xarray
from affine import Affine
from numpy.testing import assert_array_equal

import rioxarray  # noqa: F401
from rioxarray._spatial_utils import affine_to_coords

# https://github.com/corteva/rioxarray/issues/937
pytestmark = pytest.mark.filterwarnings("error::PendingDeprecationWarning")


def test_affine_to_coords():
    coords = affine_to_coords(Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0), width=3, height=2)
    assert_array_equal(coords["x"], [0.5, 1.5, 2.5])
    assert_array_equal(coords["y"], [1.5, 0.5])


def test_affine_to_coords__rotation():
    coords = affine_to_coords(Affine(1.0, 1.0, 0.0, 0.0, -1.0, 2.0), width=2, height=2)
    assert_array_equal(coords["x"], [[1.0, 2.0], [2.0, 3.0]])
    assert_array_equal(coords["y"], [[1.5, 1.5], [0.5, 0.5]])


def test_transform__recalc():
    xda = xarray.DataArray(
        numpy.zeros((2, 3)),
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]},
    )
    assert xda.rio.transform(recalc=True) == Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
