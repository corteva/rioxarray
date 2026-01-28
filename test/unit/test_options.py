import pytest

from rioxarray import set_options
from rioxarray._options import CONVENTION, EXPORT_GRID_MAPPING, get_option
from rioxarray.enum import Convention


def test_set_options__contextmanager():
    assert get_option(EXPORT_GRID_MAPPING)
    with set_options(**{EXPORT_GRID_MAPPING: False}):
        assert not get_option(EXPORT_GRID_MAPPING)
    assert get_option(EXPORT_GRID_MAPPING)


def test_set_options__global():
    assert get_option(EXPORT_GRID_MAPPING)
    try:
        set_options(export_grid_mapping=False)
        assert not get_option(EXPORT_GRID_MAPPING)
    finally:
        set_options(export_grid_mapping=True)
    assert get_option(EXPORT_GRID_MAPPING)


def test_set_options__invalid_argument():
    with pytest.raises(
        ValueError,
        match="argument name does_not_exist is not in the set of valid options",
    ):
        with set_options(does_not_exist=False):
            pass


def test_set_options__invalid_value():
    with pytest.raises(
        ValueError,
        match="option 'export_grid_mapping' gave an invalid value: 12345.",
    ):
        with set_options(export_grid_mapping=12345):
            pass


def test_set_options__convention_default():
    """Test that convention defaults to None."""
    assert get_option(CONVENTION) is None


def test_set_options__convention_cf():
    """Test setting convention to CF."""
    assert get_option(CONVENTION) is None
    with set_options(convention=Convention.CF):
        assert get_option(CONVENTION) is Convention.CF
    assert get_option(CONVENTION) is None


def test_set_options__convention_zarr():
    """Test setting convention to Zarr."""
    assert get_option(CONVENTION) is None
    with set_options(convention=Convention.Zarr):
        assert get_option(CONVENTION) is Convention.Zarr
    assert get_option(CONVENTION) is None


def test_set_options__convention_none():
    """Test setting convention back to None."""
    with set_options(convention=Convention.CF):
        assert get_option(CONVENTION) is Convention.CF
        with set_options(convention=None):
            assert get_option(CONVENTION) is None
        assert get_option(CONVENTION) is Convention.CF


def test_set_options__convention_invalid():
    """Test that invalid convention values raise error."""
    with pytest.raises(
        ValueError,
        match="option 'convention' gave an invalid value: 'invalid'.",
    ):
        with set_options(convention="invalid"):
            pass
