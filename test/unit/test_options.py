import pytest

from rioxarray import set_options
from rioxarray._options import EXPORT_GRID_MAPPING, get_option


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
