"""
This file contains global options for rioxarray

Credits:

This file was adopted from: https://github.com/pydata/xarray # noqa
Source file: https://github.com/pydata/xarray/blob/2ab0666c1fcc493b1e0ebc7db14500c427f8804e/xarray/core/options.py  # noqa
"""
from typing import Any

EXPORT_GRID_MAPPING = "export_grid_mapping"
SKIP_MISSING_SPATIAL_DIMS = "skip_missing_spatial_dims"

OPTIONS = {
    EXPORT_GRID_MAPPING: True,
    SKIP_MISSING_SPATIAL_DIMS: False,
}
OPTION_NAMES = set(OPTIONS)

VALIDATORS = {
    EXPORT_GRID_MAPPING: lambda choice: isinstance(choice, bool),
}


def get_option(key: str) -> Any:
    """
    Get the global rioxarray option.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    key: str
        The name of the option.

    Returns
    -------
    Any: the value of the option.
    """
    return OPTIONS[key]


class set_options:  # pylint: disable=invalid-name
    """
    Set the global rioxarray option.

    .. versionadded:: 0.3.0
    .. versionadded:: 0.7.0 skip_missing_spatial_dims

    Parameters
    ----------
    export_grid_mapping: bool, default=True
        If True, this option will export the full Climate and Forecasts (CF)
        grid mapping attributes for the CRS. This is useful if you are exporting
        your file to netCDF using :meth:`xarray.Dataset.to_netcdf()`. When disabled,
        only the ``crs_wkt`` and ``spatial_ref`` attributes will be written and the
        program will be faster due to not needing to use
        :meth:`pyproj.CRS.to_cf() <pyproj.crs.CRS.to_cf>`.
    skip_missing_spatial_dims: bool, default=False
        If True, it will not perform spatial operations on variables
        within a :class:`xarray.Dataset` if the spatial dimensions
        are not found.


    Usage as a context manager::

        with rioxarray.set_options(export_grid_mapping=False):
            rds = rioxarray.open_rasterio(...)

    Usage for global settings::

        rioxarray.set_options(export_grid_mapping=False)

    """

    def __init__(self, **kwargs):
        self.old = OPTIONS.copy()
        for key, value in kwargs.items():
            if key not in OPTIONS:
                raise ValueError(
                    f"argument name {key} is not in the set of valid options "
                    f"{OPTION_NAMES}."
                )
            if key in VALIDATORS and not VALIDATORS[key](value):
                raise ValueError(f"option {key!r} gave an invalid value: {value!r}.")
            OPTIONS[key] = value

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_value, traceback):
        global OPTIONS  # pylint: disable=global-statement
        OPTIONS = self.old
