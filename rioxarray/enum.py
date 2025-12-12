"""Enums for rioxarray."""
from enum import Enum


class Convention(Enum):
    """Supported geospatial metadata conventions."""

    #: https://github.com/cf-convention/cf-conventions
    CF = "CF"

    #: https://github.com/zarr-conventions/spatial
    Zarr = "Zarr"
