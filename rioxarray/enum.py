"""Enums for rioxarray."""
from enum import Enum


class Convention(Enum):
    """
    Supported geospatial metadata conventions.

    rioxarray supports two conventions for storing geospatial metadata:

    - CF: Climate and Forecasts convention using grid_mapping coordinates
    - Zarr: Zarr spatial and proj conventions using direct attributes

    The convention can be set globally using set_options() or per-method
    using the convention parameter.

    Examples
    --------
    Set global convention:

    >>> import rioxarray
    >>> rioxarray.set_options(convention=rioxarray.Convention.Zarr)

    Use specific convention for a method:

    >>> data.rio.write_crs("EPSG:4326", convention=rioxarray.Convention.CF)

    See Also
    --------
    rioxarray.set_options : Set global options including convention

    References
    ----------
    .. [1] CF Conventions: https://github.com/cf-convention/cf-conventions
    .. [2] Zarr Spatial Convention: https://github.com/zarr-conventions/spatial
    .. [3] Zarr Geo-Proj Convention: https://github.com/zarr-experimental/geo-proj
    """

    #: Climate and Forecasts convention (default)
    #: https://github.com/cf-convention/cf-conventions
    CF = "CF"

    #: Zarr spatial and proj conventions
    #: https://github.com/zarr-conventions/spatial
    #: https://github.com/zarr-experimental/geo-proj
    Zarr = "Zarr"
