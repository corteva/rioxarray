"""Enums for rioxarray."""
from enum import Enum


class Convention(Enum):
    """
    Supported geospatial metadata conventions.

    rioxarray supports conventions for storing geospatial metadata.
    Currently supported:

    - CF: Climate and Forecasts convention using grid_mapping coordinates

    The convention can be set globally using set_options() or per-method
    using the convention parameter.

    Examples
    --------
    Set global convention:

    >>> import rioxarray
    >>> from rioxarray.enum import Convention
    >>> rioxarray.set_options(convention=Convention.CF)

    Use specific convention for a method:

    >>> from rioxarray.enum import Convention
    >>> data.rio.write_crs("EPSG:4326", convention=Convention.CF)

    See Also
    --------
    rioxarray.set_options : Set global options including convention

    References
    ----------
    .. [1] CF Conventions: https://github.com/cf-convention/cf-conventions
    """

    #: Climate and Forecasts convention (default)
    #: https://github.com/cf-convention/cf-conventions
    CF = "CF"
