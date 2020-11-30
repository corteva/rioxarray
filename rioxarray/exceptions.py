"""
This contains exceptions for rioxarray.
"""


class RioXarrayError(RuntimeError):
    """This is the base exception for errors in the rioxarray extension."""


class NoDataInBounds(RioXarrayError):
    """This is for when there are no data in the bounds for clipping a raster."""


class SingleVariableDataset(RioXarrayError):
    """This is for when you have a dataset with a single variable."""


class DimensionError(RioXarrayError):
    """This is raised when there are more dimensions than is supported by the method"""


class TooManyDimensions(DimensionError):
    """This is raised when there are more dimensions than is supported by the method"""


class InvalidDimensionOrder(DimensionError):
    """This is raised when there the dimensions are not ordered correctly."""


class OneDimensionalRaster(DimensionError):
    """This is an error when you have a 1 dimensional raster."""


class DimensionMissingCoordinateError(RioXarrayError):
    """This is raised when the dimension does not have the supporting coordinate."""


class MissingCRS(RioXarrayError):
    """Missing the CRS in the dataset."""
