# -*- coding: utf-8 -*-
"""
This contains exceptions for rioxarray.
"""


class RioXarrayError(RuntimeError):
    """This is the base exception for errors in the rioxarray extension."""


class NoDataInBounds(RioXarrayError):
    """This is for when there are no data in the bounds for clipping a raster."""


class OneDimensionalRaster(RioXarrayError):
    """This is an error when you have a 1 dimensional raster."""


class SingleVariableDataset(RioXarrayError):
    """This is for when you have a dataset with a single variable."""


class TooManyDimensions(RioXarrayError):
    """This is raised when there are more dimensions than is supported by the method"""


class InvalidDimensionOrder(RioXarrayError):
    """This is raised when there the dimensions are not ordered correctly."""


class MissingCRS(RioXarrayError):
    """Missing the CRS in the dataset."""
