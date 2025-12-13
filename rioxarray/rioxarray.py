"""
This module is an extension for xarray to provide rasterio capabilities
to xarray datasets/dataarrays.
"""

# pylint: disable=too-many-lines
import json
import math
import warnings
from collections.abc import Hashable, Iterable
from typing import Any, Literal, Optional, Union

import numpy
import rasterio.warp
import rasterio.windows
import xarray
from affine import Affine
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

from rioxarray._convention import cf, zarr
from rioxarray._options import CONVENTION, get_option
from rioxarray.crs import crs_from_user_input
from rioxarray.enum import Convention
from rioxarray.exceptions import (
    DimensionError,
    DimensionMissingCoordinateError,
    InvalidDimensionOrder,
    MissingCRS,
    MissingSpatialDimensionError,
    NoDataInBounds,
    OneDimensionalRaster,
    RioXarrayError,
    TooManyDimensions,
)

DEFAULT_GRID_MAP = "spatial_ref"


def _affine_has_rotation(affine: Affine) -> bool:
    """
    Determine if the affine has rotation.

    Parameters
    ----------
    affine: :obj:`affine.Affine`
        The affine of the grid.

    Returns
    -------
    bool
    """
    return affine.b == affine.d != 0


def _resolution(affine: Affine) -> tuple[float, float]:
    """
    Determine if the resolution of the affine.
    If it has rotation, the sign of the resolution is lost.

    Based on: https://github.com/mapbox/rasterio/blob/6185a4e4ad72b5669066d2d5004bf46d94a6d298/rasterio/_base.pyx#L943-L951

    Parameters
    ----------
    affine: :obj:`affine.Affine`
        The affine of the grid.


    Returns
    --------
    x_resolution: float
        The X resolution of the affine.
    y_resolution: float
        The Y resolution of the affine.
    """
    if not _affine_has_rotation(affine):
        return affine.a, affine.e
    return (
        math.sqrt(affine.a**2 + affine.d**2),
        math.sqrt(affine.b**2 + affine.e**2),
    )


def affine_to_coords(
    affine: Affine, width: int, height: int, *, x_dim: str = "x", y_dim: str = "y"
) -> dict[str, numpy.ndarray]:
    """Generate 1d pixel centered coordinates from affine.

    Based on code from the xarray rasterio backend.

    Parameters
    ----------
    affine: :obj:`affine.Affine`
        The affine of the grid.
    width: int
        The width of the grid.
    height: int
        The height of the grid.
    x_dim: str, optional
        The name of the X dimension. Default is 'x'.
    y_dim: str, optional
        The name of the Y dimension. Default is 'y'.

    Returns
    -------
    dict: x and y coordinate arrays.

    """
    transform = affine * affine.translation(0.5, 0.5)
    if affine.is_rectilinear and not _affine_has_rotation(affine):
        x_coords, _ = transform * (numpy.arange(width), numpy.zeros(width))
        _, y_coords = transform * (numpy.zeros(height), numpy.arange(height))
    else:
        x_coords, y_coords = transform * numpy.meshgrid(
            numpy.arange(width),
            numpy.arange(height),
        )
    return {y_dim: y_coords, x_dim: x_coords}


def _generate_spatial_coords(
    *, affine: Affine, width: int, height: int
) -> dict[Hashable, Any]:
    """get spatial coords in new transform"""
    new_spatial_coords = affine_to_coords(affine, width, height)
    if new_spatial_coords["x"].ndim == 1:
        return {
            "x": xarray.IndexVariable("x", new_spatial_coords["x"]),
            "y": xarray.IndexVariable("y", new_spatial_coords["y"]),
        }
    return {
        "xc": (("y", "x"), new_spatial_coords["x"]),
        "yc": (("y", "x"), new_spatial_coords["y"]),
    }


def _get_nonspatial_coords(
    src_data_array: Union[xarray.DataArray, xarray.Dataset],
) -> dict[Hashable, Union[xarray.Variable, xarray.IndexVariable]]:
    coords: dict[Hashable, Union[xarray.Variable, xarray.IndexVariable]] = {}
    for coord in set(src_data_array.coords) - {
        src_data_array.rio.x_dim,
        src_data_array.rio.y_dim,
        DEFAULT_GRID_MAP,
    }:
        # skip 2D spatial coords
        if (
            src_data_array.rio.x_dim in src_data_array[coord].dims
            and src_data_array.rio.y_dim in src_data_array[coord].dims
        ):
            continue
        if src_data_array[coord].ndim == 1:
            coords[coord] = xarray.IndexVariable(
                src_data_array[coord].dims,
                src_data_array[coord].values,
                src_data_array[coord].attrs,
            )
        else:
            coords[coord] = xarray.Variable(
                src_data_array[coord].dims,
                src_data_array[coord].values,
                src_data_array[coord].attrs,
            )
    return coords


def _make_coords(
    *,
    src_data_array: Union[xarray.DataArray, xarray.Dataset],
    dst_affine: Affine,
    dst_width: int,
    dst_height: int,
    force_generate: bool = False,
) -> dict[Hashable, Any]:
    """Generate the coordinates of the new projected `xarray.DataArray`"""
    coords = _get_nonspatial_coords(src_data_array)
    if (
        force_generate
        or (
            src_data_array.rio.x_dim in src_data_array.coords
            and src_data_array.rio.y_dim in src_data_array.coords
        )
        or ("xc" in src_data_array.coords and "yc" in src_data_array.coords)
    ):
        new_coords = _generate_spatial_coords(
            affine=dst_affine, width=dst_width, height=dst_height
        )
        new_coords.update(coords)
        return new_coords
    return coords


def _get_data_var_message(obj: Union[xarray.DataArray, xarray.Dataset]) -> str:
    """
    Get message for named data variables.
    """
    try:
        return f" Data variable: {obj.name}" if obj.name else ""
    except AttributeError:
        return ""


def _get_spatial_dims(
    obj: Union[xarray.Dataset, xarray.DataArray], *, var: Union[Any, Hashable]
) -> tuple[str, str]:
    """
    Retrieve the spatial dimensions of the dataset
    """
    try:
        return obj[var].rio.x_dim, obj[var].rio.y_dim
    except MissingSpatialDimensionError as err:
        try:
            obj[var].rio.set_spatial_dims(
                x_dim=obj.rio.x_dim, y_dim=obj.rio.y_dim, inplace=True
            )
            return obj.rio.x_dim, obj.rio.y_dim
        except MissingSpatialDimensionError:
            raise err from None


def _has_spatial_dims(
    obj: Union[xarray.Dataset, xarray.DataArray], *, var: Union[Any, Hashable]
) -> bool:
    """
    Check to see if the variable in the Dataset has spatial dimensions
    """
    try:
        # pylint: disable=pointless-statement
        _get_spatial_dims(obj, var=var)
    except MissingSpatialDimensionError:
        return False
    return True


def _order_bounds(
    *,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    resolution_x: float,
    resolution_y: float,
) -> tuple[float, float, float, float]:
    """
    Make sure that the bounds are in the correct order
    """
    if resolution_y < 0:
        top = maxy
        bottom = miny
    else:
        top = miny
        bottom = maxy
    if resolution_x < 0:
        left = maxx
        right = minx
    else:
        left = minx
        right = maxx

    return left, bottom, right, top


class XRasterBase:
    """This is the base class for the GIS extensions for xarray"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, xarray_obj: Union[xarray.DataArray, xarray.Dataset]):
        self._obj: Union[xarray.DataArray, xarray.Dataset] = xarray_obj

        self._x_dim: Optional[Hashable] = None
        self._y_dim: Optional[Hashable] = None

        # Read spatial dimensions using the global convention setting
        convention = get_option(CONVENTION)

        if convention == Convention.Zarr:
            spatial_dims = zarr.read_spatial_dimensions(self._obj)
            if spatial_dims is not None:
                self._y_dim, self._x_dim = spatial_dims
        elif convention == Convention.CF or convention is None:
            # Use CF convention logic for dimension detection
            # Also use this as fallback when convention is None
            if "x" in self._obj.dims and "y" in self._obj.dims:
                self._x_dim = "x"
                self._y_dim = "y"
            elif "longitude" in self._obj.dims and "latitude" in self._obj.dims:
                self._x_dim = "longitude"
                self._y_dim = "latitude"
            else:
                # look for coordinates with CF attributes
                for coord in self._obj.coords:
                    # make sure to only look in 1D coordinates
                    # that has the same dimension name as the coordinate
                    if self._obj.coords[coord].dims != (coord,):
                        continue
                    if (
                        self._obj.coords[coord].attrs.get("axis", "").upper() == "X"
                    ) or (
                        self._obj.coords[coord].attrs.get("standard_name", "").lower()
                        in ("longitude", "projection_x_coordinate")
                    ):
                        self._x_dim = coord
                    elif (
                        self._obj.coords[coord].attrs.get("axis", "").upper() == "Y"
                    ) or (
                        self._obj.coords[coord].attrs.get("standard_name", "").lower()
                        in ("latitude", "projection_y_coordinate")
                    ):
                        self._y_dim = coord

            # If no dimensions found by CF when convention is None and Zarr conventions are declared, try Zarr as fallback
            if (
                (self._x_dim is None or self._y_dim is None)
                and convention is None
                and zarr.has_convention_declared(self._obj.attrs, "spatial:")
            ):
                spatial_dims = zarr.read_spatial_dimensions(self._obj)
                if spatial_dims is not None:
                    self._y_dim, self._x_dim = spatial_dims

        # properties
        self._count: Optional[int] = None
        self._height: Optional[int] = None
        self._width: Optional[int] = None
        self._crs: Union[rasterio.crs.CRS, None, Literal[False]] = None
        self._gcps: Optional[list[GroundControlPoint]] = None

    @property
    def crs(self) -> Optional[rasterio.crs.CRS]:
        """:obj:`rasterio.crs.CRS`:
        Retrieve projection from :obj:`xarray.Dataset` | :obj:`xarray.DataArray`
        """
        if self._crs is not None:
            return None if self._crs is False else self._crs

        # Read using global convention setting
        parsed_crs = None

        # Check global convention setting
        convention = get_option(CONVENTION)
        if convention == Convention.CF:
            parsed_crs = cf.read_crs(self._obj, self.grid_mapping)
        elif convention == Convention.Zarr:
            parsed_crs = zarr.read_crs(self._obj)
        elif convention is None:
            # Use CF as default when convention is None
            parsed_crs = cf.read_crs(self._obj, self.grid_mapping)
            # If CF didn't find anything and Zarr conventions are declared, try Zarr as fallback
            if parsed_crs is None and zarr.has_convention_declared(
                self._obj.attrs, "proj:"
            ):
                parsed_crs = zarr.read_crs(self._obj)

        if parsed_crs is not None:
            self._set_crs(parsed_crs, inplace=True)
            return self._crs

        # No CRS found
        self._crs = False
        return None

    def _get_obj(self, inplace: bool) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Get the object to modify.

        Parameters
        ----------
        inplace: bool
            If True, returns self.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`
        """
        if inplace:
            return self._obj
        obj_copy = self._obj.copy(deep=True)
        # preserve attribute information
        obj_copy.rio._x_dim = self._x_dim
        obj_copy.rio._y_dim = self._y_dim
        obj_copy.rio._width = self._width
        obj_copy.rio._height = self._height
        obj_copy.rio._crs = self._crs
        obj_copy.rio._gcps = self._gcps
        return obj_copy

    def set_crs(
        self, input_crs: Any, inplace: bool = True
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Set the CRS value for the Dataset/DataArray without modifying
        the dataset/data array.

        .. deprecated:: 0.15.8
            It is recommended to use `rio.write_crs()` instead. This
        method will likely be removed in a future release.

        Parameters
        ----------
        input_crs: object
            Anything accepted by `rasterio.crs.CRS.from_user_input`.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Dataset with crs attribute.
        """
        warnings.warn(
            "It is recommended to use 'rio.write_crs()' instead. 'rio.set_crs()' will likely"
            "be removed in a future release.",
            FutureWarning,
            stacklevel=2,
        )

        return self._set_crs(input_crs, inplace=inplace)

    def _set_crs(
        self, input_crs: Any, inplace: bool = True
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Set the CRS value for the Dataset/DataArray without modifying
        the dataset/data array.

        Parameters
        ----------
        input_crs: object
            Anything accepted by `rasterio.crs.CRS.from_user_input`.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        xarray.Dataset | xarray.DataArray
            Dataset with crs attribute.
        """
        crs = crs_from_user_input(input_crs)
        obj = self._get_obj(inplace=inplace)
        obj.rio._crs = crs
        return obj

    @property
    def grid_mapping(self) -> str:
        """
        str: The CF grid_mapping attribute. 'spatial_ref' is the default.
        """
        grid_mapping = self._obj.encoding.get(
            "grid_mapping", self._obj.attrs.get("grid_mapping")
        )
        if grid_mapping is not None:
            return grid_mapping
        grid_mapping = DEFAULT_GRID_MAP
        # search the dataset for the grid mapping name
        if hasattr(self._obj, "data_vars"):
            grid_mappings = set()
            for var in self._obj.data_vars:
                if not _has_spatial_dims(self._obj, var=var):
                    continue
                var_grid_mapping = self._obj[var].encoding.get(
                    "grid_mapping", self._obj[var].attrs.get("grid_mapping")
                )
                if var_grid_mapping is not None:
                    grid_mapping = var_grid_mapping
                    grid_mappings.add(grid_mapping)
            if len(grid_mappings) > 1:
                raise RioXarrayError("Multiple grid mappings exist.")
        return grid_mapping

    def write_grid_mapping(
        self, grid_mapping_name: str = DEFAULT_GRID_MAP, inplace: bool = False
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Write the CF grid_mapping attribute to the encoding.

        Parameters
        ----------
        grid_mapping_name: str, optional
            Name of the grid_mapping coordinate.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with CF compliant CRS information.
        """
        data_obj = self._get_obj(inplace=inplace)
        if hasattr(data_obj, "data_vars"):
            for var in data_obj.data_vars:
                try:
                    x_dim, y_dim = _get_spatial_dims(data_obj, var=var)
                except MissingSpatialDimensionError:
                    continue
                # remove grid_mapping from attributes if it exists
                # and update the grid_mapping in encoding
                new_attrs = dict(data_obj[var].attrs)
                new_attrs.pop("grid_mapping", None)
                data_obj[var].rio.update_encoding(
                    {"grid_mapping": grid_mapping_name}, inplace=True
                ).rio.set_attrs(new_attrs, inplace=True).rio.set_spatial_dims(
                    x_dim=x_dim, y_dim=y_dim, inplace=True
                )
        # remove grid_mapping from attributes if it exists
        # and update the grid_mapping in encoding
        new_attrs = dict(data_obj.attrs)
        new_attrs.pop("grid_mapping", None)
        return data_obj.rio.update_encoding(
            {"grid_mapping": grid_mapping_name}, inplace=True
        ).rio.set_attrs(new_attrs, inplace=True)

    def write_crs(
        self,
        input_crs: Optional[Any] = None,
        grid_mapping_name: Optional[str] = None,
        convention: Optional[Convention] = None,
        inplace: bool = False,
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Write the CRS to the dataset using the specified convention.

        .. warning:: When using CF convention, the grid_mapping attribute is written to the encoding.

        Parameters
        ----------
        input_crs: Any
            Anything accepted by `rasterio.crs.CRS.from_user_input`.
        grid_mapping_name: str, optional
            Name of the grid_mapping coordinate to store the CRS information in.
            Only used with CF convention. Default is the grid_mapping name of the dataset.
        convention: Convention, optional
            Convention to use for writing CRS. If None, uses the global default
            from set_options().
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with CRS information.

        Examples
        --------
        Write the CRS using the default convention:

        >>> raster.rio.write_crs("epsg:4326", inplace=True)

        Write the CRS using CF convention:

        >>> raster = raster.rio.write_crs("epsg:4326", convention=Convention.CF)

        Write the CRS using Zarr convention:

        >>> raster = raster.rio.write_crs("epsg:4326", convention=Convention.Zarr)
        """
        if input_crs is None and self.crs is None:
            raise MissingCRS(
                "CRS not found. Please set the CRS with 'rio.write_crs()'."
            )

        # Get the object to modify
        data_obj = self._get_obj(inplace=inplace)
        if input_crs is not None:
            data_obj.rio._set_crs(input_crs, inplace=True)

        # Determine which convention to use
        if convention is None:
            convention = get_option(CONVENTION) or Convention.CF

        if convention == Convention.CF:
            return cf.write_crs(
                data_obj,
                data_obj.rio.crs,
                grid_mapping_name or self.grid_mapping,
                inplace=True,
            )
        elif convention == Convention.Zarr:
            return zarr.write_crs(
                data_obj,
                data_obj.rio.crs,
                format="wkt2",  # Default to wkt2 format for performance
                inplace=True,
            )
        else:
            raise ValueError(f"Unsupported convention: {convention}")

    def estimate_utm_crs(self, datum_name: str = "WGS 84") -> rasterio.crs.CRS:
        """Returns the estimated UTM CRS based on the bounds of the dataset.

        .. versionadded:: 0.2

        .. note:: Requires pyproj 3+

        Parameters
        ----------
        datum_name : str, optional
            The name of the datum to use in the query. Default is WGS 84.

        Returns
        -------
        rasterio.crs.CRS
        """
        if self.crs is None:
            raise RuntimeError("crs must be set to estimate UTM CRS.")

        # ensure using geographic coordinates
        if self.crs.is_geographic:  # pylint: disable=no-member
            minx, miny, maxx, maxy = self.bounds(recalc=True)
        else:
            minx, miny, maxx, maxy = self.transform_bounds("EPSG:4326", recalc=True)

        x_center = numpy.mean([minx, maxx]).item()
        y_center = numpy.mean([miny, maxy]).item()

        utm_crs_list = query_utm_crs_info(
            datum_name=datum_name,
            area_of_interest=AreaOfInterest(
                west_lon_degree=x_center,
                south_lat_degree=y_center,
                east_lon_degree=x_center,
                north_lat_degree=y_center,
            ),
        )
        try:
            return CRS.from_epsg(utm_crs_list[0].code)
        except IndexError:
            raise RuntimeError("Unable to determine UTM CRS") from None

    def _cached_transform(self) -> Optional[Affine]:
        """
        Get the transform using the global convention setting.
        """
        # Read using the global convention setting
        convention = get_option(CONVENTION)

        if convention == Convention.Zarr:
            return zarr.read_transform(self._obj)
        elif convention == Convention.CF:
            return cf.read_transform(self._obj, self.grid_mapping)
        elif convention is None:
            # Use CF as default when convention is None
            transform = cf.read_transform(self._obj, self.grid_mapping)
            # If CF didn't find anything and Zarr conventions are declared, try Zarr
            if transform is None and zarr.has_convention_declared(
                self._obj.attrs, "spatial:"
            ):
                transform = zarr.read_transform(self._obj)
            return transform

        return None

    def write_transform(
        self,
        transform: Optional[Affine] = None,
        grid_mapping_name: Optional[str] = None,
        convention: Optional[Convention] = None,
        inplace: bool = False,
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        .. versionadded:: 0.0.30

        Write the transform to the dataset using the specified convention.

        Parameters
        ----------
        transform: affine.Affine, optional
            The transform of the dataset. If not provided, it will be calculated.
        grid_mapping_name: str, optional
            Name of the grid_mapping coordinate to store the transform information in.
            Only used with CF convention. Default is the grid_mapping name of the dataset.
        convention: Convention, optional
            Convention to use for writing transform. If None, uses the global default
            from set_options().
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with transform written.
        """
        transform = transform or self.transform(recalc=True)
        data_obj = self._get_obj(inplace=inplace)

        # Determine which convention to use
        if convention is None:
            convention = get_option(CONVENTION) or Convention.CF

        if convention == Convention.CF:
            return cf.write_transform(
                data_obj,
                transform,
                grid_mapping_name or self.grid_mapping,
                inplace=True,
            )
        elif convention == Convention.Zarr:
            return zarr.write_transform(data_obj, transform, inplace=True)
        else:
            raise ValueError(f"Unsupported convention: {convention}")

    def transform(self, recalc: bool = False) -> Affine:
        """
        Parameters
        ----------
        recalc: bool, optional
            If True, it will re-calculate the transform instead of using
            the cached transform.

        Returns
        -------
        :obj:`affine.Affine`:
            The affine of the :obj:`xarray.Dataset` | :obj:`xarray.DataArray`
        """
        transform = self._cached_transform()
        if transform and (
            not transform.is_rectilinear or _affine_has_rotation(transform)
        ):
            if recalc:
                warnings.warn(
                    "Transform that is non-rectilinear or with rotation found. "
                    "Unable to recalculate."
                )
            return transform

        try:
            src_left, _, _, src_top = self._unordered_bounds(recalc=recalc)
            src_resolution_x, src_resolution_y = self.resolution(recalc=recalc)
        except (DimensionMissingCoordinateError, DimensionError):
            return Affine.identity() if transform is None else transform
        return Affine.translation(src_left, src_top) * Affine.scale(
            src_resolution_x, src_resolution_y
        )

    def write_coordinate_system(
        self, inplace: bool = False
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Write the coordinate system CF metadata.

        .. versionadded:: 0.0.30

        Parameters
        ----------
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            The dataset with the CF coordinate system attributes added.
        """
        data_obj = self._get_obj(inplace=inplace)
        # add metadata to x,y coordinates
        is_projected = data_obj.rio.crs and data_obj.rio.crs.is_projected
        is_geographic = data_obj.rio.crs and data_obj.rio.crs.is_geographic
        x_coord_attrs = dict(data_obj.coords[self.x_dim].attrs)
        x_coord_attrs["axis"] = "X"
        y_coord_attrs = dict(data_obj.coords[self.y_dim].attrs)
        y_coord_attrs["axis"] = "Y"
        if is_projected:
            units = None
            if hasattr(data_obj.rio.crs, "linear_units_factor"):
                unit_factor = data_obj.rio.crs.linear_units_factor[-1]
                if unit_factor != 1:
                    units = f"{unit_factor} metre"
                else:
                    units = "metre"
            # X metadata
            x_coord_attrs["long_name"] = "x coordinate of projection"
            x_coord_attrs["standard_name"] = "projection_x_coordinate"
            if units:
                x_coord_attrs["units"] = units
            # Y metadata
            y_coord_attrs["long_name"] = "y coordinate of projection"
            y_coord_attrs["standard_name"] = "projection_y_coordinate"
            if units:
                y_coord_attrs["units"] = units
        elif is_geographic:
            # X metadata
            x_coord_attrs["long_name"] = "longitude"
            x_coord_attrs["standard_name"] = "longitude"
            x_coord_attrs["units"] = "degrees_east"
            # Y metadata
            y_coord_attrs["long_name"] = "latitude"
            y_coord_attrs["standard_name"] = "latitude"
            y_coord_attrs["units"] = "degrees_north"
        data_obj.coords[self.y_dim].attrs = y_coord_attrs
        data_obj.coords[self.x_dim].attrs = x_coord_attrs
        return data_obj

    def set_attrs(
        self, new_attrs: dict, inplace: bool = False
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Set the attributes of the dataset/dataarray and reset
        rioxarray properties to re-search for them.

        Parameters
        ----------
        new_attrs: dict
            A dictionary of new attributes.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with new attributes.
        """
        data_obj = self._get_obj(inplace=inplace)
        # set the attributes
        data_obj.attrs = new_attrs
        # reset rioxarray properties depending
        # on attributes to be generated
        data_obj.rio._nodata = None
        data_obj.rio._crs = None
        return data_obj

    def update_attrs(
        self, new_attrs: dict, inplace: bool = False
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Update the attributes of the dataset/dataarray and reset
        rioxarray properties to re-search for them.

        Parameters
        ----------
        new_attrs: dict
            A dictionary of new attributes to update with.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with updated attributes.
        """
        data_attrs = dict(self._obj.attrs)
        data_attrs.update(**new_attrs)
        return self.set_attrs(data_attrs, inplace=inplace)

    def set_encoding(
        self, new_encoding: dict, inplace: bool = False
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Set the encoding of the dataset/dataarray and reset
        rioxarray properties to re-search for them.

        .. versionadded:: 0.4

        Parameters
        ----------
        new_encoding: dict
            A dictionary for encoding.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with new attributes.
        """
        data_obj = self._get_obj(inplace=inplace)
        # set the attributes
        data_obj.encoding = new_encoding
        # reset rioxarray properties depending
        # on attributes to be generated
        data_obj.rio._nodata = None
        data_obj.rio._crs = None
        return data_obj

    def update_encoding(
        self, new_encoding: dict, inplace: bool = False
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Update the encoding of the dataset/dataarray and reset
        rioxarray properties to re-search for them.

        .. versionadded:: 0.4

        Parameters
        ----------
        new_encoding: dict
            A dictionary with encoding values to update with.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with updated attributes.
        """
        data_encoding = dict(self._obj.encoding)
        data_encoding.update(**new_encoding)
        return self.set_encoding(data_encoding, inplace=inplace)

    def set_spatial_dims(
        self, x_dim: str, y_dim: str, inplace: bool = True
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        This sets the spatial dimensions of the dataset.

        Parameters
        ----------
        x_dim: str
            The name of the x dimension.
        y_dim: str
            The name of the y dimension.
        inplace: bool, optional
            If True, it will modify the dataframe in place.
            Otherwise it will return a modified copy.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Dataset with spatial dimensions set.
        """

        data_obj = self._get_obj(inplace=inplace)
        if x_dim in data_obj.dims:
            data_obj.rio._x_dim = x_dim
        else:
            raise MissingSpatialDimensionError(
                f"x dimension ({x_dim}) not found.{_get_data_var_message(data_obj)}"
            )
        if y_dim in data_obj.dims:
            data_obj.rio._y_dim = y_dim
        else:
            raise MissingSpatialDimensionError(
                f"y dimension ({y_dim}) not found.{_get_data_var_message(data_obj)}"
            )
        return data_obj

    @property
    def x_dim(self) -> Hashable:
        """Hashable: The dimension for the X-axis."""
        if self._x_dim is not None:
            return self._x_dim
        raise MissingSpatialDimensionError(
            "x dimension not found. 'rio.set_spatial_dims()' or "
            "using 'rename()' to change the dimension name to 'x' can address this."
            f"{_get_data_var_message(self._obj)}"
        )

    @property
    def y_dim(self) -> Hashable:
        """Hashable: The dimension for the Y-axis."""
        if self._y_dim is not None:
            return self._y_dim
        raise MissingSpatialDimensionError(
            "y dimension not found. 'rio.set_spatial_dims()' or "
            "using 'rename()' to change the dimension name to 'y' can address this."
            f"{_get_data_var_message(self._obj)}"
        )

    @property
    def width(self) -> int:
        """int: Returns the width of the dataset (x dimension size)"""
        if self._width is not None:
            return self._width
        self._width = self._obj[self.x_dim].size
        return self._width

    @property
    def height(self) -> int:
        """int: Returns the height of the dataset (y dimension size)"""
        if self._height is not None:
            return self._height
        self._height = self._obj[self.y_dim].size
        return self._height

    @property
    def shape(self) -> tuple[int, int]:
        """tuple(int, int): Returns the shape (height, width)"""
        return (self.height, self.width)

    def _check_dimensions(self) -> Optional[str]:
        """
        This function validates that the dimensions 2D/3D and
        they are are in the proper order.

        Returns
        -------
        str or None: Name extra dimension.
        """
        extra_dims = tuple(set(list(self._obj.dims)) - {self.x_dim, self.y_dim})
        if len(extra_dims) > 1:
            raise TooManyDimensions(
                "Only 2D and 3D data arrays supported."
                f"{_get_data_var_message(self._obj)}"
            )
        if extra_dims and self._obj.dims != (extra_dims[0], self.y_dim, self.x_dim):
            dim_info: tuple = (extra_dims[0], self.y_dim, self.x_dim)
            raise InvalidDimensionOrder(
                f"Invalid dimension order. Expected order: {dim_info}. "
                f"You can use `DataArray.transpose{dim_info}`"
                " to reorder your dimensions."
                f"{_get_data_var_message(self._obj)}"
            )
        if not extra_dims and self._obj.dims != (self.y_dim, self.x_dim):
            dim_info = (self.y_dim, self.x_dim)
            raise InvalidDimensionOrder(
                f"Invalid dimension order. Expected order: {dim_info}. "
                f"You can use `DataArray.transpose{dim_info}`"
                " to reorder your dimensions."
                f"{_get_data_var_message(self._obj)}"
            )
        return str(extra_dims[0]) if extra_dims else None

    @property
    def count(self) -> int:
        """int: Returns the band count (z dimension size)"""
        if self._count is not None:
            return self._count
        extra_dim = self._check_dimensions()
        self._count = 1
        if extra_dim is not None:
            self._count = self._obj[extra_dim].size
        return self._count

    def _internal_bounds(self) -> tuple[float, float, float, float]:
        """Determine the internal bounds of the `xarray.DataArray`"""
        if self.x_dim not in self._obj.coords:
            raise DimensionMissingCoordinateError(f"{self.x_dim} missing coordinates.")
        if self.y_dim not in self._obj.coords:
            raise DimensionMissingCoordinateError(f"{self.y_dim} missing coordinates.")
        try:
            left = float(self._obj[self.x_dim][0])
            right = float(self._obj[self.x_dim][-1])
            top = float(self._obj[self.y_dim][0])
            bottom = float(self._obj[self.y_dim][-1])
        except IndexError:
            raise NoDataInBounds(
                "Unable to determine bounds from coordinates."
                f"{_get_data_var_message(self._obj)}"
            ) from None
        return left, bottom, right, top

    def resolution(self, recalc: bool = False) -> tuple[float, float]:
        """
        Determine if the resolution of the grid.
        If the transformation has rotation, the sign of the resolution is lost.

        Parameters
        ----------
        recalc: bool, optional
            Will force the resolution to be recalculated instead of using the
            transform attribute.

        Returns
        -------
        x_resolution, y_resolution: float
            The resolution of the `xarray.DataArray` | `xarray.Dataset`
        """
        transform = self._cached_transform()

        if (
            not recalc or self.width == 1 or self.height == 1
        ) and transform is not None:
            return _resolution(transform)

        # if the coordinates of the spatial dimensions are missing
        # use the cached transform resolution
        try:
            left, bottom, right, top = self._internal_bounds()
        except DimensionMissingCoordinateError:
            if transform is None:
                raise
            return _resolution(transform)

        if self.width == 1 or self.height == 1:
            raise OneDimensionalRaster(
                "Only 1 dimenional array found. Cannot calculate the resolution."
                f"{_get_data_var_message(self._obj)}"
            )

        resolution_x = (right - left) / (self.width - 1)
        resolution_y = (bottom - top) / (self.height - 1)
        return resolution_x, resolution_y

    def _unordered_bounds(
        self, recalc: bool = False
    ) -> tuple[float, float, float, float]:
        """
        Unordered bounds.

        Parameters
        ----------
        recalc: bool, optional
            Will force the bounds to be recalculated instead of using the
            transform attribute.

        Returns
        -------
        left, bottom, right, top: float
            Outermost coordinates of the `xarray.DataArray` | `xarray.Dataset`.
        """
        resolution_x, resolution_y = self.resolution(recalc=recalc)

        try:
            # attempt to get bounds from xarray coordinate values
            left, bottom, right, top = self._internal_bounds()
            left -= resolution_x / 2.0
            right += resolution_x / 2.0
            top -= resolution_y / 2.0
            bottom += resolution_y / 2.0
        except DimensionMissingCoordinateError as error:
            transform = self._cached_transform()
            if not transform:
                raise RioXarrayError("Transform not able to be determined.") from error
            left = transform.c
            top = transform.f
            right = left + resolution_x * self.width
            bottom = top + resolution_y * self.height

        return left, bottom, right, top

    def bounds(self, *, recalc: bool = False) -> tuple[float, float, float, float]:
        """
        Parameters
        ----------
        recalc: bool, optional
            Will force the bounds to be recalculated instead of using the
            transform attribute.

        Returns
        -------
        left, bottom, right, top: float
            Outermost coordinates of the `xarray.DataArray` | `xarray.Dataset`.
        """
        minx, miny, maxx, maxy = self._unordered_bounds(recalc=recalc)
        resolution_x, resolution_y = self.resolution(recalc=recalc)
        return _order_bounds(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
        )

    def isel_window(
        self, window: rasterio.windows.Window, *, pad: bool = False
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Use a rasterio.windows.Window to select a subset of the data.

        .. versionadded:: 0.6.0 pad

        .. warning:: Float indices are converted to integers.

        Parameters
        ----------
        window: :class:`rasterio.windows.Window`
            The window of the dataset to read.
        pad: bool, default=False
            Set to True to expand returned DataArray to dimensions of the window

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            The data in the window.
        """
        (row_start, row_stop), (col_start, col_stop) = window.toranges()
        row_start = 0 if row_start < 0 else math.floor(row_start)
        row_stop = 0 if row_stop < 0 else math.ceil(row_stop)
        col_start = 0 if col_start < 0 else math.floor(col_start)
        col_stop = 0 if col_stop < 0 else math.ceil(col_stop)
        row_slice = slice(int(row_start), int(row_stop))
        col_slice = slice(int(col_start), int(col_stop))
        array_subset = (
            self._obj.isel({self.y_dim: row_slice, self.x_dim: col_slice})
            .copy()  # this is to prevent sharing coordinates with the original dataset
            .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
            .rio.write_transform(
                transform=rasterio.windows.transform(
                    rasterio.windows.Window.from_slices(
                        rows=row_slice,
                        cols=col_slice,
                        width=self.width,
                        height=self.height,
                    ),
                    self.transform(recalc=True),
                ),
                inplace=True,
            )
        )
        if pad:
            return array_subset.rio.pad_box(
                *rasterio.windows.bounds(window, self.transform(recalc=True))
            )
        return array_subset

    def slice_xy(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """Slice the array by x,y bounds.

        Parameters
        ----------
        minx: float
            Minimum bound for x coordinate.
        miny: float
            Minimum bound for y coordinate.
        maxx: float
            Maximum bound for x coordinate.
        maxy: float
            Maximum bound for y coordinate.


        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            The data in the slice.
        """
        left, bottom, right, top = self._internal_bounds()
        if top > bottom:
            y_slice = slice(maxy, miny)
        else:
            y_slice = slice(miny, maxy)

        if left > right:
            x_slice = slice(maxx, minx)
        else:
            x_slice = slice(minx, maxx)

        subset = (
            self._obj.sel({self.x_dim: x_slice, self.y_dim: y_slice})
            .copy()  # this is to prevent sharing coordinates with the original dataset
            .rio.set_spatial_dims(x_dim=self.x_dim, y_dim=self.y_dim, inplace=True)
            .rio.write_transform(inplace=True)
        )
        return subset

    def transform_bounds(
        self, dst_crs: Any, *, densify_pts: int = 21, recalc: bool = False
    ) -> tuple[float, float, float, float]:
        """Transform bounds from src_crs to dst_crs.

        Optionally densifying the edges (to account for nonlinear transformations
        along these edges) and extracting the outermost bounds.

        Note: this does not account for the antimeridian.

        Parameters
        ----------
        dst_crs: str, :obj:`rasterio.crs.CRS`, or dict
            Target coordinate reference system.
        densify_pts: uint, optional
            Number of points to add to each edge to account for nonlinear
            edges produced by the transform process.  Large numbers will produce
            worse performance.  Default: 21 (gdal default).
        recalc: bool, optional
            Will force the bounds to be recalculated instead of using the transform
            attribute.

        Returns
        -------
        left, bottom, right, top: float
            Outermost coordinates in target coordinate reference system.
        """
        return rasterio.warp.transform_bounds(
            self.crs, dst_crs, *self.bounds(recalc=recalc), densify_pts=densify_pts
        )

    def write_gcps(
        self,
        gcps: Iterable[GroundControlPoint],
        gcp_crs: Any,
        *,
        grid_mapping_name: Optional[str] = None,
        inplace: bool = False,
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Write the GroundControlPoints to the dataset.

        https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html#ground-control-points

        Parameters
        ----------
        gcp: list of :obj:`rasterio.control.GroundControlPoint`
            The Ground Control Points to integrate to the dataset.
        gcp_crs: str, :obj:`rasterio.crs.CRS`, or dict
            Coordinate reference system for the GCPs.
        grid_mapping_name: str, optional
            Name of the grid_mapping coordinate to store the GCPs information in.
            Default is the grid_mapping name of the dataset.
        inplace: bool, optional
            If True, it will write to the existing dataset. Default is False.

        Returns
        -------
        :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
            Modified dataset with Ground Control Points written.
        """
        grid_mapping_name = (
            self.grid_mapping if grid_mapping_name is None else grid_mapping_name
        )
        data_obj = self._get_obj(inplace=True)

        if gcp_crs:
            data_obj = data_obj.rio.write_crs(
                gcp_crs, grid_mapping_name=grid_mapping_name, inplace=inplace
            )
        try:
            grid_map_attrs = data_obj.coords[grid_mapping_name].attrs.copy()
        except KeyError:
            data_obj.coords[grid_mapping_name] = xarray.Variable((), 0)
            grid_map_attrs = data_obj.coords[grid_mapping_name].attrs.copy()
        geojson_gcps = _convert_gcps_to_geojson(gcps)
        grid_map_attrs["gcps"] = json.dumps(geojson_gcps)
        data_obj.coords[grid_mapping_name].rio.set_attrs(grid_map_attrs, inplace=True)
        self._gcps = list(gcps)
        return data_obj

    def get_gcps(self) -> Optional[list[GroundControlPoint]]:
        """
        Get the GroundControlPoints from the dataset.

        https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html#ground-control-points

        Returns
        -------
        list of :obj:`rasterio.control.GroundControlPoint` or None
            The Ground Control Points from the dataset or None if not applicable
        """
        if self._gcps is not None:
            return self._gcps
        try:
            geojson_gcps = json.loads(self._obj.coords[self.grid_mapping].attrs["gcps"])
        except (KeyError, AttributeError):
            return None

        def _parse_gcp(gcp) -> GroundControlPoint:
            x, y, *z = gcp["geometry"]["coordinates"]
            z = z[0] if z else None
            return GroundControlPoint(
                x=x,
                y=y,
                z=z,
                row=gcp["properties"]["row"],
                col=gcp["properties"]["col"],
                id=gcp["properties"]["id"],
                info=gcp["properties"]["info"],
            )

        self._gcps = [_parse_gcp(gcp) for gcp in geojson_gcps["features"]]
        return self._gcps


def _convert_gcps_to_geojson(
    gcps: Iterable[GroundControlPoint],
) -> dict:
    """
    Convert GCPs to geojson.

    Parameters
    ----------
    gcps: The list of GroundControlPoint instances.

    Returns
    -------
    A FeatureCollection dict.
    """

    def _gcp_coordinates(gcp):
        if gcp.z is None:
            return [gcp.x, gcp.y]
        return [gcp.x, gcp.y, gcp.z]

    features = [
        {
            "type": "Feature",
            "properties": {
                "id": gcp.id,
                "info": gcp.info,
                "row": gcp.row,
                "col": gcp.col,
            },
            "geometry": {"type": "Point", "coordinates": _gcp_coordinates(gcp)},
        }
        for gcp in gcps
    ]
    return {"type": "FeatureCollection", "features": features}


def _convert_str_to_resampling(name: str) -> rasterio.warp.Resampling:
    """
    Convert from string to rasterio.warp.Resampling enum, raises ValueError on bad input.

    Parameters
    ----------
    name: str
        The string to convert.

    Returns
    -------
    :obj:`rasterio.warp.Resampling`
    """
    try:
        return getattr(rasterio.warp.Resampling, name.lower())
    except AttributeError:
        raise ValueError(f"Bad resampling parameter: {name}") from None
