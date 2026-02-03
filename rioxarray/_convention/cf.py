"""
CF (Climate and Forecasts) convention support for rioxarray.

This module provides functions for reading and writing geospatial metadata according to
the CF conventions: https://github.com/cf-convention/cf-conventions
"""
from typing import Optional, Tuple, Union

import numpy
import pyproj
import rasterio.crs
import xarray
from affine import Affine

from rioxarray._options import EXPORT_GRID_MAPPING, get_option
from rioxarray.crs import crs_from_user_input


class CFConvention:
    """CF convention class implementing ConventionProtocol."""

    @classmethod
    def read_crs(
        cls, obj: Union[xarray.Dataset, xarray.DataArray]
    ) -> Optional[rasterio.crs.CRS]:
        """
        Read CRS from CF conventions.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to read CRS from

        Returns
        -------
        rasterio.crs.CRS or None
            CRS object, or None if not found
        """
        try:
            grid_mapping_coord = obj.coords[obj.rio.grid_mapping]

            # Look in wkt attributes first for performance
            for crs_attr in ("spatial_ref", "crs_wkt"):
                try:
                    return crs_from_user_input(grid_mapping_coord.attrs[crs_attr])
                except KeyError:
                    pass

            # Look in grid_mapping CF attributes
            try:
                return pyproj.CRS.from_cf(grid_mapping_coord.attrs)
            except (KeyError, pyproj.exceptions.CRSError):
                pass
        except KeyError:
            # grid_mapping coordinate doesn't exist
            pass
        return None

    @classmethod
    def read_transform(
        cls, obj: Union[xarray.Dataset, xarray.DataArray]
    ) -> Optional[Affine]:
        """
        Read transform from CF conventions (GeoTransform attribute).

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to read transform from

        Returns
        -------
        affine.Affine or None
            Transform object, or None if not found
        """
        try:
            transform = numpy.fromstring(
                obj.coords[obj.rio.grid_mapping].attrs["GeoTransform"], sep=" "
            )
            # Calling .tolist() to assure the arguments are Python float and JSON serializable
            return Affine.from_gdal(*transform.tolist())
        except KeyError:
            pass

        return None

    @classmethod
    def read_spatial_dimensions(
        cls,
        obj: Union[xarray.Dataset, xarray.DataArray],
    ) -> Optional[Tuple[str, str]]:
        """
        Read spatial dimensions from CF conventions.

        This function detects spatial dimensions based on:
        1. Standard dimension names ('x'/'y', 'longitude'/'latitude')
        2. CF coordinate attributes ('axis', 'standard_name')

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to read spatial dimensions from

        Returns
        -------
        tuple of (y_dim, x_dim) or None
            Tuple of dimension names, or None if not found
        """
        x_dim = None
        y_dim = None

        # Check standard dimension names
        if "x" in obj.dims and "y" in obj.dims:
            return "y", "x"
        if "longitude" in obj.dims and "latitude" in obj.dims:
            return "latitude", "longitude"

        # Look for coordinates with CF attributes
        for coord in obj.coords:
            # Make sure to only look in 1D coordinates
            # that has the same dimension name as the coordinate
            if obj.coords[coord].dims != (coord,):
                continue
            if (obj.coords[coord].attrs.get("axis", "").upper() == "X") or (
                obj.coords[coord].attrs.get("standard_name", "").lower()
                in ("longitude", "projection_x_coordinate")
            ):
                x_dim = coord
            elif (obj.coords[coord].attrs.get("axis", "").upper() == "Y") or (
                obj.coords[coord].attrs.get("standard_name", "").lower()
                in ("latitude", "projection_y_coordinate")
            ):
                y_dim = coord

        if x_dim is not None and y_dim is not None:
            return str(y_dim), str(x_dim)
        return None

    @classmethod
    def write_crs(
        cls,
        obj: Union[xarray.Dataset, xarray.DataArray],
        crs: rasterio.crs.CRS,
        *,
        grid_mapping_name: Optional[str] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Write CRS using CF conventions.

        This also writes the grid_mapping attribute to encoding for CF compliance.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to write CRS to
        crs : rasterio.crs.CRS
            CRS to write
        grid_mapping_name : str
            Name of the grid_mapping coordinate
        **kwargs
            Additional convention-specific parameters

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Object with CRS written
        """
        # get original transform
        transform = obj.rio._cached_transform()
        # remove old grid mapping coordinate if exists
        grid_mapping_name = (
            obj.rio.grid_mapping if grid_mapping_name is None else grid_mapping_name
        )
        try:
            del obj.coords[grid_mapping_name]
        except KeyError:
            pass

        # add grid mapping coordinate
        obj.coords[grid_mapping_name] = xarray.Variable((), 0)
        grid_map_attrs = {}
        if get_option(EXPORT_GRID_MAPPING):
            try:
                grid_map_attrs = pyproj.CRS.from_user_input(crs).to_cf()
            except KeyError:
                pass
        # spatial_ref is for compatibility with GDAL
        crs_wkt = crs.to_wkt()
        grid_map_attrs["spatial_ref"] = crs_wkt
        grid_map_attrs["crs_wkt"] = crs_wkt
        if transform is not None:
            grid_map_attrs["GeoTransform"] = " ".join(
                [str(item) for item in transform.to_gdal()]
            )
        obj.coords[grid_mapping_name].rio.set_attrs(grid_map_attrs, inplace=True)

        return obj.rio.write_grid_mapping(
            grid_mapping_name=grid_mapping_name, inplace=True
        )

    @classmethod
    def write_transform(
        cls,
        obj: Union[xarray.Dataset, xarray.DataArray],
        *,
        transform: Affine,
        grid_mapping_name: Optional[str] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Union[xarray.Dataset, xarray.DataArray]:
        """
        Write transform using CF conventions (GeoTransform attribute).

        This also writes the grid_mapping attribute to encoding for CF compliance.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            Object to write transform to
        transform : affine.Affine
            Transform to write
        grid_mapping_name : Optional[str]
            Name of the grid_mapping coordinate
        **kwargs
            Additional convention-specific parameters

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Object with transform written
        """
        transform = transform or obj.rio.transform(recalc=True)
        grid_mapping_name = (
            obj.rio.grid_mapping if grid_mapping_name is None else grid_mapping_name
        )
        try:
            grid_map_attrs = obj.coords[grid_mapping_name].attrs.copy()
        except KeyError:
            obj.coords[grid_mapping_name] = xarray.Variable((), 0)
            grid_map_attrs = obj.coords[grid_mapping_name].attrs.copy()
        grid_map_attrs["GeoTransform"] = " ".join(
            [str(item) for item in transform.to_gdal()]
        )
        obj.coords[grid_mapping_name].rio.set_attrs(grid_map_attrs, inplace=True)
        return obj.rio.write_grid_mapping(
            grid_mapping_name=grid_mapping_name, inplace=True
        )
