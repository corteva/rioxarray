"""
Zarr spatial and proj convention support for rioxarray.

This module provides functions for reading and writing geospatial metadata according to:
- Zarr spatial convention: https://github.com/zarr-conventions/spatial
- Zarr geo-proj convention: https://github.com/zarr-experimental/geo-proj
"""
import json
from typing import Optional, Tuple, Union

import rasterio.crs
import xarray
from affine import Affine

from rioxarray.crs import crs_from_user_input

# Convention identifiers
PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj:",
    "description": "Coordinate reference system information for geospatial data",
}

SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial:",
    "description": "Spatial coordinate information",
}


def read_crs(
    obj: Union[xarray.Dataset, xarray.DataArray]
) -> Optional[rasterio.crs.CRS]:
    """
    Read CRS from Zarr proj: convention.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read CRS from

    Returns
    -------
    rasterio.crs.CRS or None
        CRS object, or None if not found
    """
    # Only interpret proj:* attributes if convention is declared
    if not has_convention_declared(obj.attrs, "proj:"):
        return None

    # Try array-level attributes first
    for proj_attr, parser in [
        ("proj:wkt2", parse_proj_wkt2),
        ("proj:code", parse_proj_code),
        ("proj:projjson", parse_proj_projjson),
    ]:
        try:
            proj_value = obj.attrs.get(proj_attr)
            if proj_value is not None:
                parsed_crs = parser(proj_value)
                if parsed_crs is not None:
                    return parsed_crs
        except (KeyError, Exception):
            pass

    # For Datasets, check group-level proj: convention (inheritance)
    if hasattr(obj, "data_vars") and has_convention_declared(obj.attrs, "proj:"):
        for proj_attr, parser in [
            ("proj:wkt2", parse_proj_wkt2),
            ("proj:code", parse_proj_code),
            ("proj:projjson", parse_proj_projjson),
        ]:
            try:
                proj_value = obj.attrs.get(proj_attr)
                if proj_value is not None:
                    parsed_crs = parser(proj_value)
                    if parsed_crs is not None:
                        return parsed_crs
            except (KeyError, Exception):
                pass

    return None


def read_transform(obj: Union[xarray.Dataset, xarray.DataArray]) -> Optional[Affine]:
    """
    Read transform from Zarr spatial: convention.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read transform from

    Returns
    -------
    affine.Affine or None
        Transform object, or None if not found
    """
    # Only interpret spatial:* attributes if convention is declared
    if not has_convention_declared(obj.attrs, "spatial:"):
        return None

    # Try array-level spatial:transform attribute
    try:
        spatial_transform = obj.attrs.get("spatial:transform")
        if spatial_transform is not None:
            return parse_spatial_transform(spatial_transform)
    except (KeyError, Exception):
        pass

    # For Datasets, check group-level spatial:transform
    if hasattr(obj, "data_vars"):
        try:
            spatial_transform = obj.attrs.get("spatial:transform")
            if spatial_transform is not None:
                return parse_spatial_transform(spatial_transform)
        except (KeyError, Exception):
            pass

    return None


def read_spatial_dimensions(
    obj: Union[xarray.Dataset, xarray.DataArray]
) -> Optional[Tuple[str, str]]:
    """
    Read spatial dimensions from Zarr spatial: convention.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to read spatial dimensions from

    Returns
    -------
    tuple of (y_dim, x_dim) or None
        Tuple of dimension names, or None if not found
    """
    # Only interpret spatial:* attributes if convention is declared
    if not has_convention_declared(obj.attrs, "spatial:"):
        return None

    try:
        spatial_dims = obj.attrs.get("spatial:dimensions")
        if spatial_dims is not None and len(spatial_dims) >= 2:
            # spatial:dimensions format is ["y", "x"] or similar
            y_dim_name, x_dim_name = spatial_dims[-2:]  # Take last two
            if y_dim_name in obj.dims and x_dim_name in obj.dims:
                return y_dim_name, x_dim_name
    except (KeyError, Exception):
        pass

    return None


def write_crs(
    obj: Union[xarray.Dataset, xarray.DataArray],
    input_crs: Optional[object] = None,
    format: str = "wkt2",
    inplace: bool = True,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write CRS using Zarr proj: convention.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write CRS to
    input_crs : object, optional
        CRS to write. Can be anything accepted by rasterio.crs.CRS.from_user_input
    format : {"code", "wkt2", "projjson", "all"}
        Which proj: format(s) to write
    inplace : bool, default True
        If True, modify object in place

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with CRS written
    """
    if input_crs is None:
        return obj

    crs = crs_from_user_input(input_crs)
    if crs is None:
        return obj

    obj_out = obj if inplace else obj.copy(deep=True)

    # Ensure proj: convention is declared
    obj_out.attrs = add_convention_declaration(obj_out.attrs, "proj:", inplace=True)

    if format in ("code", "all"):
        proj_code = format_proj_code(crs)
        if proj_code:
            obj_out.attrs["proj:code"] = proj_code

    if format in ("wkt2", "all"):
        obj_out.attrs["proj:wkt2"] = format_proj_wkt2(crs)

    if format in ("projjson", "all"):
        obj_out.attrs["proj:projjson"] = format_proj_projjson(crs)

    return obj_out


def write_transform(
    obj: Union[xarray.Dataset, xarray.DataArray],
    transform: Optional[Affine] = None,
    inplace: bool = True,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write transform using Zarr spatial: convention.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write transform to
    transform : affine.Affine, optional
        Transform to write
    inplace : bool, default True
        If True, modify object in place

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with transform written
    """
    if transform is None:
        return obj

    obj_out = obj if inplace else obj.copy(deep=True)

    # Ensure spatial: convention is declared
    obj_out.attrs = add_convention_declaration(obj_out.attrs, "spatial:", inplace=True)

    # Write spatial:transform as numeric array
    obj_out.attrs["spatial:transform"] = format_spatial_transform(transform)

    return obj_out


def write_spatial_metadata(
    obj: Union[xarray.Dataset, xarray.DataArray],
    y_dim: str,
    x_dim: str,
    transform: Optional[Affine] = None,
    include_bbox: bool = True,
    include_registration: bool = True,
    inplace: bool = True,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write complete Zarr spatial: metadata.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write metadata to
    y_dim, x_dim : str
        Names of spatial dimensions
    transform : affine.Affine, optional
        Transform to use for bbox calculation
    include_bbox : bool, default True
        Whether to include spatial:bbox
    include_registration : bool, default True
        Whether to include spatial:registration
    inplace : bool, default True
        If True, modify object in place

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Object with spatial metadata written
    """
    obj_out = obj if inplace else obj.copy(deep=True)

    # Ensure spatial: convention is declared
    obj_out.attrs = add_convention_declaration(obj_out.attrs, "spatial:", inplace=True)

    # Write spatial:dimensions
    obj_out.attrs["spatial:dimensions"] = [y_dim, x_dim]

    # Write spatial:shape
    if y_dim in obj.dims and x_dim in obj.dims:
        height = obj.sizes[y_dim]
        width = obj.sizes[x_dim]
        obj_out.attrs["spatial:shape"] = [height, width]

    # Write spatial:bbox if transform is available
    if include_bbox and transform is not None:
        try:
            height = obj.sizes[y_dim] if y_dim in obj.dims else 1
            width = obj.sizes[x_dim] if x_dim in obj.dims else 1
            bbox = calculate_spatial_bbox(transform, (height, width))
            obj_out.attrs["spatial:bbox"] = list(bbox)
        except Exception:
            pass

    # Write spatial:registration (default to pixel)
    if include_registration:
        obj_out.attrs["spatial:registration"] = "pixel"

    return obj_out


# Utility functions moved from zarr_conventions.py
def parse_spatial_transform(spatial_transform: Union[list, tuple]) -> Optional[Affine]:
    """Convert spatial:transform array to Affine object."""
    if not isinstance(spatial_transform, (list, tuple)):
        return None
    if len(spatial_transform) != 6:
        return None
    try:
        return Affine(*spatial_transform)
    except (TypeError, ValueError):
        return None


def format_spatial_transform(affine: Affine) -> list:
    """Convert Affine object to spatial:transform array."""
    return [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f]


def parse_proj_code(proj_code: str) -> Optional[rasterio.crs.CRS]:
    """Parse proj:code to CRS."""
    if not isinstance(proj_code, str):
        return None
    try:
        return rasterio.crs.CRS.from_user_input(proj_code)
    except Exception:
        return None


def format_proj_code(crs: rasterio.crs.CRS) -> Optional[str]:
    """Format CRS as proj:code if it has an authority code."""
    try:
        auth_code = crs.to_authority()
        if auth_code:
            authority, code = auth_code
            return f"{authority}:{code}"
    except Exception:
        pass
    return None


def parse_proj_wkt2(proj_wkt2: str) -> Optional[rasterio.crs.CRS]:
    """Parse proj:wkt2 to CRS."""
    if not isinstance(proj_wkt2, str):
        return None
    try:
        return rasterio.crs.CRS.from_wkt(proj_wkt2)
    except Exception:
        return None


def format_proj_wkt2(crs: rasterio.crs.CRS) -> str:
    """Format CRS as proj:wkt2 (WKT2 string)."""
    return crs.to_wkt()


def parse_proj_projjson(proj_projjson: Union[dict, str]) -> Optional[rasterio.crs.CRS]:
    """Parse proj:projjson to CRS."""
    if isinstance(proj_projjson, str):
        try:
            proj_projjson = json.loads(proj_projjson)
        except json.JSONDecodeError:
            return None

    if not isinstance(proj_projjson, dict):
        return None

    try:
        return rasterio.crs.CRS.from_json(json.dumps(proj_projjson))
    except Exception:
        return None


def format_proj_projjson(crs: rasterio.crs.CRS) -> dict:
    """Format CRS as proj:projjson (PROJJSON object)."""
    try:
        projjson_str = crs.to_json()
        return json.loads(projjson_str)
    except Exception:
        # Fallback - create minimal PROJJSON-like structure
        return {"type": "CRS", "wkt": crs.to_wkt()}


def calculate_spatial_bbox(
    transform: Affine, shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """Calculate bounding box from transform and shape."""
    height, width = shape

    # Corner coordinates in pixel space
    corners = [
        (0, 0),  # top-left
        (width, 0),  # top-right
        (width, height),  # bottom-right
        (0, height),  # bottom-left
    ]

    # Transform to spatial coordinates
    spatial_corners = [transform * corner for corner in corners]

    # Extract x and y coordinates
    x_coords = [corner[0] for corner in spatial_corners]
    y_coords = [corner[1] for corner in spatial_corners]

    # Return bounding box as [xmin, ymin, xmax, ymax]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


def has_convention_declared(attrs: dict, convention_name: str) -> bool:
    """Check if a convention is declared in zarr_conventions."""
    zarr_conventions = attrs.get("zarr_conventions", [])
    if not isinstance(zarr_conventions, list):
        return False

    for convention in zarr_conventions:
        if isinstance(convention, dict) and convention.get("name") == convention_name:
            return True

    return False


def get_declared_conventions(attrs: dict) -> set:
    """Get set of declared convention names."""
    zarr_conventions = attrs.get("zarr_conventions", [])
    if not isinstance(zarr_conventions, list):
        return set()

    declared = set()
    for convention in zarr_conventions:
        if isinstance(convention, dict) and "name" in convention:
            declared.add(convention["name"])

    return declared


def add_convention_declaration(
    attrs: dict, convention_name: str, inplace: bool = False
) -> dict:
    """Add convention declaration to zarr_conventions."""
    attrs_out = attrs if inplace else attrs.copy()

    # Get the convention identifier
    if convention_name == "proj:":
        convention = PROJ_CONVENTION
    elif convention_name == "spatial:":
        convention = SPATIAL_CONVENTION
    else:
        return attrs_out

    # Initialize zarr_conventions if needed
    if "zarr_conventions" not in attrs_out:
        attrs_out["zarr_conventions"] = []

    # Check if already declared
    if has_convention_declared(attrs_out, convention_name):
        return attrs_out

    # Add the convention
    attrs_out["zarr_conventions"].append(convention)

    return attrs_out


def write_conventions(
    obj: Union[xarray.Dataset, xarray.DataArray],
    input_crs: Optional[str] = None,
    transform: Optional[Affine] = None,
    crs_format: str = "wkt2",
    inplace: bool = True,
) -> Union[xarray.Dataset, xarray.DataArray]:
    """
    Write complete Zarr spatial and proj conventions.

    Convenience method that writes both CRS (proj:) and spatial (spatial:)
    convention metadata in a single call.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        Object to write metadata to
    input_crs : str, optional
        CRS to write. If not provided, object must have existing CRS.
    transform : affine.Affine, optional
        Transform to write. If not provided, it will be calculated from obj.
    crs_format : str, default "wkt2"
        Which proj: format(s) to write: "code", "wkt2", "projjson", "all"
    inplace : bool, default True
        Whether to modify object in place

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Modified object with complete Zarr conventions
    """
    from rioxarray.raster_array import RasterArray

    # Get CRS if provided
    if input_crs:
        crs = crs_from_user_input(input_crs)
    else:
        # Try to get CRS from object
        rio = RasterArray(obj)
        crs = rio.crs
        if crs is None:
            raise ValueError("No CRS available and input_crs not provided")

    # Write CRS
    obj_modified = write_crs(obj, crs, format=crs_format, inplace=inplace)

    # Write transform
    if transform is not None:
        obj_modified = write_transform(obj_modified, transform, inplace=True)

    # Write spatial metadata - need to get dimensions
    rio = RasterArray(obj_modified)
    if rio.x_dim and rio.y_dim:
        obj_modified = write_spatial_metadata(
            obj_modified,
            rio.y_dim,
            rio.x_dim,
            transform=transform,
            inplace=True
        )

    return obj_modified
