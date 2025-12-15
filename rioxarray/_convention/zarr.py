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


def _parse_crs_from_attrs(attrs: dict, convention_check: bool = True) -> Optional[rasterio.crs.CRS]:
    """
    Parse CRS from proj: attributes with fallback priority.

    Parameters
    ----------
    attrs : dict
        Attributes dictionary to parse from
    convention_check : bool, default True
        Whether to check for convention declaration

    Returns
    -------
    rasterio.crs.CRS or None
        Parsed CRS object, or None if not found
    """
    if convention_check and not has_convention_declared(attrs, "proj:"):
        return None

    for proj_attr, parser in [
        ("proj:wkt2", parse_proj_wkt2),
        ("proj:code", parse_proj_code),
        ("proj:projjson", parse_proj_projjson),
    ]:
        try:
            proj_value = attrs.get(proj_attr)
            if proj_value is not None:
                parsed_crs = parser(proj_value)
                if parsed_crs is not None:
                    return parsed_crs
        except (KeyError, Exception):
            pass
    return None


def _parse_transform_from_attrs(attrs: dict, convention_check: bool = True) -> Optional[Affine]:
    """
    Parse transform from spatial: attributes.

    Parameters
    ----------
    attrs : dict
        Attributes dictionary to parse from
    convention_check : bool, default True
        Whether to check for convention declaration

    Returns
    -------
    affine.Affine or None
        Parsed transform object, or None if not found
    """
    if convention_check and not has_convention_declared(attrs, "spatial:"):
        return None

    try:
        spatial_transform = attrs.get("spatial:transform")
        if spatial_transform is not None:
            return parse_spatial_transform(spatial_transform)
    except (KeyError, Exception):
        pass
    return None


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
    # Parse CRS from object attributes
    return _parse_crs_from_attrs(obj.attrs)


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
    # Parse transform from object attributes
    return _parse_transform_from_attrs(obj.attrs)


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

    from rioxarray.raster_array import RasterArray

    rio = RasterArray(obj)

    if rio.y_dim and rio.x_dim:
        obj_out = _write_spatial_metadata(
            obj_out, rio.y_dim, rio.x_dim, transform=transform, inplace=True
        )

    return obj_out


def _write_spatial_metadata(
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
        # Write spatial:bbox if dimensions are available
        if x_dim in obj.dims and y_dim in obj.dims:
            height = obj.sizes[y_dim]
            width = obj.sizes[x_dim]
            bbox = calculate_spatial_bbox(transform, (height, width))
            obj_out.attrs["spatial:bbox"] = list(bbox)

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
    return rasterio.crs.CRS.from_user_input(proj_code)


def format_proj_code(crs: rasterio.crs.CRS) -> Optional[str]:
    """Format CRS as proj:code if it has an authority code."""
    auth_code = crs.to_authority()
    if auth_code:
        authority, code = auth_code
        return f"{authority}:{code}"
    return None


def parse_proj_wkt2(proj_wkt2: str) -> Optional[rasterio.crs.CRS]:
    """Parse proj:wkt2 to CRS."""
    if not isinstance(proj_wkt2, str):
        return None
    return rasterio.crs.CRS.from_wkt(proj_wkt2)


def format_proj_wkt2(crs: rasterio.crs.CRS) -> str:
    """Format CRS as proj:wkt2 (WKT2 string)."""
    return crs.to_wkt()


def parse_proj_projjson(proj_projjson: Union[dict, str]) -> Optional[rasterio.crs.CRS]:
    """Parse proj:projjson to CRS."""
    if isinstance(proj_projjson, str):
        proj_projjson = json.loads(proj_projjson)

    if not isinstance(proj_projjson, dict):
        return None

    return rasterio.crs.CRS.from_json(json.dumps(proj_projjson))


def format_proj_projjson(crs: rasterio.crs.CRS) -> dict:
    """Format CRS as proj:projjson (PROJJSON object)."""
    # Use _projjson() method for proper PROJJSON format
    projjson_str = crs._projjson()
    return json.loads(projjson_str)


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

    # Write transform and spatial metadata
    if transform is not None:
        # Get dimensions
        rio = RasterArray(obj_modified)
        obj_modified = write_transform(
            obj_modified,
            transform,
            y_dim=rio.y_dim,
            x_dim=rio.x_dim,
            inplace=True
        )
    else:
        # Write just spatial metadata if no transform
        rio = RasterArray(obj_modified)
        if rio.x_dim and rio.y_dim:
            obj_modified = _write_spatial_metadata(
                obj_modified, rio.y_dim, rio.x_dim, transform=None, inplace=True
            )

    return obj_modified
