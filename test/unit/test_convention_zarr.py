"""Unit tests for the Zarr convention module."""
import numpy as np
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
import rioxarray  # noqa: F401
from rioxarray._convention import zarr
from rioxarray._convention.zarr import ZarrConvention


def test_has_convention_declared__proj():
    """Test checking for proj: convention declaration."""
    attrs = {
        "zarr_conventions": [
            {
                "name": "proj:",
                "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
            }
        ]
    }
    assert zarr.has_convention_declared(attrs, "proj:") is True
    assert zarr.has_convention_declared(attrs, "spatial:") is False


def test_has_convention_declared__spatial():
    """Test checking for spatial: convention declaration."""
    attrs = {
        "zarr_conventions": [
            {
                "name": "spatial:",
                "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
            }
        ]
    }
    assert zarr.has_convention_declared(attrs, "spatial:") is True
    assert zarr.has_convention_declared(attrs, "proj:") is False


def test_has_convention_declared__not_declared():
    """Test when no convention is declared."""
    attrs = {}
    assert zarr.has_convention_declared(attrs, "proj:") is False
    assert zarr.has_convention_declared(attrs, "spatial:") is False


def test_get_declared_conventions():
    """Test getting all declared conventions."""
    attrs = {
        "zarr_conventions": [
            {"name": "proj:", "uuid": "test-uuid-1"},
            {"name": "spatial:", "uuid": "test-uuid-2"},
        ]
    }
    declared = zarr.get_declared_conventions(attrs)
    assert declared == {"proj:", "spatial:"}


def test_parse_spatial_transform():
    """Test parsing spatial:transform array."""
    transform_array = [1.0, 0.0, 100.0, 0.0, -1.0, 200.0]
    result = zarr.parse_spatial_transform(transform_array)
    assert result == Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)


def test_parse_spatial_transform__invalid():
    """Test parsing invalid spatial:transform."""
    assert zarr.parse_spatial_transform([1, 2, 3]) is None
    assert zarr.parse_spatial_transform("invalid") is None


def test_read_crs__from_wkt2():
    """Test reading CRS from proj:wkt2 attribute."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    data.attrs["zarr_conventions"] = [zarr.PROJ_CONVENTION]
    data.attrs["proj:wkt2"] = CRS.from_epsg(4326).to_wkt()

    crs = ZarrConvention.read_crs(data)
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_read_crs__from_code():
    """Test reading CRS from proj:code attribute."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    data.attrs["zarr_conventions"] = [zarr.PROJ_CONVENTION]
    data.attrs["proj:code"] = "EPSG:4326"

    crs = ZarrConvention.read_crs(data)
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_read_crs__not_found():
    """Test that None is returned when no CRS is found."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])

    crs = ZarrConvention.read_crs(data)
    assert crs is None


def test_read_crs__no_convention_declared():
    """Test that CRS is not read when convention is not declared."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    # Add proj attributes but no convention declaration
    data.attrs["proj:wkt2"] = CRS.from_epsg(4326).to_wkt()

    crs = ZarrConvention.read_crs(data)
    assert crs is None


def test_read_transform__from_spatial_transform():
    """Test reading transform from spatial:transform attribute."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    data.attrs["zarr_conventions"] = [zarr.SPATIAL_CONVENTION]
    data.attrs["spatial:transform"] = [1.0, 0.0, 100.0, 0.0, -1.0, 200.0]

    transform = ZarrConvention.read_transform(data)
    assert transform is not None
    assert transform == Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)


def test_read_transform__not_found():
    """Test that None is returned when no transform is found."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])

    transform = ZarrConvention.read_transform(data)
    assert transform is None


def test_read_transform__no_convention_declared():
    """Test that transform is not read when convention is not declared."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    # Add spatial attributes but no convention declaration
    data.attrs["spatial:transform"] = [1.0, 0.0, 100.0, 0.0, -1.0, 200.0]

    transform = ZarrConvention.read_transform(data)
    assert transform is None


def test_read_spatial_dimensions():
    """Test reading spatial dimensions from spatial:dimensions attribute."""
    data = xr.DataArray(np.random.rand(10, 20), dims=["lat", "lon"])
    data.attrs["zarr_conventions"] = [zarr.SPATIAL_CONVENTION]
    data.attrs["spatial:dimensions"] = ["lat", "lon"]

    dims = ZarrConvention.read_spatial_dimensions(data)
    assert dims == ("lat", "lon")


def test_read_spatial_dimensions__not_found():
    """Test that None is returned when no spatial dimensions are found."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])

    dims = ZarrConvention.read_spatial_dimensions(data)
    assert dims is None


def test_read_spatial_dimensions__no_convention_declared():
    """Test that spatial dims are not read when convention is not declared."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    # Add spatial attributes but no convention declaration
    data.attrs["spatial:dimensions"] = ["y", "x"]

    dims = ZarrConvention.read_spatial_dimensions(data)
    assert dims is None


# ============================================================================
# Formatting utilities
# ============================================================================


def test_format_proj_wkt2():
    """Test formatting CRS as WKT2 string."""
    crs = CRS.from_epsg(4326)
    result = zarr.format_proj_wkt2(crs)
    assert isinstance(result, str)
    assert CRS.from_wkt(result) == crs


def test_format_spatial_transform():
    """Test converting Affine to [a, b, c, d, e, f] list."""
    affine = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
    assert zarr.format_spatial_transform(affine) == [1.0, 0.0, 100.0, 0.0, -1.0, 200.0]


# ============================================================================
# Convention declaration
# ============================================================================


def test_add_convention_declaration():
    """Test adding a convention declaration to empty attrs."""
    attrs = {}
    zarr.add_convention_declaration(attrs, "proj:")
    assert zarr.has_convention_declared(attrs, "proj:") is True


def test_add_convention_declaration__idempotent():
    """Test that duplicate declarations are not added."""
    attrs = {}
    zarr.add_convention_declaration(attrs, "proj:")
    zarr.add_convention_declaration(attrs, "proj:")
    assert len(attrs["zarr_conventions"]) == 1


# ============================================================================
# ZarrConvention.write_crs
# ============================================================================


def test_write_crs():
    """Test writing CRS writes proj:wkt2 and declares convention."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    crs = CRS.from_epsg(4326)
    result = ZarrConvention.write_crs(data, crs=crs)
    assert zarr.has_convention_declared(result.attrs, "proj:") is True
    assert "proj:wkt2" in result.attrs
    assert CRS.from_wkt(result.attrs["proj:wkt2"]) == crs


def test_write_crs__ignores_grid_mapping_name():
    """Test that grid_mapping_name kwarg (CF-specific) is silently ignored."""
    data = xr.DataArray(np.random.rand(10, 10), dims=["y", "x"])
    result = ZarrConvention.write_crs(
        data, crs=CRS.from_epsg(4326), grid_mapping_name="spatial_ref"
    )
    assert "proj:wkt2" in result.attrs


# ============================================================================
# ZarrConvention.write_transform
# ============================================================================


def test_write_transform():
    """Test writing transform writes all spatial: attributes."""
    data = xr.DataArray(np.random.rand(10, 20), dims=["y", "x"])
    transform = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
    result = ZarrConvention.write_transform(data, transform=transform)
    assert zarr.has_convention_declared(result.attrs, "spatial:") is True
    assert result.attrs["spatial:transform"] == [1.0, 0.0, 100.0, 0.0, -1.0, 200.0]
    assert result.attrs["spatial:dimensions"] == ["y", "x"]
    assert result.attrs["spatial:shape"] == [10, 20]
    assert "spatial:bbox" in result.attrs
    assert result.attrs["spatial:registration"] == "pixel"


def test_write_transform__ignores_grid_mapping_name():
    """Test that grid_mapping_name kwarg (CF-specific) is silently ignored."""
    data = xr.DataArray(np.random.rand(10, 20), dims=["y", "x"])
    transform = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
    result = ZarrConvention.write_transform(
        data, transform=transform, grid_mapping_name="spatial_ref"
    )
    assert "spatial:transform" in result.attrs
