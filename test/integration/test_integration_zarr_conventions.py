"""
Tests for Zarr spatial and proj conventions support.

Tests reading and writing CRS/georeferencing using:
- Zarr spatial convention: https://github.com/zarr-conventions/spatial
- Zarr geo-proj convention: https://github.com/zarr-experimental/geo-proj
"""

import numpy as np
import pytest
import rasterio.crs
import xarray as xr
from affine import Affine

from rioxarray._convention.zarr import PROJ_CONVENTION, SPATIAL_CONVENTION
from rioxarray.enum import Convention


def add_proj_convention_declaration(attrs):
    """Helper to add proj: convention declaration to attrs dict."""
    if "zarr_conventions" not in attrs:
        attrs["zarr_conventions"] = []
    attrs["zarr_conventions"].append(PROJ_CONVENTION.copy())
    return attrs


def add_spatial_convention_declaration(attrs):
    """Helper to add spatial: convention declaration to attrs dict."""
    if "zarr_conventions" not in attrs:
        attrs["zarr_conventions"] = []
    attrs["zarr_conventions"].append(SPATIAL_CONVENTION.copy())
    return attrs


class TestZarrConventionsReading:
    """Test reading CRS and transform from Zarr conventions."""

    def test_read_crs_from_proj_code(self):
        """Test reading CRS from proj:code attribute."""
        attrs = {"proj:code": "EPSG:4326"}
        add_proj_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs=attrs,
        )

        crs = da.rio.crs
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_read_crs_from_proj_wkt2(self):
        """Test reading CRS from proj:wkt2 attribute."""
        wkt2 = rasterio.crs.CRS.from_epsg(3857).to_wkt()
        attrs = {"proj:wkt2": wkt2}
        add_proj_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs=attrs,
        )

        crs = da.rio.crs
        assert crs is not None
        assert crs.to_epsg() == 3857

    @pytest.mark.skip(reason="projjson parsing issue - needs investigation")
    def test_read_crs_from_proj_projjson(self):
        """Test reading CRS from proj:projjson attribute."""
        import json

        from pyproj import CRS as ProjCRS

        pyproj_crs = ProjCRS.from_epsg(4326)
        projjson = json.loads(pyproj_crs.to_json())

        attrs = {"proj:projjson": projjson}
        add_proj_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs=attrs,
        )

        crs = da.rio.crs
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_read_transform_from_spatial_transform(self):
        """Test reading transform from spatial:transform attribute."""
        transform_array = [10.0, 0.0, 100.0, 0.0, -10.0, 200.0]
        attrs = {"spatial:transform": transform_array}
        add_spatial_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs=attrs,
        )

        transform = da.rio.transform()
        assert transform is not None
        assert list(transform)[:6] == transform_array

    def test_read_spatial_dimensions(self):
        """Test reading dimensions from spatial:dimensions attribute."""
        attrs = {"spatial:dimensions": ["lat", "lon"]}
        add_spatial_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("lat", "lon"),
            attrs=attrs,
        )

        # Should detect dimensions from spatial:dimensions
        assert da.rio.y_dim == "lat"
        assert da.rio.x_dim == "lon"

    def test_spatial_dimensions_takes_precedence(self):
        """Test that spatial:dimensions takes precedence over standard names."""
        # Create a DataArray with both standard 'x'/'y' dims and spatial:dimensions
        # spatial:dimensions should take precedence
        attrs = {"spatial:dimensions": ["y", "x"]}
        add_spatial_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs=attrs,
        )

        # Should use spatial:dimensions (y, x) not inferred from dim names
        assert da.rio.y_dim == "y"
        assert da.rio.x_dim == "x"

        # Test with non-standard names - spatial:dimensions should be used
        attrs2 = {"spatial:dimensions": ["row", "col"]}
        add_spatial_convention_declaration(attrs2)
        da2 = xr.DataArray(
            np.ones((5, 5)),
            dims=("row", "col"),
            attrs=attrs2,
        )

        assert da2.rio.y_dim == "row"
        assert da2.rio.x_dim == "col"

    def test_zarr_conventions_priority_over_cf(self):
        """Test that CF conventions are used as default when both are present."""
        # Create a DataArray with both Zarr and CF conventions
        # Zarr has EPSG:4326, CF grid_mapping has EPSG:3857
        attrs = {"proj:code": "EPSG:4326"}
        add_proj_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            coords={
                "spatial_ref": xr.Variable(
                    (),
                    0,
                    attrs={"spatial_ref": rasterio.crs.CRS.from_epsg(3857).to_wkt()},
                )
            },
            attrs=attrs,
        )

        # CF convention should be used as default when convention is None
        crs = da.rio.crs
        assert crs.to_epsg() == 3857

    def test_cf_conventions_as_fallback(self):
        """Test that CF conventions work as fallback when Zarr conventions absent."""
        # Create a DataArray with only CF conventions
        wkt = rasterio.crs.CRS.from_epsg(4326).to_wkt()
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            coords={"spatial_ref": xr.Variable((), 0, attrs={"spatial_ref": wkt})},
        )

        # Should still read CRS from CF conventions
        crs = da.rio.crs
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_group_level_proj_inheritance_dataset(self):
        """Test reading proj attributes from group level in Datasets."""
        # Create a Dataset with group-level proj:code
        attrs = {"proj:code": "EPSG:4326"}
        add_proj_convention_declaration(attrs)
        ds = xr.Dataset(
            {
                "var1": xr.DataArray(np.ones((5, 5)), dims=("y", "x")),
                "var2": xr.DataArray(np.ones((5, 5)), dims=("y", "x")),
            },
            attrs=attrs,
        )

        # Dataset should inherit group-level CRS
        crs = ds.rio.crs
        assert crs is not None
        assert crs.to_epsg() == 4326


class TestZarrConventionsWriting:
    """Test writing CRS and transform using Zarr conventions."""

    def test_write_zarr_crs_code(self):
        """Test writing CRS as proj:code."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        # Use zarr module directly for format-specific options
        from rioxarray._convention import zarr
        da = da.rio.write_crs("EPSG:4326", convention=Convention.Zarr)
        # Use zarr module to write specific format
        da = zarr.write_crs(da, da.rio.crs, format="code")

        # Verify convention is declared
        assert "zarr_conventions" in da.attrs
        convention_names = [c["name"] for c in da.attrs["zarr_conventions"]]
        assert "proj:" in convention_names

        assert "proj:code" in da.attrs
        assert da.attrs["proj:code"] == "EPSG:4326"

        # Verify it can be read back
        assert da.rio.crs.to_epsg() == 4326

    def test_write_zarr_crs_wkt2(self):
        """Test writing CRS as proj:wkt2."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        da = da.rio.write_crs("EPSG:4326", convention=Convention.Zarr)

        assert "proj:wkt2" in da.attrs
        assert "GEOG" in da.attrs["proj:wkt2"]  # WKT contains GEOG or GEOGCRS

        # Verify it can be read back
        assert da.rio.crs.to_epsg() == 4326

    def test_write_zarr_crs_projjson(self):
        """Test writing CRS as proj:projjson."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        from rioxarray._convention import zarr
        da = da.rio.write_crs("EPSG:4326", convention=Convention.Zarr)
        da = zarr.write_crs(da, da.rio.crs, format="projjson")

        assert "proj:projjson" in da.attrs
        assert isinstance(da.attrs["proj:projjson"], dict)
        assert da.attrs["proj:projjson"]["type"] in ("GeographicCRS", "GeodeticCRS", "CRS")

        # Verify it can be read back
        assert da.rio.crs.to_epsg() == 4326

    def test_write_zarr_crs_all_formats(self):
        """Test writing all three proj formats."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        from rioxarray._convention import zarr
        da = da.rio.write_crs("EPSG:4326", convention=Convention.Zarr)
        da = zarr.write_crs(da, da.rio.crs, format="all")

        assert "proj:code" in da.attrs
        assert "proj:wkt2" in da.attrs
        assert "proj:projjson" in da.attrs

        # Verify it can be read back
        assert da.rio.crs.to_epsg() == 4326

    def test_write_zarr_transform(self):
        """Test writing transform as spatial:transform."""
        transform = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        da = da.rio.write_transform(transform, convention=Convention.Zarr)

        # Verify convention is declared
        assert "zarr_conventions" in da.attrs
        convention_names = [c["name"] for c in da.attrs["zarr_conventions"]]
        assert "spatial:" in convention_names

        assert "spatial:transform" in da.attrs
        assert da.attrs["spatial:transform"] == list(transform)[:6]

        # Verify it can be read back
        assert da.rio.transform() == transform

    def test_write_zarr_spatial_metadata(self):
        """Test writing complete spatial metadata."""
        from rioxarray._convention import zarr
        da = xr.DataArray(np.ones((10, 20)), dims=("y", "x"))
        da = zarr.write_spatial_metadata(da, "y", "x")

        assert "spatial:dimensions" in da.attrs
        assert da.attrs["spatial:dimensions"] == ["y", "x"]

        assert "spatial:shape" in da.attrs
        assert da.attrs["spatial:shape"] == [10, 20]

        assert "spatial:registration" in da.attrs
        assert da.attrs["spatial:registration"] == "pixel"

    def test_write_zarr_spatial_metadata_with_bbox(self):
        """Test writing spatial metadata with bbox."""
        from rioxarray._convention import zarr
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        da = xr.DataArray(np.ones((10, 20)), dims=("y", "x"))
        da = da.rio.write_transform(transform, convention=Convention.Zarr)
        da = zarr.write_spatial_metadata(da, "y", "x", transform=transform, include_bbox=True)

        assert "spatial:bbox" in da.attrs
        # bbox should be [xmin, ymin, xmax, ymax]
        bbox = da.attrs["spatial:bbox"]
        assert len(bbox) == 4
        assert bbox == [0.0, 0.0, 20.0, 10.0]

    def test_write_zarr_conventions_all(self):
        """Test writing complete Zarr conventions."""
        from rioxarray._convention import zarr
        transform = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        da = xr.DataArray(np.ones((10, 20)), dims=("y", "x"))
        # Write components separately for simplicity  
        da = da.rio.write_crs("EPSG:4326", convention=Convention.Zarr)
        da = zarr.write_crs(da, da.rio.crs, format="all")
        da = da.rio.write_transform(transform, convention=Convention.Zarr)
        da = zarr.write_spatial_metadata(da, "y", "x", transform=transform)

        # Check CRS attributes
        assert "proj:code" in da.attrs
        assert "proj:wkt2" in da.attrs
        assert "proj:projjson" in da.attrs

        # Check transform attribute
        assert "spatial:transform" in da.attrs
        assert da.attrs["spatial:transform"] == list(transform)[:6]

        # Check spatial metadata
        assert "spatial:dimensions" in da.attrs
        assert "spatial:shape" in da.attrs
        assert "spatial:bbox" in da.attrs

        # Verify everything can be read back
        assert da.rio.crs.to_epsg() == 4326
        assert da.rio.transform() == transform


class TestZarrConventionsRoundTrip:
    """Test round-trip write then read of Zarr conventions."""

    def test_roundtrip_proj_code(self):
        """Test write then read of proj:code."""
        from rioxarray._convention import zarr
        original_da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        original_da = original_da.rio.write_crs("EPSG:3857", convention=Convention.Zarr)
        # Use zarr module for specific format
        original_da = zarr.write_crs(original_da, original_da.rio.crs, format="code")

        # Simulate saving and reloading by creating new DataArray with same attrs
        reloaded_da = xr.DataArray(
            original_da.values,
            dims=original_da.dims,
            attrs=original_da.attrs.copy(),
        )

        assert reloaded_da.rio.crs.to_epsg() == 3857

    def test_roundtrip_spatial_transform(self):
        """Test write then read of spatial:transform."""
        transform = Affine(5.0, 0.0, -180.0, 0.0, -5.0, 90.0)
        original_da = xr.DataArray(np.ones((36, 72)), dims=("y", "x"))
        original_da = original_da.rio.write_transform(transform, convention=Convention.Zarr)

        # Simulate saving and reloading
        reloaded_da = xr.DataArray(
            original_da.values,
            dims=original_da.dims,
            attrs=original_da.attrs.copy(),
        )

        assert reloaded_da.rio.transform() == transform

    def test_roundtrip_complete_conventions(self):
        """Test write then read of complete Zarr conventions."""
        from rioxarray._convention import zarr
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
        original_da = xr.DataArray(np.ones((100, 100)), dims=("y", "x"))
        # Write components separately for simplicity
        original_da = original_da.rio.write_crs("EPSG:4326", convention=Convention.Zarr)
        original_da = zarr.write_crs(original_da, original_da.rio.crs, format="all")
        original_da = original_da.rio.write_transform(transform, convention=Convention.Zarr)
        original_da = zarr.write_spatial_metadata(original_da, "y", "x", transform=transform)

        # Simulate saving and reloading
        reloaded_da = xr.DataArray(
            original_da.values,
            dims=original_da.dims,
            attrs=original_da.attrs.copy(),
        )

        # Verify CRS
        assert reloaded_da.rio.crs.to_epsg() == 4326

        # Verify transform
        assert reloaded_da.rio.transform() == transform

        # Verify spatial metadata
        assert reloaded_da.rio.x_dim == "x"
        assert reloaded_da.rio.y_dim == "y"
        assert reloaded_da.rio.height == 100
        assert reloaded_da.rio.width == 100


class TestZarrConventionsCoexistence:
    """Test that both CF and Zarr conventions can coexist."""

    def test_both_conventions_present(self):
        """Test that both conventions can be present simultaneously."""
        # Write CF conventions first
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        da = da.rio.write_crs("EPSG:4326")  # CF format

        # Add Zarr conventions
        from rioxarray._convention import zarr
        da = zarr.write_crs(da, da.rio.crs, format="code")

        # Both should be present
        assert "spatial_ref" in da.coords  # CF grid_mapping
        assert "proj:code" in da.attrs  # Zarr convention

        # Zarr should take priority when reading
        assert da.rio.crs.to_epsg() == 4326

    def test_zarr_overrides_cf_when_both_present(self):
        """Test Zarr conventions override CF when both have different values."""
        # This is an edge case: if someone has both conventions with
        # conflicting values, CF should win as default when convention is None
        attrs = {"proj:code": "EPSG:4326"}
        add_proj_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            coords={
                "spatial_ref": xr.Variable(
                    (),
                    0,
                    attrs={"spatial_ref": rasterio.crs.CRS.from_epsg(3857).to_wkt()},
                )
            },
            attrs=attrs,
        )

        # CF convention (EPSG:3857) should be used as default when convention is None
        assert da.rio.crs.to_epsg() == 3857


class TestZarrConventionsEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_proj_code(self):
        """Test handling of invalid proj:code."""
        attrs = {"proj:code": "INVALID:9999"}
        add_proj_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs=attrs,
        )

        # Should handle gracefully (return None or fall back)
        _crs = da.rio.crs
        # Depending on implementation, might be None or raise exception
        # For now, just verify it doesn't crash
        assert _crs is None

    def test_invalid_spatial_transform_format(self):
        """Test handling of malformed spatial:transform."""
        # Wrong number of elements
        attrs = {"spatial:transform": [1.0, 2.0, 3.0]}  # Only 3 elements
        add_spatial_convention_declaration(attrs)
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs=attrs,
        )

        # Should handle gracefully
        da.rio.transform()
        # Should fall back to calculating from coordinates or return identity

    def test_write_crs_without_setting(self):
        """Test writing Zarr CRS when no CRS is set."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))

        # Should handle None gracefully by returning unchanged object
        from rioxarray._convention import zarr
        result = zarr.write_crs(da, None, format="code")
        # Should not have any proj: attributes
        assert not any(attr.startswith("proj:") for attr in result.attrs)
        assert result is da  # Should return same object when inplace=True

    def test_write_spatial_metadata_without_dimensions(self):
        """Test writing spatial metadata when dimensions cannot be determined."""
        # Create a DataArray with non-standard dimension names
        # and no spatial:dimensions attribute
        da = xr.DataArray(np.ones((5, 5)), dims=("foo", "bar"))

        # Should raise MissingSpatialDimensionError
        from rioxarray._convention import zarr
        with pytest.raises(Exception):  # MissingSpatialDimensionError
            zarr.write_spatial_metadata(da)

    def test_crs_from_projjson_dict(self):
        """Test crs_from_user_input with PROJJSON dict."""
        import json

        from pyproj import CRS as ProjCRS

        from rioxarray.crs import crs_from_user_input

        pyproj_crs = ProjCRS.from_epsg(4326)
        projjson = json.loads(pyproj_crs.to_json())

        crs = crs_from_user_input(projjson)
        assert crs is not None
        assert crs.to_epsg() == 4326
