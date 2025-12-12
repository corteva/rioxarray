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


class TestZarrConventionsReading:
    """Test reading CRS and transform from Zarr conventions."""

    def test_read_crs_from_proj_code(self):
        """Test reading CRS from proj:code attribute."""
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs={"proj:code": "EPSG:4326"},
        )

        crs = da.rio.crs
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_read_crs_from_proj_wkt2(self):
        """Test reading CRS from proj:wkt2 attribute."""
        wkt2 = rasterio.crs.CRS.from_epsg(3857).to_wkt()
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs={"proj:wkt2": wkt2},
        )

        crs = da.rio.crs
        assert crs is not None
        assert crs.to_epsg() == 3857

    def test_read_crs_from_proj_projjson(self):
        """Test reading CRS from proj:projjson attribute."""
        import json

        from pyproj import CRS as ProjCRS

        pyproj_crs = ProjCRS.from_epsg(4326)
        projjson = json.loads(pyproj_crs.to_json())

        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs={"proj:projjson": projjson},
        )

        crs = da.rio.crs
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_read_transform_from_spatial_transform(self):
        """Test reading transform from spatial:transform attribute."""
        transform_array = [10.0, 0.0, 100.0, 0.0, -10.0, 200.0]
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs={"spatial:transform": transform_array},
        )

        transform = da.rio.transform()
        assert transform is not None
        assert list(transform)[:6] == transform_array

    def test_read_spatial_dimensions(self):
        """Test reading dimensions from spatial:dimensions attribute."""
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("lat", "lon"),
            attrs={"spatial:dimensions": ["lat", "lon"]},
        )

        # Should detect dimensions from spatial:dimensions
        assert da.rio.y_dim == "lat"
        assert da.rio.x_dim == "lon"

    def test_zarr_conventions_priority_over_cf(self):
        """Test that Zarr conventions take priority over CF conventions."""
        # Create a DataArray with both Zarr and CF conventions
        # Zarr has EPSG:4326, CF grid_mapping has EPSG:3857
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
            attrs={"proj:code": "EPSG:4326"},
        )

        # Zarr convention should take priority
        crs = da.rio.crs
        assert crs.to_epsg() == 4326

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
        ds = xr.Dataset(
            {
                "var1": xr.DataArray(np.ones((5, 5)), dims=("y", "x")),
                "var2": xr.DataArray(np.ones((5, 5)), dims=("y", "x")),
            },
            attrs={"proj:code": "EPSG:4326"},
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
        da = da.rio.write_zarr_crs("EPSG:4326", format="code")

        assert "proj:code" in da.attrs
        assert da.attrs["proj:code"] == "EPSG:4326"

        # Verify it can be read back
        assert da.rio.crs.to_epsg() == 4326

    def test_write_zarr_crs_wkt2(self):
        """Test writing CRS as proj:wkt2."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        da = da.rio.write_zarr_crs("EPSG:4326", format="wkt2")

        assert "proj:wkt2" in da.attrs
        assert "GEOG" in da.attrs["proj:wkt2"]  # WKT contains GEOG or GEOGCRS

        # Verify it can be read back
        assert da.rio.crs.to_epsg() == 4326

    def test_write_zarr_crs_projjson(self):
        """Test writing CRS as proj:projjson."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        da = da.rio.write_zarr_crs("EPSG:4326", format="projjson")

        assert "proj:projjson" in da.attrs
        assert isinstance(da.attrs["proj:projjson"], dict)
        assert da.attrs["proj:projjson"]["type"] in ("GeographicCRS", "GeodeticCRS")

        # Verify it can be read back
        assert da.rio.crs.to_epsg() == 4326

    def test_write_zarr_crs_all_formats(self):
        """Test writing all three proj formats."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        da = da.rio.write_zarr_crs("EPSG:4326", format="all")

        assert "proj:code" in da.attrs
        assert "proj:wkt2" in da.attrs
        assert "proj:projjson" in da.attrs

        # Verify it can be read back
        assert da.rio.crs.to_epsg() == 4326

    def test_write_zarr_transform(self):
        """Test writing transform as spatial:transform."""
        transform = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        da = da.rio.write_zarr_transform(transform)

        assert "spatial:transform" in da.attrs
        assert da.attrs["spatial:transform"] == list(transform)[:6]

        # Verify it can be read back
        assert da.rio.transform() == transform

    def test_write_zarr_spatial_metadata(self):
        """Test writing complete spatial metadata."""
        da = xr.DataArray(np.ones((10, 20)), dims=("y", "x"))
        da = da.rio.write_zarr_spatial_metadata()

        assert "spatial:dimensions" in da.attrs
        assert da.attrs["spatial:dimensions"] == ["y", "x"]

        assert "spatial:shape" in da.attrs
        assert da.attrs["spatial:shape"] == [10, 20]

        assert "spatial:registration" in da.attrs
        assert da.attrs["spatial:registration"] == "pixel"

    def test_write_zarr_spatial_metadata_with_bbox(self):
        """Test writing spatial metadata with bbox."""
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        da = xr.DataArray(np.ones((10, 20)), dims=("y", "x"))
        da = da.rio.write_zarr_transform(transform)
        da = da.rio.write_zarr_spatial_metadata(include_bbox=True)

        assert "spatial:bbox" in da.attrs
        # bbox should be [xmin, ymin, xmax, ymax]
        bbox = da.attrs["spatial:bbox"]
        assert len(bbox) == 4
        assert bbox == [0.0, 0.0, 20.0, 10.0]

    def test_write_zarr_conventions_all(self):
        """Test writing complete Zarr conventions."""
        transform = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        da = xr.DataArray(np.ones((10, 20)), dims=("y", "x"))
        da = da.rio.write_zarr_conventions(
            input_crs="EPSG:4326",
            transform=transform,
            crs_format="all",
        )

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
        original_da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))
        original_da = original_da.rio.write_zarr_crs("EPSG:3857", format="code")

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
        original_da = original_da.rio.write_zarr_transform(transform)

        # Simulate saving and reloading
        reloaded_da = xr.DataArray(
            original_da.values,
            dims=original_da.dims,
            attrs=original_da.attrs.copy(),
        )

        assert reloaded_da.rio.transform() == transform

    def test_roundtrip_complete_conventions(self):
        """Test write then read of complete Zarr conventions."""
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
        original_da = xr.DataArray(np.ones((100, 100)), dims=("y", "x"))
        original_da = original_da.rio.write_zarr_conventions(
            input_crs="EPSG:4326",
            transform=transform,
            crs_format="all",
        )

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
        da = da.rio.write_zarr_conventions("EPSG:4326", crs_format="code")

        # Both should be present
        assert "spatial_ref" in da.coords  # CF grid_mapping
        assert "proj:code" in da.attrs  # Zarr convention

        # Zarr should take priority when reading
        assert da.rio.crs.to_epsg() == 4326

    def test_zarr_overrides_cf_when_both_present(self):
        """Test Zarr conventions override CF when both have different values."""
        # This is an edge case: if someone has both conventions with
        # conflicting values, Zarr should win
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
            attrs={"proj:code": "EPSG:4326"},
        )

        # Zarr convention (EPSG:4326) should take priority over CF (EPSG:3857)
        assert da.rio.crs.to_epsg() == 4326


class TestZarrConventionsEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_proj_code(self):
        """Test handling of invalid proj:code."""
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs={"proj:code": "INVALID:9999"},
        )

        # Should handle gracefully (return None or fall back)
        _crs = da.rio.crs
        # Depending on implementation, might be None or raise exception
        # For now, just verify it doesn't crash
        assert _crs is None

    def test_invalid_spatial_transform_format(self):
        """Test handling of malformed spatial:transform."""
        # Wrong number of elements
        da = xr.DataArray(
            np.ones((5, 5)),
            dims=("y", "x"),
            attrs={"spatial:transform": [1.0, 2.0, 3.0]},  # Only 3 elements
        )

        # Should handle gracefully
        da.rio.transform()
        # Should fall back to calculating from coordinates or return identity

    def test_write_crs_without_setting(self):
        """Test writing Zarr CRS when no CRS is set."""
        da = xr.DataArray(np.ones((5, 5)), dims=("y", "x"))

        # Should raise MissingCRS
        with pytest.raises(Exception):  # MissingCRS
            da.rio.write_zarr_crs(format="code")

    def test_write_spatial_metadata_without_dimensions(self):
        """Test writing spatial metadata when dimensions cannot be determined."""
        # Create a DataArray with non-standard dimension names
        # and no spatial:dimensions attribute
        da = xr.DataArray(np.ones((5, 5)), dims=("foo", "bar"))

        # Should raise MissingSpatialDimensionError
        with pytest.raises(Exception):  # MissingSpatialDimensionError
            da.rio.write_zarr_spatial_metadata()

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
