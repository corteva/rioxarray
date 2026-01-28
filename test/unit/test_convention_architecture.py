"""
Test the convention architecture for both CF and Zarr conventions.
"""
import numpy as np
import pytest
import xarray as xr
from affine import Affine

import rioxarray
from rioxarray.enum import Convention


@pytest.fixture
def sample_data():
    """Create a simple test dataset."""
    da = xr.DataArray(
        np.random.rand(10, 10),
        dims=["y", "x"],
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    return da


@pytest.fixture
def sample_transform():
    """Create a simple transform."""
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)


class TestConventionArchitecture:
    """Test the new convention-based architecture."""

    def test_convention_enum(self):
        """Test that Convention enum is available and works."""
        assert Convention.CF.value == "CF"
        assert Convention.Zarr.value == "Zarr"

    def test_default_convention_options(self):
        """Test that default options work."""
        with rioxarray.set_options(convention=Convention.CF):
            from rioxarray._options import CONVENTION, get_option

            assert get_option(CONVENTION) is Convention.CF

    def test_zarr_convention_options(self):
        """Test that Zarr convention can be set."""
        with rioxarray.set_options(convention=Convention.Zarr):
            from rioxarray._options import CONVENTION, get_option

            assert get_option(CONVENTION) is Convention.Zarr

    def test_write_crs_cf_convention(self, sample_data):
        """Test writing CRS with CF convention."""
        da_with_crs = sample_data.rio.write_crs("EPSG:4326", convention=Convention.CF)

        # Should have grid_mapping coordinate
        assert "spatial_ref" in da_with_crs.coords
        # Should have grid_mapping attribute
        assert da_with_crs.attrs.get("grid_mapping") == "spatial_ref"
        # Should have WKT in grid_mapping coordinate
        assert "spatial_ref" in da_with_crs.coords["spatial_ref"].attrs

    def test_write_crs_zarr_convention(self, sample_data):
        """Test writing CRS with Zarr convention."""
        da_with_crs = sample_data.rio.write_crs("EPSG:4326", convention=Convention.Zarr)

        # Should have proj:wkt2 attribute (default format)
        assert "proj:wkt2" in da_with_crs.attrs
        assert (
            "GEOGCS" in da_with_crs.attrs["proj:wkt2"]
            or "GEOGCRS" in da_with_crs.attrs["proj:wkt2"]
        )
        # Should have zarr_conventions declaration
        assert "zarr_conventions" in da_with_crs.attrs
        conventions = da_with_crs.attrs["zarr_conventions"]
        assert any(conv.get("name") == "proj:" for conv in conventions)

    def test_write_transform_cf_convention(self, sample_data, sample_transform):
        """Test writing transform with CF convention."""
        da_with_transform = sample_data.rio.write_transform(
            sample_transform, convention=Convention.CF
        )

        # Should have grid_mapping coordinate with GeoTransform
        assert "spatial_ref" in da_with_transform.coords
        assert "GeoTransform" in da_with_transform.coords["spatial_ref"].attrs

    def test_write_transform_zarr_convention(self, sample_data, sample_transform):
        """Test writing transform with Zarr convention."""
        da_with_transform = sample_data.rio.write_transform(
            sample_transform, convention=Convention.Zarr
        )

        # Should have spatial:transform attribute
        assert "spatial:transform" in da_with_transform.attrs
        spatial_transform = da_with_transform.attrs["spatial:transform"]
        assert len(spatial_transform) == 6
        assert spatial_transform == [1.0, 0.0, 0.0, 0.0, -1.0, 10.0]
        # Should have zarr_conventions declaration
        assert "zarr_conventions" in da_with_transform.attrs
        conventions = da_with_transform.attrs["zarr_conventions"]
        assert any(conv.get("name") == "spatial:" for conv in conventions)


class TestConventionModules:
    """Test the individual convention modules."""

    def test_cf_module_exists(self):
        """Test that CF module can be imported."""
        from rioxarray._convention import cf

        assert hasattr(cf, "read_crs")
        assert hasattr(cf, "read_transform")
        assert hasattr(cf, "write_crs")
        assert hasattr(cf, "write_transform")

    def test_zarr_module_exists(self):
        """Test that Zarr module can be imported."""
        from rioxarray._convention import zarr

        assert hasattr(zarr, "read_crs")
        assert hasattr(zarr, "read_transform")
        assert hasattr(zarr, "write_crs")
        assert hasattr(zarr, "write_transform")


class TestBackwardCompatibility:
    """Test that existing code continues to work."""

    def test_write_crs_without_convention_parameter(self, sample_data):
        """Test that write_crs works without convention parameter (uses default)."""
        # Default should be CF
        da_with_crs = sample_data.rio.write_crs("EPSG:4326")

        # Should use CF convention by default
        assert "spatial_ref" in da_with_crs.coords
        assert da_with_crs.attrs.get("grid_mapping") == "spatial_ref"

    def test_write_transform_without_convention_parameter(
        self, sample_data, sample_transform
    ):
        """Test that write_transform works without convention parameter."""
        da_with_transform = sample_data.rio.write_transform(sample_transform)

        # Should use CF convention by default
        assert "spatial_ref" in da_with_transform.coords
        assert "GeoTransform" in da_with_transform.coords["spatial_ref"].attrs
