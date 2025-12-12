"""
Tests for the new convention architecture.
"""
import numpy as np
import xarray as xr
from affine import Affine

import rioxarray
from rioxarray.enum import Convention


class TestConventionArchitecture:
    """Test the new convention architecture."""

    def test_convention_enum(self):
        """Test Convention enum exists and has expected values."""
        assert hasattr(Convention, "CF")
        assert hasattr(Convention, "Zarr")
        assert Convention.CF.value == "CF"
        assert Convention.Zarr.value == "Zarr"

    def test_set_options_convention(self):
        """Test setting convention through set_options."""
        # Test default convention
        with rioxarray.set_options():
            from rioxarray._options import CONVENTION, get_option

            assert get_option(CONVENTION) == Convention.CF

        # Test setting Zarr convention
        with rioxarray.set_options(convention=Convention.Zarr):
            from rioxarray._options import CONVENTION, get_option

            assert get_option(CONVENTION) == Convention.Zarr

        # Test setting CF convention explicitly
        with rioxarray.set_options(convention=Convention.CF):
            from rioxarray._options import CONVENTION, get_option

            assert get_option(CONVENTION) == Convention.CF

    def test_write_crs_with_convention_parameter(self):
        """Test write_crs with explicit convention parameter."""
        data = np.random.rand(3, 3)
        da = xr.DataArray(data, dims=("y", "x"))

        # Test CF convention
        da_cf = da.rio.write_crs("EPSG:4326", convention=Convention.CF)
        assert hasattr(da_cf, "coords")
        # CF should create a grid_mapping coordinate
        assert "spatial_ref" in da_cf.coords or any(
            "spatial_ref" in str(coord) for coord in da_cf.coords
        )

        # Test Zarr convention
        da_zarr = da.rio.write_crs("EPSG:4326", convention=Convention.Zarr)
        # Zarr should add proj: attributes and convention declaration
        assert "zarr_conventions" in da_zarr.attrs
        assert any(
            conv.get("name") == "proj:" for conv in da_zarr.attrs["zarr_conventions"]
        )

    def test_write_transform_with_convention_parameter(self):
        """Test write_transform with explicit convention parameter."""
        data = np.random.rand(3, 3)
        da = xr.DataArray(data, dims=("y", "x"))
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)

        # Test CF convention
        da_cf = da.rio.write_transform(transform, convention=Convention.CF)
        # CF should have a grid_mapping coordinate with GeoTransform
        assert hasattr(da_cf, "coords")

        # Test Zarr convention
        da_zarr = da.rio.write_transform(transform, convention=Convention.Zarr)
        # Zarr should have spatial:transform attribute and convention declaration
        assert "zarr_conventions" in da_zarr.attrs
        assert "spatial:transform" in da_zarr.attrs
        assert any(
            conv.get("name") == "spatial:" for conv in da_zarr.attrs["zarr_conventions"]
        )

        # Verify transform values
        assert da_zarr.attrs["spatial:transform"] == [1.0, 0.0, 0.0, 0.0, -1.0, 3.0]

    def test_crs_reading_follows_global_convention(self):
        """Test that CRS reading follows the global convention setting."""
        data = np.random.rand(3, 3)
        da = xr.DataArray(data, dims=("y", "x"))

        # Create data with both CF and Zarr CRS information
        da_with_cf = da.rio.write_crs("EPSG:4326", convention=Convention.CF)
        da_with_zarr = da_with_cf.rio.write_crs(
            "EPSG:3857", convention=Convention.Zarr
        )  # Different CRS

        # With CF convention setting, should read CF CRS (4326)
        with rioxarray.set_options(convention=Convention.CF):
            crs = da_with_zarr.rio.crs
            assert crs.to_epsg() == 4326

        # With Zarr convention setting, should read Zarr CRS (3857)
        with rioxarray.set_options(convention=Convention.Zarr):
            # Reset cached CRS
            da_with_zarr.rio._crs = None
            crs = da_with_zarr.rio.crs
            assert crs.to_epsg() == 3857

    def test_zarr_conventions_methods_exist(self):
        """Test that new Zarr convention methods exist."""
        data = np.random.rand(3, 3)
        da = xr.DataArray(data, dims=("y", "x"))

        # Test methods exist
        assert hasattr(da.rio, "write_zarr_crs")
        assert hasattr(da.rio, "write_zarr_transform")
        assert hasattr(da.rio, "write_zarr_spatial_metadata")
        assert hasattr(da.rio, "write_zarr_conventions")

        # Test basic functionality
        da_zarr = da.rio.write_zarr_crs("EPSG:4326")
        assert "proj:code" in da_zarr.attrs
        assert da_zarr.attrs["proj:code"] == "EPSG:4326"
