"""
Tests for the new convention architecture.
"""
import numpy as np
import xarray as xr
from affine import Affine

import rioxarray
from rioxarray.enum import Convention


class TestConventionArchitecture:
    """Test integration scenarios for the convention architecture."""

    def test_convention_interaction_with_existing_metadata(self):
        """Test how conventions interact when metadata already exists."""
        data = np.random.rand(3, 3)
        da = xr.DataArray(data, dims=("y", "x"))

        # Start with CF metadata
        da_cf = da.rio.write_crs("EPSG:4326", convention=Convention.CF)

        # Add Zarr metadata on top (should coexist)
        da_both = da_cf.rio.write_crs("EPSG:3857", convention=Convention.Zarr)

        # Should have both types of metadata
        assert "spatial_ref" in da_both.coords  # CF metadata
        assert "zarr_conventions" in da_both.attrs  # Zarr metadata
        assert any(
            conv.get("name") == "proj:" for conv in da_both.attrs["zarr_conventions"]
        )

    def test_convention_metadata_coexistence(self):
        """Test that CF and Zarr transform metadata can coexist."""
        data = np.random.rand(3, 3)
        da = xr.DataArray(data, dims=("y", "x"))
        transform1 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)
        transform2 = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 6.0)

        # Add CF transform first
        da_cf = da.rio.write_transform(transform1, convention=Convention.CF)

        # Add Zarr transform on top
        da_both = da_cf.rio.write_transform(transform2, convention=Convention.Zarr)

        # Both should coexist
        assert "spatial_ref" in da_both.coords  # CF metadata
        assert "GeoTransform" in da_both.coords["spatial_ref"].attrs
        assert "spatial:transform" in da_both.attrs  # Zarr metadata
        assert da_both.attrs["spatial:transform"] == [2.0, 0.0, 0.0, 0.0, -2.0, 6.0]

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
            # Reset cached CRS before reading
            da_with_zarr.rio._crs = None
            crs = da_with_zarr.rio.crs
            assert crs.to_epsg() == 4326

        # With Zarr convention setting, should read Zarr CRS (3857)
        with rioxarray.set_options(convention=Convention.Zarr):
            # Reset cached CRS
            da_with_zarr.rio._crs = None
            crs = da_with_zarr.rio.crs
            assert crs.to_epsg() == 3857
