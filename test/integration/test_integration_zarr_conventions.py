"""Integration tests for reading Zarr conventions."""
import numpy as np
import pyproj
import xarray as xr
from affine import Affine
from rasterio.crs import CRS

import rioxarray  # noqa: F401
from rioxarray import set_options
from rioxarray._convention import zarr
from rioxarray.enum import Convention


def _create_zarr_array_with_proj():
    """Create a DataArray with Zarr proj: convention attributes."""
    data = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
        coords={
            "y": np.arange(10),
            "x": np.arange(20),
        },
    )
    data.attrs["zarr_conventions"] = [zarr.PROJ_CONVENTION]
    data.attrs["proj:wkt2"] = CRS.from_epsg(4326).to_wkt()
    return data


def _create_zarr_array_with_spatial():
    """Create a DataArray with Zarr spatial: convention attributes."""
    data = xr.DataArray(
        np.random.rand(10, 20),
        dims=["lat", "lon"],
        coords={
            "lat": np.arange(10),
            "lon": np.arange(20),
        },
    )
    data.attrs["zarr_conventions"] = [zarr.SPATIAL_CONVENTION]
    data.attrs["spatial:transform"] = [1.0, 0.0, 100.0, 0.0, -1.0, 200.0]
    data.attrs["spatial:dimensions"] = ["lat", "lon"]
    return data


def _create_zarr_array_with_both():
    """Create a DataArray with both Zarr conventions."""
    data = xr.DataArray(
        np.random.rand(10, 20),
        dims=["lat", "lon"],
        coords={
            "lat": np.arange(10),
            "lon": np.arange(20),
        },
    )
    data.attrs["zarr_conventions"] = [zarr.PROJ_CONVENTION, zarr.SPATIAL_CONVENTION]
    data.attrs["proj:wkt2"] = CRS.from_epsg(32618).to_wkt()
    data.attrs["spatial:transform"] = [10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0]
    data.attrs["spatial:dimensions"] = ["lat", "lon"]
    return data


def test_read_crs_from_zarr_convention():
    """Test reading CRS from DataArray with Zarr proj: convention."""
    data = _create_zarr_array_with_proj()

    crs = data.rio.crs
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_read_crs_from_zarr_convention__with_setting():
    """Test reading CRS with Convention.Zarr setting."""
    data = _create_zarr_array_with_proj()

    with set_options(convention=Convention.Zarr):
        crs = data.rio.crs
        assert crs is not None
        assert crs == CRS.from_epsg(4326)


def test_read_transform_from_zarr_convention():
    """Test reading transform from DataArray with Zarr spatial: convention."""
    data = _create_zarr_array_with_spatial()

    # Access transform via rio accessor
    # Check the cached version reads from Zarr spatial:transform
    cached = data.rio._cached_transform()
    assert cached is not None
    assert cached == Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)


def test_read_spatial_dimensions_from_zarr_convention():
    """Test reading spatial dimensions from Zarr spatial: convention."""
    data = _create_zarr_array_with_spatial()

    assert data.rio.x_dim == "lon"
    assert data.rio.y_dim == "lat"


def test_read_both_conventions():
    """Test reading from DataArray with both Zarr conventions."""
    data = _create_zarr_array_with_both()

    # CRS from proj:
    crs = data.rio.crs
    assert crs is not None
    assert crs == CRS.from_epsg(32618)

    # Transform from spatial:
    cached = data.rio._cached_transform()
    assert cached is not None
    assert cached == Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)

    # Dimensions from spatial:
    assert data.rio.x_dim == "lon"
    assert data.rio.y_dim == "lat"


def test_fallback_zarr_to_cf():
    """Test that CF convention is tried as fallback when Zarr not found."""
    # Create data with CF convention
    data = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
        coords={
            "y": np.arange(10),
            "x": np.arange(20),
        },
    )
    data.coords["spatial_ref"] = xr.Variable((), 0)
    data.coords["spatial_ref"].attrs["spatial_ref"] = "EPSG:4326"

    # Even with Zarr preference, should fall back to CF
    with set_options(convention=Convention.Zarr):
        crs = data.rio.crs
        assert crs is not None
        assert crs == CRS.from_epsg(4326)


def test_fallback_cf_to_zarr():
    """Test that Zarr convention is tried as fallback when CF not found."""
    # Create data with Zarr convention only
    data = _create_zarr_array_with_proj()

    # With CF preference (default), should fall back to Zarr
    crs = data.rio.crs
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_priority_zarr_over_cf():
    """Test that Zarr convention takes priority when setting is Zarr."""
    # Create data with both conventions (different CRS values)
    data = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
        coords={
            "y": np.arange(10),
            "x": np.arange(20),
        },
    )
    # CF convention
    data.coords["spatial_ref"] = xr.Variable((), 0)
    data.coords["spatial_ref"].attrs["spatial_ref"] = "EPSG:4326"

    # Zarr convention (different CRS)
    data.attrs["zarr_conventions"] = [zarr.PROJ_CONVENTION]
    data.attrs["proj:wkt2"] = CRS.from_epsg(32618).to_wkt()

    # With Zarr setting, should prefer Zarr CRS
    with set_options(convention=Convention.Zarr):
        crs = data.rio.crs
        assert crs is not None
        assert crs == CRS.from_epsg(32618)

    # Reset to check default
    data2 = data.copy(deep=True)
    data2.rio._crs = None  # Reset cached CRS

    # With default setting (CF priority), should prefer CF CRS
    crs = data2.rio.crs
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_read_proj_code():
    """Test reading CRS from proj:code attribute."""
    data = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
    )
    data.attrs["zarr_conventions"] = [zarr.PROJ_CONVENTION]
    data.attrs["proj:code"] = "EPSG:32618"

    crs = data.rio.crs
    assert crs is not None
    assert crs == CRS.from_epsg(32618)


def test_read_proj_projjson():
    """Test reading CRS from proj:projjson attribute."""
    data = xr.DataArray(
        np.random.rand(10, 20),
        dims=["y", "x"],
    )
    data.attrs["zarr_conventions"] = [zarr.PROJ_CONVENTION]
    data.attrs["proj:projjson"] = pyproj.CRS.from_epsg(4326).to_json_dict()

    crs = data.rio.crs
    assert crs is not None
    assert crs == CRS.from_epsg(4326)
