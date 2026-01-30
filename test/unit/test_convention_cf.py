"""Unit tests for the CF convention module."""
import numpy
import xarray
from affine import Affine
from rasterio.crs import CRS

from rioxarray._convention import cf, read_crs_auto, read_transform_auto


def test_read_crs__from_grid_mapping_spatial_ref():
    """Test reading CRS from grid_mapping coordinate's spatial_ref attribute."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])
    data.coords["spatial_ref"] = xarray.Variable((), 0)
    data.coords["spatial_ref"].attrs["spatial_ref"] = "EPSG:4326"

    crs = cf.read_crs(data, grid_mapping="spatial_ref")
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_read_crs__from_grid_mapping_crs_wkt():
    """Test reading CRS from grid_mapping coordinate's crs_wkt attribute."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])
    data.coords["spatial_ref"] = xarray.Variable((), 0)
    data.coords["spatial_ref"].attrs["crs_wkt"] = CRS.from_epsg(4326).to_wkt()

    crs = cf.read_crs(data, grid_mapping="spatial_ref")
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_read_crs__from_legacy_attrs():
    """Test reading CRS from object's attrs (legacy, not CF convention).

    The 'crs' attribute is not part of CF convention but is supported
    for backwards compatibility via the auto-detect method.
    """
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])
    data.attrs["crs"] = "EPSG:4326"

    # CF convention should NOT find this
    crs = cf.read_crs(data)
    assert crs is None

    # Auto-detect should find it
    crs = read_crs_auto(data)
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_read_crs__from_legacy_attrs_with_missing_grid_mapping():
    """Test reading CRS from attrs when grid_mapping doesn't exist.

    This tests a common case where rioxarray's grid_mapping property returns
    "spatial_ref" as a default, but the coordinate doesn't actually exist.
    The CRS should still be found via auto-detect.
    """
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])
    data.attrs["crs"] = "EPSG:4326"

    # CF convention should NOT find this
    crs = cf.read_crs(data, grid_mapping="spatial_ref")
    assert crs is None

    # Auto-detect should find it
    crs = read_crs_auto(data, grid_mapping="spatial_ref")
    assert crs is not None
    assert crs == CRS.from_epsg(4326)


def test_read_crs__not_found():
    """Test that None is returned when no CRS is found."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])

    crs = cf.read_crs(data)
    assert crs is None


def test_read_transform__from_geotransform():
    """Test reading transform from GeoTransform attribute."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])
    data.coords["spatial_ref"] = xarray.Variable((), 0)
    # GeoTransform format: [c, a, b, f, d, e] (GDAL format)
    data.coords["spatial_ref"].attrs["GeoTransform"] = "0.0 1.0 0.0 10.0 0.0 -1.0"

    transform = cf.read_transform(data, grid_mapping="spatial_ref")
    assert transform is not None
    assert transform == Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)


def test_read_transform__from_legacy_attrs():
    """Test reading transform from object's attrs (legacy, not CF convention).

    The 'transform' attribute is not part of CF convention but is supported
    for backwards compatibility via the auto-detect method.
    """
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])
    # Transform stored as list in attrs
    data.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, -1.0, 10.0]

    # CF convention should NOT find this
    transform = cf.read_transform(data)
    assert transform is None

    # Auto-detect should find it
    transform = read_transform_auto(data)
    assert transform is not None
    assert transform == Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)


def test_read_transform__not_found():
    """Test that None is returned when no transform is found."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])

    transform = cf.read_transform(data)
    assert transform is None


def test_read_spatial_dimensions__xy():
    """Test detecting x/y dimension names."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])

    dims = cf.read_spatial_dimensions(data)
    assert dims == ("y", "x")


def test_read_spatial_dimensions__lonlat():
    """Test detecting longitude/latitude dimension names."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["latitude", "longitude"])

    dims = cf.read_spatial_dimensions(data)
    assert dims == ("latitude", "longitude")


def test_read_spatial_dimensions__cf_axis():
    """Test detecting dimensions from CF axis attributes."""
    data = xarray.DataArray(
        numpy.random.rand(10, 10),
        dims=["row", "col"],
        coords={
            "row": ("row", numpy.arange(10)),
            "col": ("col", numpy.arange(10)),
        },
    )
    data.coords["row"].attrs["axis"] = "Y"
    data.coords["col"].attrs["axis"] = "X"

    dims = cf.read_spatial_dimensions(data)
    assert dims == ("row", "col")


def test_read_spatial_dimensions__cf_standard_name():
    """Test detecting dimensions from CF standard_name attributes."""
    data = xarray.DataArray(
        numpy.random.rand(10, 10),
        dims=["lat", "lon"],
        coords={
            "lat": ("lat", numpy.arange(10)),
            "lon": ("lon", numpy.arange(10)),
        },
    )
    data.coords["lat"].attrs["standard_name"] = "latitude"
    data.coords["lon"].attrs["standard_name"] = "longitude"

    dims = cf.read_spatial_dimensions(data)
    assert dims == ("lat", "lon")


def test_read_spatial_dimensions__not_found():
    """Test that None is returned when spatial dimensions are not found."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["a", "b"])

    dims = cf.read_spatial_dimensions(data)
    assert dims is None


def test_write_crs():
    """Test writing CRS to a DataArray."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])
    crs = CRS.from_epsg(4326)

    result = cf.write_crs(data, crs=crs, grid_mapping_name="spatial_ref")

    assert "spatial_ref" in result.coords
    assert result.coords["spatial_ref"].attrs["spatial_ref"] == crs.to_wkt()
    assert result.coords["spatial_ref"].attrs["crs_wkt"] == crs.to_wkt()


def test_write_transform():
    """Test writing transform to a DataArray."""
    data = xarray.DataArray(numpy.random.rand(10, 10), dims=["y", "x"])
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)

    result = cf.write_transform(
        data, transform=transform, grid_mapping_name="spatial_ref"
    )

    assert "spatial_ref" in result.coords
    assert "GeoTransform" in result.coords["spatial_ref"].attrs
    assert (
        result.coords["spatial_ref"].attrs["GeoTransform"]
        == "0.0 1.0 0.0 10.0 0.0 -1.0"
    )
