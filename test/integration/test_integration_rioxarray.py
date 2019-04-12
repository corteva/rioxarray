import os

import numpy
import pytest
import xarray
from numpy.testing import assert_almost_equal, assert_array_equal
from rasterio.crs import CRS

from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster
from rioxarray.rioxarray import UNWANTED_RIO_ATTRS, _make_coords
from test.conftest import TEST_COMPARE_DATA_DIR, TEST_INPUT_DATA_DIR


@pytest.fixture(params=[xarray.open_dataset, xarray.open_dataarray])
def modis_reproject(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_UTM.nc"),
        to_proj="+datum=WGS84 +no_defs +proj=utm +units=m +zone=15",
        open=request.param,
    )


@pytest.fixture
def modis_reproject_3d():
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "PLANET_SCOPE_WGS84.nc"),
        to_proj="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
    )


@pytest.fixture(params=[xarray.open_dataset, xarray.open_dataarray])
def interpolate_na(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_INTERPOLATE.nc"),
        open=request.param,
    )


@pytest.fixture
def interpolate_na_3d():
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "PLANET_SCOPE_3D.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "PLANET_SCOPE_3D_INTERPOLATE.nc"),
    )


@pytest.fixture(params=[xarray.open_dataset, xarray.open_dataarray])
def interpolate_na_filled(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(
            TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_INTERPOLATE_FILLED.nc"
        ),
        open=request.param,
    )


@pytest.fixture
def interpolate_na_veris():
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "veris.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "veris_interpolate.nc"),
    )


@pytest.fixture(params=[xarray.open_dataset, xarray.open_dataarray])
def interpolate_na_nan(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_INTERPOLATE_NAN.nc"),
        open=request.param,
    )


@pytest.fixture(params=[xarray.open_dataset, xarray.open_dataarray])
def modis_reproject_match(request):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_MATCH_UTM.nc"),
        match=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY_MATCH.nc"),
        open=request.param,
    )


def _mod_attr(input_xr, attr, val=None, remove=False):
    if hasattr(input_xr, "variables"):
        for var in input_xr.rio.vars:
            _mod_attr(input_xr[var], attr, val=val, remove=remove)
    else:
        if remove:
            del input_xr.attrs[attr]
        else:
            input_xr.attrs[attr] = val


def _get_attr(input_xr, attr):
    if hasattr(input_xr, "variables"):
        return input_xr[input_xr.rio.vars.pop()].attrs[attr]
    return input_xr.attrs[attr]


def _del_attr(input_xr, attr):
    _mod_attr(input_xr, attr, remove=True)


def _assert_xarrays_equal(input_xarray, compare_xarray, precision=7):
    # xarray.testing.assert_equal(input_xarray, compare_xarray)
    def assert_attrs_equal(input_xr, compare_xr):
        """check attrubutes that matter"""
        if isinstance(input_xr, xarray.Dataset):
            assert "creation_date" in input_xr.attrs

        for attr in compare_xr.attrs:
            if (
                attr != "_FillValue"
                and attr not in UNWANTED_RIO_ATTRS
                and attr != "creation_date"
            ):
                try:
                    assert input_xr.attrs[attr] == compare_xr.attrs[attr]
                except ValueError:
                    assert_almost_equal(input_xr.attrs[attr], compare_xr.attrs[attr])

    assert_attrs_equal(input_xarray, compare_xarray)
    if hasattr(input_xarray, "variables"):
        # check coordinates
        for coord in input_xarray.coords:
            if coord in "xy":
                assert_almost_equal(
                    input_xarray[coord].values, compare_xarray[coord].values
                )
            else:
                assert (
                    input_xarray[coord].values == compare_xarray[coord].values
                ).all()

        for var in input_xarray.rio.vars:
            try:
                _assert_xarrays_equal(input_xarray[var], compare_xarray[var])
            except AssertionError:
                print("Error with variable {}".format(var))
                raise
    else:
        try:
            assert_almost_equal(
                input_xarray.values, compare_xarray.values, decimal=precision
            )
        except AssertionError:
            where_diff = input_xarray.values != compare_xarray.values
            print(input_xarray.values[where_diff])
            print(compare_xarray.values[where_diff])
            raise
        assert_attrs_equal(input_xarray, compare_xarray)

        compare_fill_value = compare_xarray.attrs.get(
            "_FillValue", compare_xarray.encoding.get("_FillValue")
        )
        input_fill_value = input_xarray.attrs.get(
            "_FillValue", input_xarray.encoding.get("_FillValue")
        )
        assert_array_equal([input_fill_value], [compare_fill_value])
        assert "grid_mapping" in compare_xarray.attrs
        assert (
            input_xarray[input_xarray.attrs["grid_mapping"]]
            == compare_xarray[compare_xarray.attrs["grid_mapping"]]
        )
        for unwanted_attr in UNWANTED_RIO_ATTRS:
            assert unwanted_attr not in input_xarray.attrs


@pytest.fixture(params=[xarray.open_dataset, xarray.open_dataarray])
def modis_clip(request, tmpdir):
    return dict(
        input=os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"),
        compare=os.path.join(TEST_COMPARE_DATA_DIR, "MODIS_ARRAY_CLIP.nc"),
        open=request.param,
        output=str(tmpdir.join("MODIS_CLIP_DUMP.nc")),
    )


def test_clip_box(modis_clip):
    with modis_clip["open"](modis_clip["input"], autoclose=True) as xdi, modis_clip[
        "open"
    ](modis_clip["compare"], autoclose=True) as xdc:
        clipped_ds = xdi.rio.clip_box(
            minx=xdi.x[4].values,
            miny=xdi.y[6].values,
            maxx=xdi.x[6].values,
            maxy=xdi.y[4].values,
        )
        if isinstance(xdc, xarray.Dataset):
            xdc.attrs["creation_date"] = clipped_ds.attrs["creation_date"]
        _assert_xarrays_equal(clipped_ds, xdc)

        # make sure it safely writes to netcdf
        clipped_ds.to_netcdf(modis_clip["output"])


def test_clip_box__auto_expand(modis_clip):
    with modis_clip["open"](modis_clip["input"], autoclose=True) as xdi, modis_clip[
        "open"
    ](modis_clip["compare"], autoclose=True) as xdc:
        clipped_ds = xdi.rio.clip_box(
            minx=xdi.x[5].values,
            miny=xdi.y[5].values,
            maxx=xdi.x[5].values,
            maxy=xdi.y[5].values,
            auto_expand=True,
        )

        if isinstance(xdc, xarray.Dataset):
            xdc.attrs["creation_date"] = clipped_ds.attrs["creation_date"]

        _assert_xarrays_equal(clipped_ds, xdc)
        # make sure it safely writes to netcdf
        clipped_ds.to_netcdf(modis_clip["output"])


def test_clip_box__nodata_error(modis_clip):
    with modis_clip["open"](modis_clip["input"], autoclose=True) as xdi:
        with pytest.raises(NoDataInBounds):
            xdi.rio.clip_box(
                minx=xdi.x[5].values,
                miny=xdi.y[7].values,
                maxx=xdi.x[4].values,
                maxy=xdi.y[5].values,
            )


def test_clip_box__one_dimension_error(modis_clip):
    with modis_clip["open"](modis_clip["input"], autoclose=True) as xdi:
        # test exception after raster clipped
        with pytest.raises(OneDimensionalRaster):
            xdi.rio.clip_box(
                minx=xdi.x[5].values,
                miny=xdi.y[5].values,
                maxx=xdi.x[5].values,
                maxy=xdi.y[5].values,
            )
        # test exception before raster clipped
        with pytest.raises(OneDimensionalRaster):
            xdi.isel(x=slice(5, 6), y=slice(5, 6)).rio.clip_box(
                minx=xdi.x[5].values,
                miny=xdi.y[7].values,
                maxx=xdi.x[7].values,
                maxy=xdi.y[5].values,
            )


def test_clip_geojson():
    with xarray.open_rasterio(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif")
    ) as xdi:
        # get subset for testing
        subset = xdi.isel(x=slice(150, 160), y=slice(100, 150))
        comp_subset = subset.isel(x=slice(1, None), y=slice(1, None))
        # add transform for test
        comp_subset.attrs["transform"] = tuple(comp_subset.rio.transform(recalc=True))
        # add grid mapping for test
        comp_subset.attrs["grid_mapping"] = "spatial_ref"
        comp_subset.coords["spatial_ref"] = 0
        # make sure nodata exists for test
        comp_subset.attrs["_FillValue"] = comp_subset.attrs["nodatavals"][0]

        geometries = [
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [subset.x.values[0], subset.y.values[-1]],
                        [subset.x.values[0], subset.y.values[0]],
                        [subset.x.values[-1], subset.y.values[0]],
                        [subset.x.values[-1], subset.y.values[-1]],
                        [subset.x.values[0], subset.y.values[-1]],
                    ]
                ],
            }
        ]

        # test data array
        clipped = xdi.rio.clip(geometries, subset.rio.crs)
        _assert_xarrays_equal(clipped, comp_subset)

        # test dataset
        clipped_ds = xdi.to_dataset(name="test_data").rio.clip(
            geometries, subset.rio.crs
        )
        comp_subset_ds = comp_subset.to_dataset(name="test_data")
        _assert_xarrays_equal(clipped_ds, comp_subset_ds)


def test_transform_bounds():
    with xarray.open_dataarray(
        os.path.join(TEST_INPUT_DATA_DIR, "MODIS_ARRAY.nc"), autoclose=True
    ) as xdi:
        bounds = xdi.rio.transform_bounds(
            "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84"
            " +datum=WGS84 +units=m +no_defs",
            densify_pts=100,
        )
        assert_almost_equal(
            bounds,
            (
                -10374232.525903117,
                5591295.917919335,
                -10232919.684719983,
                5656912.314724255,
            ),
        )


def test_reproject(modis_reproject):
    with modis_reproject["open"](
        modis_reproject["input"], mask_and_scale=False, autoclose=True
    ) as mda, modis_reproject["open"](
        modis_reproject["compare"], mask_and_scale=False, autoclose=True
    ) as mdc:
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject_3d(modis_reproject_3d):
    with xarray.open_dataset(
        modis_reproject_3d["input"], mask_and_scale=False, autoclose=True
    ) as mda, xarray.open_dataset(
        modis_reproject_3d["compare"], mask_and_scale=False, autoclose=True
    ) as mdc:
        mds_repr = mda.rio.reproject(modis_reproject_3d["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject__grid_mapping(modis_reproject):
    with modis_reproject["open"](
        modis_reproject["input"], mask_and_scale=False, autoclose=True
    ) as mda, modis_reproject["open"](
        modis_reproject["compare"], mask_and_scale=False, autoclose=True
    ) as mdc:

        # remove 'crs' attribute and add grid mapping
        mda.coords["spatial_ref"] = 0
        mda.coords["spatial_ref"].attrs["spatial_ref"] = CRS.from_user_input(
            _get_attr(mda, "crs")
        ).wkt
        _mod_attr(mda, "grid_mapping", val="spatial_ref")
        _del_attr(mda, "crs")
        mdc.coords["spatial_ref"] = 0
        mdc.coords["spatial_ref"].attrs["spatial_ref"] = CRS.from_user_input(
            _get_attr(mdc, "crs")
        ).wkt
        _mod_attr(mdc, "grid_mapping", val="spatial_ref")
        _del_attr(mdc, "crs")

        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject__masked(modis_reproject):
    with modis_reproject["open"](
        modis_reproject["input"], autoclose=True
    ) as mda, modis_reproject["open"](
        modis_reproject["compare"], autoclose=True
    ) as mdc:
        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject__no_transform(modis_reproject):
    with modis_reproject["open"](
        modis_reproject["input"], autoclose=True
    ) as mda, modis_reproject["open"](
        modis_reproject["compare"], autoclose=True
    ) as mdc:
        orig_trans = _get_attr(mda, "transform")
        _del_attr(mda, "transform")
        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        if hasattr(mds_repr, "variables"):
            for var in mds_repr.rio.vars:
                assert_array_equal(orig_trans, tuple(mda[var].rio.transform()))
        else:
            assert_array_equal(orig_trans, tuple(mda.rio.transform()))
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject__no_nodata(modis_reproject):
    with modis_reproject["open"](
        modis_reproject["input"], mask_and_scale=False, autoclose=True
    ) as mda, modis_reproject["open"](
        modis_reproject["compare"], mask_and_scale=False, autoclose=True
    ) as mdc:
        orig_fill = _get_attr(mda, "_FillValue")
        _del_attr(mda, "_FillValue")
        _del_attr(mda, "nodata")
        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])

        # overwrite test dataset
        # if isinstance(modis_reproject['open'], xarray.DataArray):
        #    mds_repr.to_netcdf(modis_reproject['compare'])

        # replace -9999 with original _FillValue for testing
        if hasattr(mds_repr, "variables"):
            for var in mds_repr.rio.vars:
                mds_repr[var].values[mds_repr[var].values == -9999] = orig_fill
        else:
            mds_repr.values[mds_repr.values == -9999] = orig_fill
        _mod_attr(mdc, "_FillValue", val=-9999)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject__no_nodata_masked(modis_reproject):
    with modis_reproject["open"](
        modis_reproject["input"], autoclose=True
    ) as mda, modis_reproject["open"](
        modis_reproject["compare"], autoclose=True
    ) as mdc:
        _del_attr(mda, "nodata")
        # reproject
        mds_repr = mda.rio.reproject(modis_reproject["to_proj"])
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject_match(modis_reproject_match):
    with modis_reproject_match["open"](
        modis_reproject_match["input"], mask_and_scale=False, autoclose=True
    ) as mda, modis_reproject_match["open"](
        modis_reproject_match["compare"], mask_and_scale=False, autoclose=True
    ) as mdc, xarray.open_dataarray(
        modis_reproject_match["match"], mask_and_scale=False, autoclose=True
    ) as mdm:
        # reproject
        mds_repr = mda.rio.reproject_match(mdm)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject_match__masked(modis_reproject_match):
    with modis_reproject_match["open"](
        modis_reproject_match["input"], autoclose=True
    ) as mda, modis_reproject_match["open"](
        modis_reproject_match["compare"], autoclose=True
    ) as mdc, xarray.open_dataarray(
        modis_reproject_match["match"], autoclose=True
    ) as mdm:
        # reproject
        mds_repr = mda.rio.reproject_match(mdm)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_reproject_match__no_transform_nodata(modis_reproject_match):
    with modis_reproject_match["open"](
        modis_reproject_match["input"], autoclose=True
    ) as mda, modis_reproject_match["open"](
        modis_reproject_match["compare"], autoclose=True
    ) as mdc, xarray.open_dataarray(
        modis_reproject_match["match"], autoclose=True
    ) as mdm:
        _del_attr(mda, "transform")
        _del_attr(mda, "nodata")
        # reproject
        mds_repr = mda.rio.reproject_match(mdm)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_make_src_affine(modis_reproject):
    with xarray.open_dataarray(
        modis_reproject["input"], autoclose=True
    ) as xdi, xarray.open_rasterio(modis_reproject["input"]) as xri:

        # check the transform
        attribute_transform = tuple(xdi.attrs["transform"])
        attribute_transform_func = tuple(xdi.rio.transform())
        calculated_transform = tuple(xdi.rio.transform(recalc=True))
        # delete the transform to ensure it is not being used
        del xdi.attrs["transform"]
        calculated_transform_check = tuple(xdi.rio.transform())
        calculated_transform_check2 = tuple(xdi.rio.transform())
        rio_transform = xri.attrs["transform"]

        assert_array_equal(attribute_transform, attribute_transform_func)
        assert_array_equal(calculated_transform, calculated_transform_check)
        assert_array_equal(calculated_transform, calculated_transform_check2)
        assert_array_equal(attribute_transform, calculated_transform)
        assert_array_equal(calculated_transform[:6], rio_transform)


def test_make_src_affine__single_point():
    point_input = os.path.join(TEST_INPUT_DATA_DIR, "MODIS_POINT.nc")
    with xarray.open_dataarray(point_input, autoclose=True) as xdi:
        # check the transform
        attribute_transform = tuple(xdi.attrs["transform"])
        attribute_transform_func = tuple(xdi.rio.transform())
        calculated_transform = tuple(xdi.rio.transform(recalc=True))
        # delete the transform to ensure it is not being used
        del xdi.attrs["transform"]
        with pytest.raises(ValueError):
            xdi.rio.transform(recalc=True)
        with pytest.raises(ValueError):
            xdi.rio.transform()

        assert_array_equal(attribute_transform, attribute_transform_func)
        assert_array_equal(attribute_transform, calculated_transform)


def test_make_coords__calc_trans(modis_reproject):
    with xarray.open_dataarray(
        modis_reproject["input"], autoclose=True
    ) as xdi, xarray.open_rasterio(modis_reproject["input"]) as xri:
        # calculate coordinates from the calculated transform
        width, height = xdi.rio.shape
        calculated_transform = xdi.rio.transform(recalc=True)
        calc_coords_calc_trans = _make_coords(
            xdi, calculated_transform, width, height, xdi.attrs["crs"]
        )
        widthr, heightr = xri.rio.shape
        calculated_transformr = xri.rio.transform(recalc=True)
        calc_coords_calc_transr = _make_coords(
            xri, calculated_transformr, widthr, heightr, xdi.attrs["crs"]
        )

        # check to see if they all match
        assert_array_equal(xri.coords["x"].values, calc_coords_calc_trans["x"].values)
        assert_array_equal(xri.coords["y"].values, calc_coords_calc_trans["y"].values)
        assert_array_equal(xri.coords["x"].values, calc_coords_calc_transr["x"].values)
        assert_array_equal(xri.coords["y"].values, calc_coords_calc_transr["y"].values)


def test_make_coords__attr_trans(modis_reproject):
    with xarray.open_dataarray(
        modis_reproject["input"], autoclose=True
    ) as xdi, xarray.open_rasterio(modis_reproject["input"]) as xri:
        # calculate coordinates from the attribute transform
        width, height = xdi.rio.shape
        attr_transform = xdi.rio.transform()
        calc_coords_attr_trans = _make_coords(
            xdi, attr_transform, width, height, xdi.attrs["crs"]
        )
        widthr, heightr = xri.rio.shape
        calculated_transformr = xri.rio.transform()
        calc_coords_calc_transr = _make_coords(
            xri, calculated_transformr, widthr, heightr, xdi.attrs["crs"]
        )

        # check to see if they all match
        assert_array_equal(xri.coords["x"].values, calc_coords_calc_transr["x"].values)
        assert_array_equal(xri.coords["y"].values, calc_coords_calc_transr["y"].values)
        assert_array_equal(xri.coords["x"].values, calc_coords_attr_trans["x"].values)
        assert_array_equal(xri.coords["y"].values, calc_coords_attr_trans["y"].values)
        assert_almost_equal(xdi.coords["x"].values, xri.coords["x"].values, decimal=9)
        assert_almost_equal(xdi.coords["y"].values, xri.coords["y"].values, decimal=9)


def test_interpolate_na(interpolate_na):
    with interpolate_na["open"](
        interpolate_na["input"], mask_and_scale=False, autoclose=True
    ) as mda, interpolate_na["open"](
        interpolate_na["compare"], mask_and_scale=False, autoclose=True
    ) as mdc:
        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_interpolate_na_veris(interpolate_na_veris):
    with xarray.open_dataset(
        interpolate_na_veris["input"], mask_and_scale=False, autoclose=True
    ) as mda, xarray.open_dataset(
        interpolate_na_veris["compare"], mask_and_scale=False, autoclose=True
    ) as mdc:
        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_interpolate_na_3d(interpolate_na_3d):
    with xarray.open_dataset(
        interpolate_na_3d["input"], mask_and_scale=False, autoclose=True
    ) as mda, xarray.open_dataset(
        interpolate_na_3d["compare"], mask_and_scale=False, autoclose=True
    ) as mdc:
        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_interpolate_na__nodata_filled(interpolate_na_filled):
    with interpolate_na_filled["open"](
        interpolate_na_filled["input"], mask_and_scale=False, autoclose=True
    ) as mda, interpolate_na_filled["open"](
        interpolate_na_filled["compare"], mask_and_scale=False, autoclose=True
    ) as mdc:
        if hasattr(mda, "variables"):
            for var in mda.rio.vars:
                mda[var].values[mda[var].values == mda[var].rio.nodata] = 400
        else:
            mda.values[mda.values == mda.rio.nodata] = 400

        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_interpolate_na__all_nodata(interpolate_na_nan):
    with interpolate_na_nan["open"](
        interpolate_na_nan["input"], autoclose=True
    ) as mda, interpolate_na_nan["open"](
        interpolate_na_nan["compare"], autoclose=True
    ) as mdc:
        if hasattr(mda, "variables"):
            for var in mda.rio.vars:
                mda[var].values[~numpy.isnan(mda[var].values)] = numpy.nan
        else:
            mda.values[~numpy.isnan(mda.values)] = numpy.nan

        interpolated_ds = mda.rio.interpolate_na()
        # test
        _assert_xarrays_equal(interpolated_ds, mdc)


def test_load_in_geographic_dimensions():
    sentinel_2_geographic = os.path.join(
        TEST_INPUT_DATA_DIR, "sentinel_2_L1C_geographic.nc"
    )
    with xarray.open_dataset(sentinel_2_geographic, autoclose=True) as mda:
        assert mda.rio.x_dim == "longitude"
        assert mda.rio.y_dim == "latitude"
        assert mda.rio.crs.to_epsg() == 4326
        assert mda.red.rio.x_dim == "longitude"
        assert mda.red.rio.y_dim == "latitude"
        assert mda.red.rio.crs.to_epsg() == 4326


def test_geographic_reproject():
    sentinel_2_geographic = os.path.join(
        TEST_INPUT_DATA_DIR, "sentinel_2_L1C_geographic.nc"
    )
    sentinel_2_utm = os.path.join(TEST_COMPARE_DATA_DIR, "sentinel_2_L1C_utm.nc")
    with xarray.open_dataset(
        sentinel_2_geographic, autoclose=True
    ) as mda, xarray.open_dataset(sentinel_2_utm, autoclose=True) as mdc:
        mds_repr = mda.rio.reproject("+init=epsg:32721")
        # mds_repr.to_netcdf(sentinel_2_utm)
        # test
        _assert_xarrays_equal(mds_repr, mdc)


def test_geographic_resample_integer():
    sentinel_2_geographic = os.path.join(
        TEST_INPUT_DATA_DIR, "sentinel_2_L1C_geographic.nc"
    )
    sentinel_2_interp = os.path.join(
        TEST_COMPARE_DATA_DIR, "sentinel_2_L1C_interpolate_na.nc"
    )
    with xarray.open_dataset(
        sentinel_2_geographic, autoclose=True
    ) as mda, xarray.open_dataset(sentinel_2_interp, autoclose=True) as mdc:
        mds_interp = mda.rio.interpolate_na()
        # mds_interp.to_netcdf(sentinel_2_interp)
        # test
        _assert_xarrays_equal(mds_interp, mdc)
