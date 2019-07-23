import os

import numpy
from numpy.testing import assert_almost_equal

import rioxarray
from test.conftest import TEST_COMPARE_DATA_DIR, _assert_xarrays_equal


def test_open_rasterio_mask_chunk_clip():
    with rioxarray.open_rasterio(
        os.path.join(TEST_COMPARE_DATA_DIR, "small_dem_3m_merged.tif"),
        masked=True,
        chunks=True,
    ) as xdi:
        assert str(type(xdi.data)) == "<class 'dask.array.core.Array'>"
        assert xdi.chunks == ((1,), (245,), (574,))
        assert numpy.isnan(xdi.values).sum() == 52119
        assert xdi.encoding == {"_FillValue": 0.0}
        attrs = dict(xdi.attrs)
        assert_almost_equal(
            attrs.pop("transform"),
            (3.0, 0.0, 425047.68381405267, 0.0, -3.0, 4615780.040546387),
        )
        assert attrs == {
            "grid_mapping": "spatial_ref",
            "offsets": (0.0,),
            "scales": (1.0,),
        }

        # get subset for testing
        subset = xdi.isel(x=slice(150, 160), y=slice(100, 150))
        comp_subset = subset.isel(x=slice(1, None), y=slice(1, None))
        # add transform for test
        comp_subset.attrs["transform"] = tuple(comp_subset.rio.transform(recalc=True))

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
        clipped = xdi.rio.clip(geometries, comp_subset.rio.crs)
        _assert_xarrays_equal(clipped, comp_subset)
        assert clipped.encoding == {"_FillValue": 0.0}

        # test dataset
        clipped_ds = xdi.to_dataset(name="test_data").rio.clip(
            geometries, subset.rio.crs
        )
        comp_subset_ds = comp_subset.to_dataset(name="test_data")
        _assert_xarrays_equal(clipped_ds, comp_subset_ds)
        assert clipped_ds.test_data.encoding == {"_FillValue": 0.0}
