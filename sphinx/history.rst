History
=======

0.0.14
------
- Add `windowed` kwarg to `rio.to_raster()` to write to raster using windowed writing (pull #54)
- Added add `rio.isel_window()` to allow selection using a rasterio.windows.Window (pull #54)

0.0.13
------
- Improve CRS searching for xarray.Dataset & use default grid mapping name (pull #51)

0.0.12
------
- Use `xarray.open_rasterio()` for `rioxarray.open_rasterio()` with xarray<0.12.3 (pull #40)

0.0.11
------
- Added `open_kwargs` to pass into `rasterio.open()` when using `rioxarray.open_rasterio()` (pull #48)
- Added example opening Cloud Optimized GeoTiff (issue #45)

0.0.10
------
- Add support for opening netcdf/hdf files with `rioxarray.open_rasterio` (issue #32)
- Added support for custom CRS with wkt attribute for datacube CRS support (issue #35)
- Added `rio.set_nodata()`, `rio.write_nodata()`, `rio.set_attrs()`, `rio.update_attrs()` (issue #37)

0.0.9
-----
- Add `rioxarray.open_rasterio` (issue #7)

0.0.8
-----
- Fix setting nodata in _add_attrs_proj (pull #30) 

0.0.7
-----
- Add option to do an inverted clip (pull #29) 

0.0.6
-----
- Add support for scalar coordinates in reproject (issue #15)
- Updated writing encoding for FutureWarning (issue #18)
- Use input raster profile for defaults to write output raster profile if opened with `xarray.open_rasterio` (issue #19)
- Preserve None nodata if opened with `xarray.open_rasterio` (issue #20)
- Added `drop` argument for `clip()` (issue #25)
- Fix order of `CRS` for reprojecting geometries in `clip()` (pull #24)
- Added `set_spatial_dims()` method for datasets when dimensions not found (issue #27)

0.0.5
-----
- Find nodata and nodatavals in 'nodata' property (pull #12)
- Added 'encoded_nodata' property to DataArray (pull #12)
- Write the raster with encoded_nodata instead of NaN for nodata (pull #12)
- Added methods to set and write CRS (issue #5)

0.0.4
------
- Added ability to export data array to raster (pull #8)
