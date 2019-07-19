History
=======

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
