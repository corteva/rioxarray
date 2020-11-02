History
=======

0.1.1
------
- BUG: Check all CRS are the same in the dataset in crs() method

0.1.0
------
- BUG: Ensure transform correct in rio.clip without coords (pull #165)
- BUG: Ensure the nodata value matches the dtype (pull #166)
- Raise deprecation exception in add_spatial_ref and add_xy_grid_meta (pull #168)

0.0.31
------
- Deprecate add_spatial_ref and fix warning for add_xy_grid_meta (pull #158)

0.0.30
------
- BUG: Fix assigning fill value in `rio.pad_box` (pull #140)
- ENH: Add `rio.write_transform` to store cache in GDAL location (issue #129 & #139)
- ENH: Use rasterio windows for `rio.clip_box` (issue #142)
- BUG: Add support for negative indexes in rio.isel_window (pull #145)
- BUG: Write transform based on window in rio.isel_window (pull #145)
- ENH: Add `rio.count`, `rio.slice_xy()`, `rio.bounds()`, `rio.resolution()`, `rio.transform_bounds()` to Dataset level
- ENH: Add `rio.write_coordinate_system()` (issue #147)
- ENH: Search CF coordinate metadata to find coordinates (issue #147)
- ENH: Default `rio.clip` to assume geometry has CRS of dataset (pull #150)
- ENH: Add `rio.grid_mapping` and `rio.write_grid_mapping` & preserve original grid mapping (pull #151)

0.0.29
-------
- BUG: Remove unnecessary memory copies in reproject method (pull #136)
- BUG: Fix order of axis in `rio.isel_window` (pull #133)
- BUG: Allow clipping with disjoint geometries (issue #132)
- BUG: Remove automatically setting tiled=True for windowed writing (pull #134)
- ENH: Add `rio.pad_box` (pull #138)

0.0.28
-------
- rio.reproject: change input kwarg dst_affine_width_height -> shape & transform (#125)
- ENH: Use pyproj.CRS to read/write CF parameters (issue #124)

0.0.27
------
- ENH: Added optional `shape` argument to `rio.reproject` (pull #116)
- Fix ``RasterioDeprecationWarning`` (pull #117)
- BUG: Make rio.shape order same as rasterio dataset shape (height, width) (pull #121)
- Fix open_rasterio() for WarpedVRT with specified src_crs (pydata/xarray/pull/4104 & pull #120)
- BUG: Use internal reprojection as engine for resampling window in merge (pull #123)

0.0.26
------
- ENH: Added :func:`rioxarray.show_versions` (issue #106)

0.0.25
------
- BUG: Use recalc=True when using transform internally & ensure stable when coordinates unavailable. (issue #97)

0.0.24
------
- ENH: Add variable names to error messages for clarity (pull #99)
- BUG: Use assign_coords in _decode_datetime_cf (issue #101)

0.0.23
------
- BUG: Fix 'rio.set_spatial_dims' so information saved with 'rio' accesors (issue #94)
- ENH: Make 'rio.isel_window' available for datasets (pull #95)

0.0.22
-------
- ENH: Use pyproj.CRS internally to manage GDAL 2/3 transition (issue #92)
- ENH: Add MissingCRS exceptions for 'rio.clip' and 'rio.reproject' (pull #93)

0.0.21
-------
- ENH: Added to_raster method for Datasets (issue #76)

0.0.20
------
- BUG: ensure band_key is list when iterating over bands for mask and scale (pull #87)

0.0.19
-------
- Add support for writing scales & offsets to raster (pull #79)
- Don't write standard raster metadata to raster tags (issue #78)

0.0.18
------
- Fixed windowed writing to require tiled output raster (pull #66)
- Write data array attributes using `rio.to_raster` (issue #64)
- Write variable name to descriptions if possible in `rio.to_raster` (issue #64)
- Add `mask_and_scale` option to `rioxarray.open_rasterio()` (issue #67)
- Hide NotGeoreferencedWarning warning when subdatasets are present using open_rasterio (issue #65)
- Add support for loading in 1D variables in `xarray.open_rasterio()` (issue #43)
- Load in netCDF metadata on the variable level (pull #73)
- Add rioxarray.merge module (issue #46)

0.0.17
------
- Renamed `descriptions` to `long_name` when opening with `open_rasterio()` (pull #63)
- Make `units` & `long_name` scalar if they exist in rasterio attributes (pull #63)

0.0.16
------
-  Add support for netcdf/hdf groups with different shapes (pull #62)

0.0.15
------
- Added `variable` and `group` kwargs to `rioxarray.open_rasterio()` to allow filtering of subdatasets (pull #57)
- Added `default_name` kwarg to `rioxarray.open_rasterio()` for backup when the original does not exist (pull #59)
- Added `recalc_transform` kwarg to `rio.to_raster()` (pull #56)

0.0.14
------
- Added `windowed` kwarg to `rio.to_raster()` to write to raster using windowed writing (pull #54)
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
