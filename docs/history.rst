History
=======

Latest
-------
- DEP: Support Python 3.10-3.12 (pull #723)
- DEP: rasterio 1.3+, pyproj 3.3+ (pull #725, #727)
- DEP: xarray 2022.3.0+ & numpy 1.23+ (pull #728)

0.15.0
------
- BUG: Fix setting spatial dims internally during propagation (pull #682)
- ENH: Pass on on-disk chunk sizes as preferred chunk sizes to the xarray backend (pull #678)
- MNT: add __all__ to top level module (issue #680)

0.14.1
------
- BUG: Fix :mod:`rioxarray.merge` CRS check (pull #655)
- BUG: Remove tags with metadata added by rasterio in :func:`rioxarray.open_rasterio` (issue #666)

0.14.0
------
- DEP: Drop Python 3.8 support (issue #582)
- DEP: pin rasterio>=1.2 (pull #642)
- BUG: Fix WarpedVRT in :func:`rioxarray.open_rasterio` when band_as_variable=True (issue #644)
- BUG: Fix usage of `encode_cf_variable` in `rio.to_raster` (pull #652)

0.13.4
------
- DEP: pin numpy>=1.21 (pull #636)

0.13.3
------
- BUG: Handle data type error in `rio.reproject` (issue #618)

0.13.2
------
- BUG:dataset: Fix writing tags for bands (issue #615)
- BUG:dataset: prevent overwriting long_name attribute (pull #616)

0.13.1
------
- BUG: Fix closing files manually (pull #607)
- BUG: Add GDAL 3.6 driver auto-select fix (pull #606)

0.13.0
-------
- ENH: Added band_as_variable option to open_rasterio (pull #600)

0.12.4
------
- ENH: Added band_as_variable option to open_rasterio (issue #296)
- BUG: Pass warp_extras dictionary to raster.vrt.WarpedVRT (issue #598)

0.12.3
------
- BUG: Handle CF CRS export errors in `rio.write_crs` (discussion #591)

0.12.2
------
- BUG: Fix `mask_and_scale` data load after `.sel` (issue #580)

0.12.1
------
- BUG: Handle `_Unsigned` and load in all attributes (pull #575)

0.12.0
-------
- ENH: Allow passing in bounds of different CRS in `rio.clip_box` (pull #563)

0.11.2
------
- BUG: Fix reading file handle with dask (issue #550)
- BUG: Fix reading cint16 files with dask (issue #542)
- BUG: Ensure `rio.bounds` ordered correctly (issue #545)
- BUG: Allow reading from `io.BytesIO` (issue #549)

0.11.1
------
- BUG: Fix WarpedVRT param cache in :func:`rioxarray.open_rasterio` (issue #515)
- BUG: Always generate coordinates in `rio.reproject` when GCPS|RPCS present (issue #517)

0.11.0
------
- TYPE: Add more type hints (issue #373)
- ENH: Add additional GDAL information to :func:`rioxarray.show_versions` (pull #513)

0.10.3
------
- BUG: Remove xarray crs attribute in rio.write_crs (issue #488)

0.10.2
-------
- BUG: Lazy load colormap through _manager.acquire() in merge (issue #479)

0.10.1
-------
- DEP: pin rasterio>=1.1.1 (pull #471)
- BUG: Corrected bounds and transform args to float (pull #475)

0.10.0
-------
- DEP: Drop Python 3.7 support (issue #451)
- ENH: Add GCPs reading and writing (issue #376)

0.9.1
------
- BUG: Force coordinates to be exactly the same in `rio.reproject_match` (issue #298)

0.9.0
------
- ENH: Allow additional kwargs to pass from reproject_match() -> reproject() (pull #436)

0.8.0
------
- DEP: Make scipy an optional dependency (issue #413)
- BUG: Return cached transform when axis data missing (pull #419)
- BUG: Fix negative indexes in `rio.isel_window` (issue #421)

0.7.1
------
- BUG: Handle transforms with rotation (pull #401)

0.7.0
------
- BUG: `rio.clip` and `rio.clip_box` skip non-geospatial arrays in datasets when clipping (pull #392)
- ENH: Add option for users to skip variables without spatial dimensions (pull #395)

0.6.1
------
- BUG: Fix indexing error when `mask_and_scale=True` was combined with band dim chunking (issue #387, pull #388)

0.6.0
------
- ENH: Add pad option to `rio.isel_window` (issue #381; pull #383)
- BUG: Fix negative start in row or col window offsets in `rio.isel_window` (issue #381; pull #383)

0.5.0
------
- ENH: Allow passing in kwargs to `rio.reproject` (issue #369; pull #370)
- ENH: Allow nodata override and provide default nodata based on dtype in `rio.reproject` (pull #370)
- ENH: Add support for passing in gcps to rio.reproject (issue #339; pull #370)
- BUG: Remove duplicate acquire in open_rasterio (pull #364)
- BUG: Fix exporting dataset to raster with non-standard dimensions (issue #372)

0.4.3
------
- BUG: support GDAL CInt16, rasterio complex_int16 (pull #353)
- TST: Fix merge tests for rasterio 1.2.5+ (issue #358)

0.4.2
------
- BUG: Improve WarpedVRT support for gcps (pull #351)

0.4.1
------
- BUG: pass kwargs with lock=False (issue #344)
- BUG: Close file handle with lock=False (pull #346)

0.4.0
------
- DEP: Python 3.7+ (issue #215)
- DEP: xarray 0.17+ (needed for issue #282)
- REF: Store `grid_mapping` in `encoding` instead of `attrs` (issue #282)
- ENH: enable `engine="rasterio"` via xarray backend API (issue #197 pull #281)
- ENH: Generate 2D coordinates for non-rectilinear sources (issue #290)
- ENH: Add `encoded` kwarg to `rio.write_nodata` (discussions #313)
- ENH: Added `decode_times` and `decode_timedelta` kwargs to `rioxarray.open_rasterio` (issue #316)
- BUG: Use float32 for smaller dtypes when masking (discussions #302)
- BUG: Return correct transform in `rio.transform` with non-rectilinear transform (discussions #280)
- BUG: Update to handle WindowError in rasterio 1.2.2 (issue #286)
- BUG: Don't generate x,y coords in `rio` methods if not previously there (pull #294)
- BUG: Preserve original data type for writing to disk (issue #305)
- BUG: handle lock=True in open_rasterio (issue #273)

0.3.1
------
- BUG: Compatibility changes with xarray 0.17 (issue #254)
- BUG: Raise informative error in interpolate_na if missing nodata (#250)

0.3.0
------
- REF: Reduce pyproj.CRS internal usage for speed (issue #241)
- ENH: Add `rioxarray.set_options` to disable exporting CRS CF grid mapping (issue #241)
- BUG: Handle merging 2D DataArray (discussion #244)

0.2.0
------
- ENH: Added `rio.estimate_utm_crs` (issue #181)
- ENH: Add support for merging datasets with different CRS (issue #173)
- ENH: Add support for using dask in `rio.to_raster` (issue #9, pull #219, pull #223)
- ENH: Use the list version of `transform_geom` with rasterio 1.2+ (issue #180)
- ENH: Support driver autodetection with rasterio 1.2+ (issue #180)
- ENH: Allow multithreaded, lockless reads with `rioxarray.open_rasterio` (issue #214)
- ENH: Add support to clip from disk (issue #115)
- BUG: Allow `rio.write_crs` when spatial dimensions not found (pull #186)
- BUG: Update to support rasterio 1.2+ merge (issue #180)

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
