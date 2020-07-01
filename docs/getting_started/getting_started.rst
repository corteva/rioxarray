.. _getting_started:

Getting Started
================

Welcome! This page aims to help you gain a foundational understanding of rioxarray.

You can learn how to `clip`, `merge`, and `reproject` rasters in the :ref:`usage_examples`
section of the documentation. Need to export to a raster (GeoTiff)? There is an example for
that as well.

Why use :func:`rioxarray.open_rasterio` instead of `xarray.open_rasterio`?

1. It supports multidimensional datasets such as netCDF.
2. It stores the CRS as a WKT, which is the recommended format (`PROJ FAQ <https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems>`__).
3. It loads in the CRS, transform, and nodata metadata in standard CF & GDAL locations.
4. It supports masking and scaling data with the `masked` and `mask_and_scale` kwargs.
5. It adds the coordinate axis CF metadata.
6. It loads in raster metadata into the attributes.


.. toctree::
   :maxdepth: 1
   :caption: Data management:

   crs_management.ipynb
   nodata_management.ipynb
   manage_information_loss.ipynb
