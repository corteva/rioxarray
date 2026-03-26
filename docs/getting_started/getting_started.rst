.. _getting_started:

Getting Started
================

Welcome! This page aims to help you gain a foundational understanding of rioxarray.

rio accessor
-------------

rioxarray `extends xarray <https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`__
with the `rio` accessor. The `rio` accessor is activated by importing rioxarray like so:

.. code-block:: python

    import rioxarray


You can learn how to `clip`, `merge`, and `reproject` rasters in the :ref:`usage_examples`
section of the documentation. Need to export to a raster (GeoTiff)? There is an example for
that as well.


Reading Files
-------------

xarray
~~~~~~~

Since `rioxarray` is an extension of `xarray`, you can load in files using the standard
`xarray` open methods. If you use one of xarray's open methods such as ``xarray.open_dataset``
to load netCDF files with the default engine, it is recommended to use `decode_coords="all"`.
This will load the grid mapping variable into coordinates for compatibility with rioxarray.

.. code-block:: python

    import xarray

    xds = xarray.open_dataset("file.nc", decode_coords="all")


rioxarray
~~~~~~~~~~

rioxarray 0.4+ enables passing `engine="rasterio"` to ``xarray.open_dataset``
and ``xarray.open_mfdataset`` for xarray 0.18+. This uses
:func:`rioxarray.open_rasterio` as the backend and always returns an ``xarray.Dataset``.

.. code-block:: python

    import xarray

    xds = xarray.open_dataset("my.tif", engine="rasterio")

You can also use :func:`rioxarray.open_rasterio`. This objects returned depend on
your input file type.

.. code-block:: python

    import rioxarray

    xds = rioxarray.open_rasterio("my.tif")


Why use :func:`rioxarray.open_rasterio` instead of `xarray.open_rasterio`?

1. It supports multidimensional datasets such as netCDF.
2. It stores the CRS as a WKT, which is the recommended format (`PROJ FAQ <https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems>`__).
3. It loads in the CRS, transform, and nodata metadata in standard CF & GDAL locations.
4. It supports masking and scaling data with the `masked` and `mask_and_scale` kwargs.
5. It adds the coordinate axis CF metadata.
6. It loads raster metadata into the attributes.
7. `xarray.open_rasterio` is deprecated (since v0.20.0)

rio string representation
--------------------------

The rio accessor has a string representation, this can help you check quickly the more relevant attributes of your raster:

.. code-block:: python

    import rioxarray

    xds = rioxarray.open_rasterio("my.tif")
    xds.rio

Wich gives here (and for DataArrays):

.. code-block::

    rioxarray.RasterDataArray: (y[latitude (m)]: 10, x[latitude (m)]: 10, z[band]: 1)
    Profile:
        count: 1
        crs: 32605
        dtype: float32
        nodata: 0.0
        transform:
            | 10.00, 0.00, 346860.00|
            | 0.00,-10.00, 6392220.00|
            | 0.00, 0.00, 1.00|
        height: 22547
        width: 21710
        blockxsize: 256
        blockysize: 256
        bounds: (346860.0, 6166750.0, 563960.0, 6392220.0)

For Datasets, it looks like:

.. code-block::

    rioxarray.RasterDataset
    Dimensions: y[latitude (m)]: 10, x[latitude (m)]: 10, z[time]: 2
    Data variables:
        blue	y[latitude (m)]: 10, x[latitude (m)]: 10, z[time]: 2
        green	y[latitude (m)]: 10, x[latitude (m)]: 10, z[time]: 2
    Profile:
        count: 2
        crs: 32722
        dtype: float64
        nodata: nan
        transform:
            | 3.00, 0.00, 466266.00|
            | 0.00,-3.00, 8084700.00|
            | 0.00, 0.00, 1.00|
        height: 10
        width: 10
        blockxsize: 10
        blockysize: 10
        bounds: (466266.0, 8084670.0, 466296.0, 8084700.0)

Introductory Information
--------------------------

.. toctree::
   :maxdepth: 1

   switching_from_rasterio
   crs_management.ipynb
   nodata_management.ipynb
   manage_information_loss.ipynb
   gdal_env
