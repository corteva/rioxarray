.. _switching_from_rasterio:

Switching from ``rasterio``
===========================

Reasons to switch from ``rasterio`` to ``rioxarray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usually, switching from ``rasterio`` to ``rioxarray`` means you are working with rasters and you have to adapt your code to ``xarray``.

``xarray`` is a powerful abstraction of both the raster dataset and the raster array. There is a lot of advantages to unite these two notions under the same object, as it simplifies the use of the functions, using attributes stored in the object rather than passing arguments to the functions.

``xarray`` comes also with a lot of very interesting built-in functions and can leverage several backends to replace ``numpy`` in cases where it is limiting (out-of-memory computation, running the code on clusters, on GPU...). Dask is one of the most well-knwown. ``rioxarray`` handles some basic ``dask`` features in I/O (see `Dask I/O example <https://corteva.github.io/rioxarray/html/examples/dask_read_write.html>`__) but is not designed to support ``dask`` in more evolved functions such as reproject.

Beware, ``xarray`` comes also with gotchas! You can see some of them in `the dedicated section <https://corteva.github.io/rioxarray/html/getting_started/manage_information_loss.html>`__.


  .. note::

    ``rasterio`` Dataset and xarray Dataset are two completely different things! Please be careful with these overlapping names.

Equivalences between ``rasterio`` and ``rioxarray``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ease the switch from ``rasterio`` and ``rioxarray``, here is a table of the usual parameters or functions used.

``ds`` stands for ``rasterio`` Dataset and ``xda`` for ``xarray`` DataArray.

Profile
-------

Here is the parameters that you can derive from ``rasterio``'s Dataset profile:

+----------------------------------+----------------------------------------------------------------------------------------------------------+
| ``rasterio`` from ``ds.profile`` | ``rioxarray`` from DataArray                                                                             |
+==================================+==========================================================================================================+
| blockxsize                       | ``.encoding["preferred_chunks"]["x"]``                                                                   |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| blockysize                       | ``.encoding["preferred_chunks"]["y"]``                                                                   |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| compress                         | *Unused in rioxarray*                                                                                    |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| count                            | :attr:`rioxarray.rioxarray.XRasterBase.count`                                                            |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| crs                              | :attr:`rioxarray.rioxarray.XRasterBase.crs`                                                              |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| driver                           | Unused in rioxarray                                                                                      |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| dtype                            | ``.encoding["rasterio_dtype"]``                                                                          |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| height                           | :attr:`rioxarray.rioxarray.XRasterBase.height`                                                           |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| interleave                       | Unused in rioxarray                                                                                      |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| nodata                           | :attr:`rioxarray.raster_array.RasterArray.nodata` (or `encoded_nodata <nodata_management.html>`_ )       |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| tiled                            | Unused in rioxarray                                                                                      |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| transform                        | :func:`rioxarray.rioxarray.XRasterBase.transform`                                                        |
+----------------------------------+----------------------------------------------------------------------------------------------------------+
| width                            | :attr:`rioxarray.rioxarray.XRasterBase.width`                                                            |
+----------------------------------+----------------------------------------------------------------------------------------------------------+

The values not used in ``rioxarray`` comes from the abstraction of the dataset in ``xarray``: a dataset no longer belongs to a file on disk even if read from it. The driver and other file-related notions are meaningless in this context.

Other dataset parameters
------------------------

+----------------------------------+----------------------------------------------------+
| ``rasterio`` from ``ds``         | ``rioxarray`` from DataArray                       |
+==================================+====================================================+
| gcps                             | :func:`rioxarray.rioxarray.XRasterBase.get_gcps`   |
+----------------------------------+----------------------------------------------------+
| rpcs                             | :func:`rioxarray.rioxarray.XRasterBase.get_rpcs`   |
+----------------------------------+----------------------------------------------------+
| bounds                           | :func:`rioxarray.rioxarray.XRasterBase.bounds`     |
+----------------------------------+----------------------------------------------------+

Functions
---------

+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``rasterio``                    | ``rioxarray``                                                                                                   |
+=================================+=================================================================================================================+
| ``rasterio.open()``             | :func:`rioxarray.open_rasterio` or :attr:`xarray.open_dataset(..., engine="rasterio", decode_coords="all")`     |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``ds.read()``                   | :func:`xarray.DataArray.compute` (load data into memory)                                                        |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``ds.read(... window=)``        | :func:`rioxarray.rioxarray.XRasterBase.isel_window`                                                             |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``ds.write()``                  | :func:`rioxarray.raster_array.RasterArray.to_raster`                                                            |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``mask.mask(..., crop=False)``  | :func:`rioxarray.raster_array.RasterArray.clip` with ``drop=False``                                             |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``mask.mask(..., crop=True)``   | :func:`rioxarray.raster_array.RasterArray.clip` with ``drop=False``                                             |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``warp.reproject()``            | :func:`rioxarray.raster_array.RasterArray.reproject()`                                                          |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``merge.merge()``               | :func:`rioxarray.merge.merge_arrays()`                                                                          |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``fill.fillnodata()``           | :func:`rioxarray.raster_array.RasterArray.interpolate_na()`                                                     |
+---------------------------------+-----------------------------------------------------------------------------------------------------------------+



By default, ``xarray`` is lazy and therefore not loaded into memory, hence the ``compute`` equivalent to ``read``.


Going back to ``rasterio``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``rioxarray`` 0.21+ enables recreating a ``rasterio`` Dataset from ``rioxarray``.
This is useful when translating your code from ``rasterio`` to ``rioxarray``, even if it is sub-optimal, because the array will be loaded and written in memory behind the hood.
It is always better to look for ``rioxarray``'s native functions.

.. code-block:: python

    with dataarray.rio.to_rasterio_dataset() as rio_ds:
        ...
