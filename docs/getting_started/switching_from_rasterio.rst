.. _switching_from_rasterio:

Switching from ``rasterio``
================

Reasons to switch from ``rasterio`` to ``rioxarray``
~~~~~~~~~~

Usually, switching from ``rasterio`` to ``rioxarray`` means you are working with rasters and you have to adapt your code to ``xarray``.

``xarray`` is a powerful abstraction of both the raster dataset and the raster array. There is a lot of advantages to unite these two notions under the same object, as it simplifies the use of the functions, using attributes stored in the object rather than passing arguments to the functions.

``xarray`` comes also with a lot of very interesting built-in functions and can leverage several backends to replace ``numpy`` in cases where it is limiting (out-of-memory computation, running the code on clusters, on GPU...). Dask is one of the most well-knwown. ``rioxarray`` handles some basic ``dask`` features in I/O (see `Dask I/O example <https://corteva.github.io/rioxarray/html/examples/dask_read_write.html>`__) but is not designed to support ``dask`` in more evolved functions such as reproject.

Beware, ``xarray`` comes also with gotchas! You can see some of them in `the dedicated section <https://corteva.github.io/rioxarray/html/getting_started/manage_information_loss.html>`__.

Note: ``rasterio`` Dataset and xarray Dataset are two completely different things! Please be careful with these overlapping names.

Equivalences between ``rasterio`` and ``rioxarray``
~~~~~~~~~~

To ease the switch from ``rasterio`` and ``rioxarray``, here is a table of the usual parameters or functions used.

Profile
-------

Here is the parameters that you can derive from ``rasterio``'s Dataset profile:

+----------------------------------+--------------------------------------------------+
| ``rasterio`` from ``ds.profile`` | ``rioxarray`` from DataArray                     |
+==================================+==================================================+
| blockxsize                       | ``xda.encoding["preferred_chunks"]["x"]``        |
+----------------------------------+--------------------------------------------------+
| blockysize                       | ``xda.encoding["preferred_chunks"]["y"]``        |
+----------------------------------+--------------------------------------------------+
| compress                         | Unused in rioxarray                              |
+----------------------------------+--------------------------------------------------+
| count                            | ``xda.rio.count``                                |
+----------------------------------+--------------------------------------------------+
| crs                              | ``xda.rio.crs``                                  |
+----------------------------------+--------------------------------------------------+
| driver                           | Unused in rioxarray                              |
+----------------------------------+--------------------------------------------------+
| dtype                            | ``xda.encoding["rasterio_dtype"]``               |
+----------------------------------+--------------------------------------------------+
| height                           | ``xda.rio.height``                               |
+----------------------------------+--------------------------------------------------+
| interleave                       | Unused in rioxarray                              |
+----------------------------------+--------------------------------------------------+
| nodata                           | ``xda.rio.encoded_nodata`` or ``xda.rio.nodata`` |
+----------------------------------+--------------------------------------------------+
| tiled                            | Unused in rioxarray                              |
+----------------------------------+--------------------------------------------------+
| transform                        | ``xda.rio.transform()``                          |
+----------------------------------+--------------------------------------------------+
| width                            | ``xda.rio.width``                                |
+----------------------------------+--------------------------------------------------+

The values not used in ``rioxarray`` comes from the abstraction of the dataset in ``xarray``: a dataset no longer belongs to a file on disk even if read from it. The driver and other file-related notions are meaningless in this context.

Functions
-------

+---------------------------------+----------------------------------------------+
| ``rasterio``                    | ``rioxarray``                                |
+=================================+==============================================+
| ``rasterio.open()``             | ``rioxarray.open_rasterio()``                |
+---------------------------------+----------------------------------------------+
| ``ds.read()``                   | ``xda.compute()`` (load data into memory)    |
+---------------------------------+----------------------------------------------+
| ``ds.read(... window=)``        | ``xda.rio.isel_window()``                    |
+---------------------------------+----------------------------------------------+
| ``ds.write()``                  | ``xda.rio.to_raster()``                      |
+---------------------------------+----------------------------------------------+
| ``mask.mask(..., crop=False)``  | ``xda.rio.clip(..., drop=False)``            |
+---------------------------------+----------------------------------------------+
| ``mask.mask(..., crop=True)``   | ``xda.rio.clip(..., drop=True)``             |
+---------------------------------+----------------------------------------------+
| ``warp.reproject()``            | ``xda.rio.reproject()``                      |
+---------------------------------+----------------------------------------------+
| ``merge.merge()``               | ``rioxarray.merge_arrays()``                 |
+---------------------------------+----------------------------------------------+
| ``fill.fillnodata()``           | ``xda.rio.interpolate_na()``                 |
+---------------------------------+----------------------------------------------+


By default, ``xarray`` is lazy and therefore not loaded into memory, hence the ``compute`` equivalent to ``read``.


Going back to ``rasterio``
~~~~~~~~~~

``rioxarray`` 0.21+ enables recreating a ``rasterio`` Dataset from ``rioxarray``.
This is useful when translating your code from ``rasterio`` to ``rioxarray``, even if it is sub-optimal, because the array will be loaded and written in memory behind the hood.
It is always better to look for ``rioxarray``'s native functions.

.. code-block:: python

    with dataarray.rio.to_rasterio_dataset() as rio_ds:
        ...
