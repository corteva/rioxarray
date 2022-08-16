rioxarray package
=================

rioxarray.open_rasterio
-----------------------

.. autofunction:: rioxarray.open_rasterio


rioxarray.merge module
----------------------

.. autofunction:: rioxarray.merge.merge_arrays

.. autofunction:: rioxarray.merge.merge_datasets


rioxarray.set_options
-----------------------

.. autoclass:: rioxarray.set_options


rioxarray.show_versions
-----------------------

.. autofunction:: rioxarray.show_versions


rioxarray `rio` accessors
--------------------------

rioxarray `extends xarray <http://xarray.pydata.org/en/stable/internals/extending-xarray.html>`__
with the `rio` accessor. The `rio` accessor is activated by importing rioxarray like so:

.. code-block:: python

    import rioxarray


.. autoclass:: rioxarray.rioxarray.XRasterBase
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: rioxarray.raster_array.RasterArray
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: rioxarray.raster_dataset.RasterDataset
    :members:
    :undoc-members:
    :show-inheritance:


rioxarray.exceptions module
---------------------------

.. automodule:: rioxarray.exceptions
    :members:
    :undoc-members:
    :show-inheritance:
