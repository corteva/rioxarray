.. _gdal_env:

Configure GDAL environment
==========================

``rioxarray`` relies on ``rasterio`` so setting up GDAL environment stays the same. See ``rasterio``'s `documentation <https://rasterio.readthedocs.io/en/latest/topics/configuration.html#rasterio>`__ for more insights.

Setting up GDAL environment is very useful when working with cloud-stored data.
You can find Development Seed's TiTiler environment proposition `here <https://developmentseed.org/titiler/advanced/performance_tuning/#recommended-configuration-for-dynamic-tiling>`__.

With Dask clusters
~~~~~~~~~~~~~~~~~~

When setting up a Dask cluster, be sure to pass the GDAL environment (and AWS session) to every workers.
First create a function setting up the env and then submit it to every worker.

.. code-block:: python

    import os
    from dask.distributed import Client


    def set_env():
        # Set that to dask workers to make process=True work on cloud
        os.environ["AWS_S3_ENDPOINT"] = os.getenv("AWS_S3_ENDPOINT")
        os.environ["AWS_S3_AWS_ACCESS_KEY_ID"] = os.getenv("AWS_S3_AWS_ACCESS_KEY_ID")
        os.environ["AWS_S3_AWS_SECRET_ACCESS_KEY"] = os.getenv(
            "AWS_S3_AWS_SECRET_ACCESS_KEY"
        )
        os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".vrt"


    # Run the client
    with Client(processes=True) as client:

        # Propagate the env variables
        client.run(set_env)
        ...


.. note::

  There are gotchas with the environment and dask workers, see this `discussion <https://github.com/corteva/rioxarray/discussions/630>`__ for more insights.
