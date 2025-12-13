Geospatial Metadata Conventions
==============================

Overview
--------

rioxarray supports two geospatial metadata conventions for storing coordinate reference system (CRS) and transform information:

1. **CF (Climate and Forecasts) Convention** - NetCDF convention using grid_mapping coordinates
2. **Zarr Spatial and Proj Conventions** - Cloud-native conventions using direct attributes

Convention Selection
--------------------

rioxarray uses CF conventions by default. When convention is set to ``None`` (the default), rioxarray uses CF conventions but will fallback to reading Zarr conventions if they are explicitly declared in the data.

Global Setting
~~~~~~~~~~~~~~

Set the default convention globally using ``set_options``:

.. code-block:: python

    import rioxarray
    from rioxarray import Convention

    # Use CF convention with Zarr fallback (default)
    rioxarray.set_options(convention=None)

    # Use CF conventions exclusively
    rioxarray.set_options(convention=Convention.CF)

    # Use Zarr conventions exclusively
    rioxarray.set_options(convention=Convention.Zarr)

Per-Method Override
~~~~~~~~~~~~~~~~~~~

Override the global setting for individual method calls:

.. code-block:: python

    # Write CRS using CF convention (default)
    data.rio.write_crs("EPSG:4326")

    # Write CRS using Zarr convention
    data.rio.write_crs("EPSG:4326", convention=Convention.Zarr)

    # Write transform using Zarr convention
    data.rio.write_transform(transform, convention=Convention.Zarr)

CF Convention
-------------

The CF (Climate and Forecasts) convention:

- CRS information stored in a grid_mapping coordinate variable
- Transform information stored as ``GeoTransform`` attribute on the grid_mapping coordinate
- Compatible with NetCDF and GDAL tools
- Verbose but widely supported

Example:

.. code-block:: python

    import rioxarray
    from rioxarray import Convention

    # Write using CF convention
    data_cf = data.rio.write_crs("EPSG:4326", convention=Convention.CF)
    data_cf = data_cf.rio.write_transform(transform, convention=Convention.CF)

    # Results in:
    # - Grid mapping coordinate with CRS attributes
    # - GeoTransform attribute with space-separated transform values

Zarr Conventions
----------------

The Zarr spatial and proj conventions provide a cloud-native approach:

- CRS information stored as direct attributes (``proj:code``, ``proj:wkt2``, ``proj:projjson``)
- Transform stored as ``spatial:transform`` numeric array attribute
- Spatial metadata in ``spatial:dimensions``, ``spatial:shape``, ``spatial:bbox``
- Lightweight and efficient for cloud storage

Example:

.. code-block:: python

    import rioxarray
    from rioxarray import Convention

    # Write using Zarr conventions
    data_zarr = data.rio.write_crs("EPSG:4326", convention=Convention.Zarr)
    data_zarr = data_zarr.rio.write_transform(transform, convention=Convention.Zarr)

    # Write both CRS and transform using Zarr conventions
    data_zarr = data.rio.write_crs("EPSG:4326", convention=Convention.Zarr)
    data_zarr = data_zarr.rio.write_transform(transform, convention=Convention.Zarr)

Writing Zarr Conventions
------------------------

To write data using Zarr conventions, use the ``convention`` parameter:

.. code-block:: python

    from affine import Affine
    from rioxarray import Convention

    # Write CRS using Zarr conventions
    data = data.rio.write_crs("EPSG:4326", convention=Convention.Zarr)

    # Write transform using Zarr conventions
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    data = data.rio.write_transform(transform, convention=Convention.Zarr)

    # Results in:
    # - proj:wkt2: CRS as WKT2 string
    # - spatial:transform: [1.0, 0.0, 0.0, 0.0, -1.0, 100.0]
    # - spatial:dimensions: ["y", "x"]
    # - spatial:shape: [height, width]
    # - zarr_conventions: Convention declarations

Reading Behavior
----------------

When reading geospatial metadata, rioxarray follows this priority order based on the global convention setting:

- **None (default)**: CF conventions first, with Zarr conventions as fallback if explicitly declared
- **Convention.CF**: CF conventions only (grid_mapping coordinates and CF attributes)
- **Convention.Zarr**: Zarr conventions only (spatial: and proj: attributes)

The fallback behavior ensures that CF remains the primary convention while allowing Zarr conventions to be read when they are the only available metadata.

Convention Declaration
----------------------

According to the `Zarr conventions specification <https://github.com/zarr-conventions/zarr-conventions-spec>`, conventions must be explicitly declared in the ``zarr_conventions`` array. rioxarray automatically handles this when writing Zarr conventions:

.. code-block:: python

    data_zarr = data.rio.write_crs("EPSG:4326", convention=Convention.Zarr)

    # Automatically adds to zarr_conventions:
    print(data_zarr.attrs["zarr_conventions"])
    # [{"name": "proj:", "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f", ...}]

References
----------

- `CF Conventions <https://github.com/cf-convention/cf-conventions>`_
- `Zarr Spatial Convention <https://github.com/zarr-conventions/spatial>`_
- `Zarr Geo-Proj Convention <https://github.com/zarr-experimental/geo-proj>`_
- `Zarr Conventions Specification <https://github.com/zarr-conventions/zarr-conventions-spec>`_
