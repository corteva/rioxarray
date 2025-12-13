Geospatial Metadata Conventions
==============================

Overview
--------

rioxarray supports two geospatial metadata conventions for storing coordinate reference system (CRS) and transform information:

1. **CF (Climate and Forecasts) Convention** - NetCDF convention using grid_mapping coordinates
2. **Zarr Spatial and Proj Conventions** - Cloud-native conventions using direct attributes

Convention Selection
--------------------

You can choose which convention to use in several ways:

Global Setting
~~~~~~~~~~~~~~

Set the default convention globally using ``set_options``:

.. code-block:: python

    import rioxarray
    from rioxarray import Convention

    # Use CF convention (default)
    rioxarray.set_options(convention=Convention.CF)

    # Use Zarr conventions
    rioxarray.set_options(convention=Convention.Zarr)

Per-Method Override
~~~~~~~~~~~~~~~~~~~

Override the global setting for individual method calls:

.. code-block:: python

    # Write CRS using CF convention regardless of global setting
    data.rio.write_crs("EPSG:4326", convention=Convention.CF)

    # Write transform using Zarr convention regardless of global setting
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

    # Or write all conventions at once
    data_zarr = data.rio.write_zarr_conventions("EPSG:4326")

Zarr-Specific Methods
---------------------

Additional methods are available specifically for Zarr conventions:

write_zarr_crs()
~~~~~~~~~~~~~~~~

Write CRS information using the Zarr proj: convention:

.. code-block:: python

    # Write as WKT2 string (default)
    data.rio.write_zarr_crs("EPSG:4326")

    # Write as EPSG code
    data.rio.write_zarr_crs("EPSG:4326", format="code")

    # Write as PROJJSON object
    data.rio.write_zarr_crs("EPSG:4326", format="projjson")

    # Write all formats for maximum compatibility
    data.rio.write_zarr_crs("EPSG:4326", format="all")

write_zarr_transform()
~~~~~~~~~~~~~~~~~~~~~~

Write transform information using the Zarr spatial: convention:

.. code-block:: python

    from affine import Affine

    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    data.rio.write_zarr_transform(transform)

    # Results in spatial:transform attribute: [1.0, 0.0, 0.0, 0.0, -1.0, 100.0]

write_zarr_spatial_metadata()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write complete spatial metadata using the Zarr spatial: convention:

.. code-block:: python

    data.rio.write_zarr_spatial_metadata(
        include_bbox=True,      # Include spatial:bbox
        include_registration=True  # Include spatial:registration
    )

    # Results in:
    # - spatial:dimensions: ["y", "x"]
    # - spatial:shape: [height, width]
    # - spatial:bbox: [xmin, ymin, xmax, ymax]
    # - spatial:registration: "pixel"

write_zarr_conventions()
~~~~~~~~~~~~~~~~~~~~~~~~

Convenience method to write both CRS and spatial conventions:

.. code-block:: python

    # Write complete Zarr metadata in one call
    data.rio.write_zarr_conventions(
        input_crs="EPSG:4326",
        crs_format="all",  # Write code, wkt2, and projjson
        transform=my_transform
    )

Reading Behavior
----------------

When reading geospatial metadata, rioxarray follows the global convention setting:

- **Convention.CF**: Reads from grid_mapping coordinates and CF attributes
- **Convention.Zarr**: Reads from Zarr spatial: and proj: attributes

The reading logic is strict - it only attempts to read from the specified convention, ensuring predictable behavior.

Convention Declaration
----------------------

According to the `Zarr conventions specification <https://github.com/zarr-conventions/zarr-conventions-spec>`, conventions must be explicitly declared in the ``zarr_conventions`` array. rioxarray automatically handles this when writing Zarr conventions:

.. code-block:: python

    data_zarr = data.rio.write_zarr_crs("EPSG:4326")

    # Automatically adds to zarr_conventions:
    print(data_zarr.attrs["zarr_conventions"])
    # [{"name": "proj:", "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f", ...}]

References
----------

- `CF Conventions <https://github.com/cf-convention/cf-conventions>`_
- `Zarr Spatial Convention <https://github.com/zarr-conventions/spatial>`_
- `Zarr Geo-Proj Convention <https://github.com/zarr-experimental/geo-proj>`_
- `Zarr Conventions Specification <https://github.com/zarr-conventions/zarr-conventions-spec>`_
