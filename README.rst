================
rioxarray README
================

rasterio xarray extension.


.. image:: https://badges.gitter.im/rioxarray/community.svg
   :alt: Join the chat at https://gitter.im/rioxarray/community
   :target: https://gitter.im/rioxarray/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://img.shields.io/badge/all_contributors-37-orange.svg?style=flat-square
    :alt: All Contributors
    :target: https://github.com/corteva/rioxarray/blob/master/AUTHORS.rst

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://github.com/corteva/rioxarray/blob/master/LICENSE

.. image:: https://img.shields.io/pypi/v/rioxarray.svg
    :target: https://pypi.python.org/pypi/rioxarray

.. image:: https://pepy.tech/badge/rioxarray
    :target: https://pepy.tech/project/rioxarray

.. image:: https://img.shields.io/conda/vn/conda-forge/rioxarray.svg
    :target: https://anaconda.org/conda-forge/rioxarray

.. image:: https://github.com/corteva/rioxarray/workflows/Tests/badge.svg
    :target: https://github.com/corteva/rioxarray/actions?query=workflow%3ATests

.. image:: https://ci.appveyor.com/api/projects/status/e6sr22mkpen261c1/branch/master?svg=true
    :target: https://ci.appveyor.com/project/snowman2/rioxarray

.. image:: https://codecov.io/gh/corteva/rioxarray/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/corteva/rioxarray

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/python/black

.. image:: https://zenodo.org/badge/181693881.svg
    :target: https://zenodo.org/badge/latestdoi/181693881


Documentation
-------------

- Stable: https://corteva.github.io/rioxarray/stable/
- Latest: https://corteva.github.io/rioxarray/latest/

Bugs/Questions
--------------

- Report bugs/feature requests: https://github.com/corteva/rioxarray/issues
- Ask questions: https://github.com/corteva/rioxarray/discussions
- Ask developer questions: https://gitter.im/rioxarray/community
- Ask questions from the GIS community: https://gis.stackexchange.com/questions/tagged/rioxarray

Credits
-------

The *reproject* functionality was adopted from https://github.com/opendatacube/datacube-core
  - Source file: `geo_xarray.py <https://github.com/opendatacube/datacube-core/blob/084c84d78cb6e1326c7fbbe79c5b5d0bef37c078/datacube/api/geo_xarray.py>`_
  - `datacube is licensed <https://github.com/opendatacube/datacube-core/blob/1d345f08a10a13c316f81100936b0ad8b1a374eb/LICENSE>`_ under the Apache License, Version 2.0.
    The datacube license is included as `LICENSE_datacube <https://github.com/corteva/rioxarray/blob/master/LICENSE_datacube>`_.

Adoptions from https://github.com/pydata/xarray:
  - *open_rasterio*: `rasterio_.py <https://github.com/pydata/xarray/blob/1d7bcbdc75b6d556c04e2c7d7a042e4379e15303/xarray/backends/rasterio_.py>`_
  - *set_options*: `options.py <https://github.com/pydata/xarray/blob/2ab0666c1fcc493b1e0ebc7db14500c427f8804e/xarray/core/options.py>`_
  - `xarray is licensed <https://github.com/pydata/xarray/blob/1d7bcbdc75b6d556c04e2c7d7a042e4379e15303/LICENSE>`_ under the Apache License, Version 2.0.
    The xarray license is included as `LICENSE_xarray <https://github.com/corteva/rioxarray/blob/master/LICENSE_xarray>`_.

RasterioWriter dask write functionality was adopted from https://github.com/dymaxionlabs/dask-rasterio
  - Source file: `write.py <https://github.com/dymaxionlabs/dask-rasterio/blob/8dd7fdece7ad094a41908c0ae6b4fe6ca49cf5e1/dask_rasterio/write.py>`_


This package was originally templated with with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
