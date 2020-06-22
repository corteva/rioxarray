================
rioxarray README
================

rasterio xarray extension.


.. image:: https://badges.gitter.im/rioxarray/community.svg
   :alt: Join the chat at https://gitter.im/rioxarray/community
   :target: https://gitter.im/rioxarray/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://img.shields.io/badge/all_contributors-10-orange.svg?style=flat-square
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

.. image:: https://travis-ci.com/corteva/rioxarray.svg?branch=master
    :target: https://travis-ci.com/corteva/rioxarray

.. image:: https://ci.appveyor.com/api/projects/status/e6sr22mkpen261c1/branch/master?svg=true
    :target: https://ci.appveyor.com/project/snowman2/rioxarray

.. image:: https://coveralls.io/repos/github/corteva/rioxarray/badge.svg?branch=master
    :target: https://coveralls.io/github/corteva/rioxarray?branch=master

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/python/black


Documentation
-------------

- Stable: https://corteva.github.io/rioxarray/stable/
- Latest: https://corteva.github.io/rioxarray/latest/

Bugs/Questions
--------------

- Report bugs/ask questions: https://github.com/corteva/rioxarray/issues
- Ask developer questions: https://gitter.im/rioxarray/community
- Ask the community: https://gis.stackexchange.com/questions/tagged/rioxarray

Credits
-------

The *reproject* functionality was adopted from https://github.com/opendatacube/datacube-core
  - Source file: `geo_xarray.py <https://github.com/opendatacube/datacube-core/blob/084c84d78cb6e1326c7fbbe79c5b5d0bef37c078/datacube/api/geo_xarray.py>`_
  - `datacube is licensed <https://github.com/opendatacube/datacube-core/blob/1d345f08a10a13c316f81100936b0ad8b1a374eb/LICENSE>`_ under the Apache License, Version 2.0.
    The datacube license is included as `LICENSE_datacube <https://github.com/corteva/rioxarray/blob/master/LICENSE_datacube>`_.

The *open_rasterio* functionality was adopted from https://github.com/pydata/xarray
  - Source file: `rasterio_.py <https://github.com/pydata/xarray/blob/1d7bcbdc75b6d556c04e2c7d7a042e4379e15303/xarray/backends/rasterio_.py>`_
  - `xarray is licensed <https://github.com/pydata/xarray/blob/1d7bcbdc75b6d556c04e2c7d7a042e4379e15303/LICENSE>`_ under the Apache License, Version 2.0.
    The xarray license is included as `LICENSE_xarray <https://github.com/corteva/rioxarray/blob/master/LICENSE_xarray>`_.


This package was originally templated with with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
