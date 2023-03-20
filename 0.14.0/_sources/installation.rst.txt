.. highlight:: shell

============
Installation
============


Stable release
--------------

1. Use pip to install package from `PyPI <https://pypi.org/project/rioxarray/>`__:

  .. code-block:: bash

      pip install rioxarray


2. Use `conda <https://conda.io/en/latest/>`__ with the `conda-forge <https://conda-forge.org/>`__ channel:

  .. code-block:: bash

      conda config --prepend channels conda-forge
      conda config --set channel_priority strict
      conda create -n rioxarray_env rioxarray
      conda activate rioxarray_env

  - `rioxarray` `conda-forge repository <http://github.com/conda-forge/rioxarray-feedstock>`__

  .. note::
      "... we recommend always installing your packages inside a
      new environment instead of the base environment from
      anaconda/miniconda. Using envs make it easier to
      debug problems with packages and ensure the stability
      of your root env."
        -- https://conda-forge.org/docs/user/tipsandtricks.html

  .. warning::
      Avoid using `pip install` with a conda environment. If you encounter
      a python package that isn't in conda-forge, consider submitting a
      recipe: https://github.com/conda-forge/staged-recipes/


From source
-----------

The source for rioxarray can be installed from the `GitHub repo`_.

.. code-block:: bash

    python -m pip install git+git://github.com/corteva/rioxarray.git#egg=rioxarray


To install for local development:

.. code-block:: bash

    git clone git@github.com:corteva/rioxarray.git
    cd rioxarray
    python -m pip install -e .[dev]


.. _GitHub repo: https://github.com/corteva/rioxarray
