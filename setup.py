# -*- coding: utf-8 -*-
"""The setup script."""

import os
import sys
from itertools import chain

from setuptools import find_packages, setup


def get_version():
    """
    retreive rioxarray version information in version variable
    (taken from pyproj)
    """
    with open(os.path.join("rioxarray", "_version.py")) as vfh:
        for line in vfh:
            if line.find("__version__") >= 0:
                # parse __version__ and remove surrounding " or '
                return line.split("=")[1].strip()[1:-1]
    sys.exit("ERROR: pyproj version not fount.")


with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["pillow", "rasterio", "scipy", "xarray", "pyproj>=2"]

test_requirements = ["pytest>=3.6", "pytest-cov", "mock"]

extras_require = {
    "dev": test_requirements
    + [
        "sphinx-click==1.1.0",
        "nbsphinx",
        "sphinx_rtd_theme",
        "black",
        "flake8",
        "pylint",
        "isort",
    ]
}
extras_require["all"] = list(chain.from_iterable(extras_require.values()))

setup(
    author="rioxarray Contributors",
    author_email="alansnow21@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="rasterio xarray extension.",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + "\n\n",
    include_package_data=True,
    keywords="rioxarray,xarray,rasterio",
    name="rioxarray",
    packages=find_packages(include=["rioxarray*"]),
    test_suite="test",
    tests_require=test_requirements,
    extras_require=extras_require,
    url="https://github.com/corteva/rioxarray",
    version=get_version(),
    zip_safe=False,
    python_requires=">=3.6",
)
