# -*- coding: utf-8 -*-
"""The setup script."""

from itertools import chain

from setuptools import find_packages, setup

from rioxarray import __version__

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["rasterio", "scipy", "xarray"]

test_requirements = ["pytest>=3.6", "pytest-cov", "mock"]

extras_require = {
    "dev": test_requirements
    + ["sphinx-click==1.1.0", "nbsphinx", "black", "flake8", "pylint", "isort"]
}
extras_require["all"] = list(chain.from_iterable(extras_require.values()))

setup(
    author="rioxarray Contributors",
    author_email="alansnow21@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
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
    version=__version__,
    zip_safe=False,
)
