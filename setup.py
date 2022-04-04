"""The setup script."""

import os
import sys

from setuptools import setup


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
    sys.exit("ERROR: rioxarray version not fount.")


setup(version=get_version())
