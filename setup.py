"""The setup script."""
from pathlib import Path

from setuptools import setup


def get_version():
    """
    retreive rioxarray version information in version variable
    (taken from pyproj)
    """
    with Path("rioxarray", "_version.py").open() as vfh:
        for line in vfh:
            if line.find("__version__") >= 0:
                # parse __version__ and remove surrounding " or '
                return line.split("=", maxsplit=2)[1].strip()[1:-1]
    raise SystemExit("ERROR: rioxarray version not found.")


setup(version=get_version())
