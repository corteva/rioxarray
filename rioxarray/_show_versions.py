"""
Utility methods to print system info for debugging

adapted from :func:`sklearn.utils._show_versions`
which was adapted from :func:`pandas.show_versions`
"""
# pylint: disable=import-outside-toplevel
import importlib
import os
import platform
import sys
from typing import Dict


def _get_sys_info() -> Dict[str, str]:
    """System information
    Return
    ------
    sys_info : dict
        system and Python version information
    """
    blob = [
        ("python", sys.version.replace("\n", " ")),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_main_info() -> Dict[str, str]:
    """Get the main dependency information to hightlight.

    Returns
    -------
    proj_info: dict
        system GDAL information
    """
    import rasterio
    import xarray

    try:
        proj_data = os.pathsep.join(rasterio._env.get_proj_data_search_paths())
    except AttributeError:
        proj_data = None
    try:
        gdal_data = rasterio._env.get_gdal_data()
    except AttributeError:
        gdal_data = None

    blob = [
        ("rasterio", rasterio.__version__),
        ("xarray", xarray.__version__),
        ("GDAL", rasterio.__gdal_version__),
        ("GEOS", getattr(rasterio, "__geos_version__", None)),
        ("PROJ", getattr(rasterio, "__proj_version__", None)),
        ("PROJ DATA", proj_data),
        ("GDAL DATA", gdal_data),
    ]

    return dict(blob)


def _get_deps_info() -> Dict[str, str]:
    """Overview of the installed version of dependencies
    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = ["scipy", "pyproj"]

    def get_version(module):
        try:
            return module.__version__
        except AttributeError:
            return module.version

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = get_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = None

    return deps_info


def _print_info_dict(info_dict: Dict[str, str]) -> None:
    """Print the information dictionary"""
    for key, stat in info_dict.items():
        print(f"{key:>10}: {stat}")


def show_versions() -> None:
    """
    .. versionadded:: 0.0.26

    Print useful debugging information

    Example
    -------
    > python -c "import rioxarray; rioxarray.show_versions()"

    """
    import rioxarray  # pylint: disable=cyclic-import

    print(f"rioxarray ({rioxarray.__version__}) deps:")
    _print_info_dict(_get_main_info())
    print("\nOther python deps:")
    _print_info_dict(_get_deps_info())
    print("\nSystem:")
    _print_info_dict(_get_sys_info())
