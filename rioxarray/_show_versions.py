"""
Utility methods to print system info for debugging

adapted from :func:`sklearn.utils._show_versions`
which was adapted from :func:`pandas.show_versions`
"""
# pylint: disable=import-outside-toplevel
import importlib.metadata
import os
import platform
import sys


def _get_sys_info() -> dict[str, str]:
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


def _get_main_info() -> dict[str, str]:
    """Get the main dependency information to hightlight.

    Returns
    -------
    proj_info: dict
        system GDAL information
    """
    import rasterio

    try:
        proj_data = os.pathsep.join(rasterio._env.get_proj_data_search_paths())
    except AttributeError:
        proj_data = None
    try:
        gdal_data = rasterio._env.get_gdal_data()
    except AttributeError:
        gdal_data = None

    blob = [
        ("rasterio", importlib.metadata.version("rasterio")),
        ("xarray", importlib.metadata.version("xarray")),
        ("GDAL", rasterio.__gdal_version__),
        ("GEOS", getattr(rasterio, "__geos_version__", None)),
        ("PROJ", getattr(rasterio, "__proj_version__", None)),
        ("PROJ DATA", proj_data),
        ("GDAL DATA", gdal_data),
    ]

    return dict(blob)


def _get_deps_info() -> dict[str, str]:
    """Overview of the installed version of dependencies
    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = ["scipy", "pyproj"]

    def get_version(module):
        try:
            return importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {dep: get_version(dep) for dep in deps}


def _print_info_dict(info_dict: dict[str, str]) -> None:
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
    print(f"rioxarray ({importlib.metadata.version('rioxarray')}) deps:")
    _print_info_dict(_get_main_info())
    print("\nOther python deps:")
    _print_info_dict(_get_deps_info())
    print("\nSystem:")
    _print_info_dict(_get_sys_info())
