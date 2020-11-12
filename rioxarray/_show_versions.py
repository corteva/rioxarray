"""
Utility methods to print system info for debugging

adapted from :func:`sklearn.utils._show_versions`
which was adapted from :func:`pandas.show_versions`
"""
# pylint: disable=import-outside-toplevel
import importlib
import platform
import sys


def _get_sys_info():
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


def _get_main_info():
    """Get the main dependency information to hightlight.

    Returns
    -------
    proj_info: dict
        system GDAL information
    """
    import rasterio
    import xarray

    blob = [
        ("rasterio", rasterio.__version__),
        ("xarray", xarray.__version__),
        ("GDAL", rasterio.__gdal_version__),
    ]

    return dict(blob)


def _get_deps_info():
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


def _print_info_dict(info_dict):
    """Print the information dictionary"""
    for key, stat in info_dict.items():
        print("{key:>10}: {stat}".format(key=key, stat=stat))


def show_versions():
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
