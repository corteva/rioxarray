from rioxarray._show_versions import (
    _get_deps_info,
    _get_main_info,
    _get_sys_info,
    show_versions,
)


def test_get_main_info():
    pyproj_info = _get_main_info()
    assert "rasterio" in pyproj_info
    assert "xarray" in pyproj_info
    assert "GDAL" in pyproj_info


def test_get_sys_info():
    sys_info = _get_sys_info()

    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


def test_get_deps_info():
    deps_info = _get_deps_info()

    assert "pillow" in deps_info
    assert "scipy" in deps_info
    assert "pyproj" in deps_info


def test_show_versions_with_proj(capsys):
    show_versions()
    out, err = capsys.readouterr()
    assert "System" in out
    assert "python" in out
    assert "GDAL" in out
    assert "rioxarray" in out
    assert "Other python deps" in out
