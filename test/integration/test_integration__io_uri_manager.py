"""
Tests based on: https://github.com/pydata/xarray/blob/071da2a900702d65c47d265192bc7e424bb57932/xarray/tests/test_backends_file_manager.py
"""
import concurrent.futures
import gc
import pickle
from unittest import mock

import pytest

from rioxarray._io import URIManager


def test_uri_manager_mock_write():
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)

    manager = URIManager(opener, "filename")
    f = manager.acquire()
    f.write("contents")
    manager.close()

    opener.assert_called_once_with("filename", mode="r")
    mock_file.write.assert_called_once_with("contents")
    mock_file.close.assert_called_once_with()


def test_uri_manager_mock_write__threaded():
    mock_file = mock.Mock()
    opener = mock.Mock(spec=open, return_value=mock_file)

    manager = URIManager(opener, "filename")

    def write(iter):
        nonlocal manager
        fh = manager.acquire()
        fh.write("contents")
        manager._local.thread_manager = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for result in executor.map(write, range(5)):
            pass

    gc.collect()

    opener.assert_has_calls([mock.call("filename", mode="r") for _ in range(5)])
    mock_file.write.assert_has_calls([mock.call("contents") for _ in range(5)])
    mock_file.close.assert_has_calls([mock.call() for _ in range(5)])


@pytest.mark.parametrize("expected_warning", [None, RuntimeWarning])
def test_uri_manager_autoclose(expected_warning):
    mock_file = mock.Mock()
    opener = mock.Mock(return_value=mock_file)

    manager = URIManager(opener, "filename")
    manager.acquire()

    del manager
    gc.collect()

    mock_file.close.assert_called_once_with()


def test_uri_manager_write_concurrent(tmpdir):
    path = str(tmpdir.join("testing.txt"))
    manager = URIManager(open, path, mode="w")
    f1 = manager.acquire()
    f2 = manager.acquire()
    f3 = manager.acquire()
    assert f1 is f2
    assert f2 is f3
    f1.write("foo")
    f1.flush()
    f2.write("bar")
    f2.flush()
    f3.write("baz")
    f3.flush()

    del manager
    gc.collect()

    with open(path) as f:
        assert f.read() == "foobarbaz"


def test_uri_manager_write_pickle(tmpdir):
    path = str(tmpdir.join("testing.txt"))
    manager = URIManager(open, path, mode="a")
    f = manager.acquire()
    f.write("foo")
    f.flush()
    manager2 = pickle.loads(pickle.dumps(manager))
    f2 = manager2.acquire()
    f2.write("bar")
    del manager
    del manager2
    gc.collect()

    with open(path) as f:
        assert f.read() == "foobar"


def test_uri_manager_read(tmpdir):
    path = str(tmpdir.join("testing.txt"))

    with open(path, "w") as f:
        f.write("foobar")

    manager = URIManager(open, path)
    f = manager.acquire()
    assert f.read() == "foobar"
    manager.close()


def test_uri_manager_acquire_context(tmpdir):
    path = str(tmpdir.join("testing.txt"))

    with open(path, "w") as f:
        f.write("foobar")

    class AcquisitionError(Exception):
        pass

    manager = URIManager(open, path)
    with pytest.raises(AcquisitionError):
        with manager.acquire_context() as f:
            assert f.read() == "foobar"
            raise AcquisitionError

    with manager.acquire_context() as f:
        assert f.read() == "foobar"

    with pytest.raises(AcquisitionError):
        with manager.acquire_context() as f:
            f.seek(0)
            assert f.read() == "foobar"
            raise AcquisitionError
    manager.close()
