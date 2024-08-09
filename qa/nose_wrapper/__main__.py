import sys

if sys.version_info >= (3, 12):
    from importlib import machinery, util
    from importlib._bootstrap import _exec, _load
    import modulefinder
    import types
    import unittest

    # the below are based on https://github.com/python/cpython/blob/3.11/Lib/imp.py
    # based on PSF license
    def find_module(name, path):
        return modulefinder.ModuleFinder(path).find_module(name, path)

    def load_module(name, file, filename, details):
        PY_SOURCE = 1
        PY_COMPILED = 2

        class _HackedGetData:
            """Compatibility support for 'file' arguments of various load_*()
            functions."""

            def __init__(self, fullname, path, file=None):
                super().__init__(fullname, path)
                self.file = file

            def get_data(self, path):
                """Gross hack to contort loader to deal w/ load_*()'s bad API."""
                if self.file and path == self.path:
                    # The contract of get_data() requires us to return bytes. Reopen the
                    # file in binary mode if needed.
                    file = None
                    if not self.file.closed:
                        file = self.file
                        if "b" not in file.mode:
                            file.close()
                    if self.file.closed:
                        self.file = file = open(self.path, "rb")

                    with file:
                        return file.read()
                else:
                    return super().get_data(path)

        class _LoadSourceCompatibility(_HackedGetData, machinery.SourceFileLoader):
            """Compatibility support for implementing load_source()."""

        _, mode, type_ = details
        if mode and (not mode.startswith("r") or "+" in mode):
            raise ValueError("invalid file open mode {!r}".format(mode))
        elif file is None and type_ in {PY_SOURCE, PY_COMPILED}:
            msg = "file object required for import (type code {})".format(type_)
            raise ValueError(msg)
        assert type_ == PY_SOURCE, "load_module replacement supports only PY_SOURCE file type"
        loader = _LoadSourceCompatibility(name, filename, file)
        spec = util.spec_from_file_location(name, filename, loader=loader)
        if name in sys.modules:
            module = _exec(spec, sys.modules[name])
        else:
            module = _load(spec)
        # To allow reloading to potentially work, use a non-hacked loader which
        # won't rely on a now-closed file object.
        module.__loader__ = machinery.SourceFileLoader(name, filename)
        module.__spec__.loader = module.__loader__
        return module

    def acquire_lock():
        pass

    def release_lock():
        pass

    context = {
        "find_module": find_module,
        "load_module": load_module,
        "acquire_lock": acquire_lock,
        "release_lock": release_lock,
    }
    imp_module = types.ModuleType("imp", "Mimics old imp module")
    imp_module.__dict__.update(context)
    sys.modules["imp"] = imp_module
    unittest._TextTestResult = unittest.TextTestResult

from nose.core import run_exit
import collections
import nose.case
import nose.inspector
import nose.loader
import nose.suite
import nose.plugins.attrib
import inspect

if sys.version_info >= (3, 10) and not hasattr(collections, "Callable"):
    nose.case.collections = collections.abc
    nose.inspector.collections = collections.abc
    nose.loader.collections = collections.abc
    nose.suite.collections = collections.abc
    nose.plugins.attrib.collections = collections.abc

if sys.version_info >= (3, 11):

    def legacy_getargspec(fun):
        args, varargs, varkw, defaults, *_ = inspect.getfullargspec(fun)
        return (args, varargs, varkw, defaults)

    inspect.getargspec = legacy_getargspec

if sys.argv[0].endswith("__main__.py"):
    sys.argv[0] = "%s -m nose_wrapper" % sys.executable

run_exit()
