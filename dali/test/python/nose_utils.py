# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import collections

if sys.version_info >= (3, 12):
    # to make sure we can import anything from nose
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
import nose.case
import nose.inspector
import nose.loader
import nose.suite
import nose.plugins.attrib
from nose import SkipTest, with_setup  # noqa: F401
from nose.plugins.attrib import attr  # noqa: F401
from nose.tools import nottest  # noqa: F401

if sys.version_info >= (3, 10) and not hasattr(collections, "Callable"):
    nose.case.collections = collections.abc
    nose.inspector.collections = collections.abc
    nose.loader.collections = collections.abc
    nose.suite.collections = collections.abc
    nose.plugins.attrib.collections = collections.abc

import nose.tools as tools
import re
import fnmatch
import unittest


class empty_case(unittest.TestCase):
    def nop():
        pass


def assert_equals(x, y):
    foo = empty_case()
    foo.assertEqual(x, y)


def glob_to_regex(glob):
    if not isinstance(glob, str):
        raise ValueError("Glob pattern must be a string")
    pattern = fnmatch.translate(glob)
    # fnmatch adds special character to match the end of the string, so that when used
    # with re.match it, by default, matches the whole string. Here, it's going to be used
    # with re.search so it would be weird to enforce matching the suffix, but not the prefix.
    if pattern[-2:] == r"\Z":
        pattern = pattern[:-2]
    return pattern


def get_pattern(glob=None, regex=None, match_case=None):
    assert glob is not None or regex is not None

    if glob is not None and regex is not None:
        raise ValueError(
            "You should specify at most one of `glob` and `regex` parameters but not both"
        )

    if glob is not None:
        pattern = glob_to_regex(glob)
    else:  # regex is not None
        if match_case is not None and not isinstance(regex, str):
            raise ValueError(
                "Regex must be a string if `match_case` is specified when "
                "calling assert_raises_pattern"
            )
        pattern = regex

    if isinstance(pattern, str) and not match_case:  # ignore case by default
        pattern = re.compile(pattern, re.IGNORECASE)

    return pattern


def assert_raises(exception, *args, glob=None, regex=None, match_case=None, **kwargs):
    """
    Wrapper combining `nose.tools.assert_raises` and `nose.tools.assert_raises_regex`.
    Specify ``regex=pattern`` or ``glob=pattern`` to check error message of expected exception
    against the pattern.
    Value for `glob` must be a string, `regex` can be either a literal or compiled regex pattern.
    By default, the check will ignore case, if called with `glob` or a literal for `regex`.
    To enforce case sensitive check pass ``match_case=True``.
    Don't specify `match_case` if passing already compiled regex pattern.
    """

    if glob is None and regex is None:
        return tools.assert_raises(exception, *args, **kwargs)

    pattern = get_pattern(glob, regex, match_case)
    return tools.assert_raises_regex(exception, pattern, *args, **kwargs)


def assert_warns(exception=Warning, *args, glob=None, regex=None, match_case=None, **kwargs):
    if glob is None and regex is None:
        return tools.assert_warns(exception, *args, **kwargs)

    pattern = get_pattern(glob, regex, match_case)
    return tools.assert_warns_regex(exception, pattern, *args, **kwargs)


def raises(exception, glob=None, regex=None, match_case=None):
    """
    To assert that the test case raises Exception with the message matching given glob pattern
        @raises(Exception, "abc * def")
        def test():
            raise Exception("It's: abc 42 def, and has some suffix.")

    To assert that the test case raises Exception with the message matching given regex pattern
        @raises(Exception, regex="abc[0-9]{2}def")
        def test():
            raise Exception("It's: abc42def, and has some suffix too.")

    You can also use it like regular nose.raises
        @raises(Exception)
        def test():
            raise Exception("This message is not checked")

    By default, the check is not case-sensitive, to change that pass `match_case`=True.

    You can pass a tuple of exception classes to assert that the raised exception is
    an instance of at least one of the classes.
    """

    def decorator(func):
        def new_func(*args, **kwargs):
            with assert_raises(exception, glob=glob, regex=regex, match_case=match_case):
                return func(*args, **kwargs)

        return tools.make_decorator(func)(new_func)

    return decorator
