# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nose2.tools.decorators import with_setup as _nose2_with_setup, with_teardown as _nose2_with_teardown
from unittest import SkipTest  # noqa: F401
import unittest
import re
import fnmatch
import functools
import warnings


def with_setup(setup=None, teardown=None):
    """
    Decorator to add setup and/or teardown functions to a test function.
    Compatible with nose's with_setup(setup, teardown) signature.

    Usage:
        @with_setup(setup_func)
        @with_setup(setup_func, teardown_func)
        @with_setup(teardown=teardown_func)
    """
    def decorator(func):
        if setup is not None:
            func = _nose2_with_setup(setup)(func)
        if teardown is not None:
            func = _nose2_with_teardown(teardown)(func)
        return func
    return decorator


def with_teardown(teardown):
    """Decorator to add teardown function to a test function."""
    return _nose2_with_teardown(teardown)


def attr(*tags):
    """Set test attributes for nose2 filtering with -A flag.

    Usage: @attr("pytorch", "slow")
    Filtering: nose2 -A 'pytorch' or nose2 -A '!slow'
    """
    def decorator(func):
        for tag in tags:
            setattr(func, tag, True)
        return func
    return decorator


def nottest(func):
    """Mark function as not a test."""
    func.__test__ = False
    return func


class empty_case(unittest.TestCase):
    def nop():
        pass


# Module-level TestCase instance for assertions
_test_case = unittest.TestCase()
_test_case.maxDiff = None  # Show full diff on assertion failures


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
    Wrapper combining unittest assertRaises and assertRaisesRegex.
    Specify ``regex=pattern`` or ``glob=pattern`` to check error message of expected exception
    against the pattern.
    Value for `glob` must be a string, `regex` can be either a literal or compiled regex pattern.
    By default, the check will ignore case, if called with `glob` or a literal for `regex`.
    To enforce case sensitive check pass ``match_case=True``.
    Don't specify `match_case` if passing already compiled regex pattern.

    Can be used as context manager or with callable:
        with assert_raises(Exception):
            raise Exception()

        assert_raises(Exception, callable, arg1, arg2, kwarg=value)
    """
    if glob is None and regex is None:
        # Use unittest's assertRaises
        if args:
            # Called with callable: assert_raises(Exception, callable, *args, **kwargs)
            callable_func = args[0]
            callable_args = args[1:]
            with _test_case.assertRaises(exception):
                callable_func(*callable_args, **kwargs)
        else:
            # Used as context manager
            return _test_case.assertRaises(exception)
    else:
        pattern = get_pattern(glob, regex, match_case)
        # Use unittest's assertRaisesRegex
        if args:
            # Called with callable
            callable_func = args[0]
            callable_args = args[1:]
            with _test_case.assertRaisesRegex(exception, pattern):
                callable_func(*callable_args, **kwargs)
        else:
            # Used as context manager
            return _test_case.assertRaisesRegex(exception, pattern)


def assert_warns(exception=Warning, *args, glob=None, regex=None, match_case=None, **kwargs):
    """
    Wrapper for asserting warnings, optionally with pattern matching.

    Can be used as context manager or with callable:
        with assert_warns(UserWarning):
            warnings.warn("test", UserWarning)

        assert_warns(UserWarning, callable, arg1, arg2, kwarg=value)
    """
    if glob is None and regex is None:
        # Use unittest's assertWarns
        if args:
            # Called with callable
            callable_func = args[0]
            callable_args = args[1:]
            with _test_case.assertWarns(exception):
                callable_func(*callable_args, **kwargs)
        else:
            # Used as context manager
            return _test_case.assertWarns(exception)
    else:
        pattern = get_pattern(glob, regex, match_case)
        # Use unittest's assertWarnsRegex
        if args:
            # Called with callable
            callable_func = args[0]
            callable_args = args[1:]
            with _test_case.assertWarnsRegex(exception, pattern):
                callable_func(*callable_args, **kwargs)
        else:
            # Used as context manager
            return _test_case.assertWarnsRegex(exception, pattern)


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
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            with assert_raises(exception, glob=glob, regex=regex, match_case=match_case):
                return func(*args, **kwargs)

        return new_func

    return decorator
