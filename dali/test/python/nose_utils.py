# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nose.tools as tools
import re
import fnmatch


def assert_raises(exception, *args, glob=None, regex=None, match_case=None, **kwargs):

    """
    Wrapper combining `nose.tools.assert_raises` and `nose.tools.assert_raises_regex`.
    Specify ``regex=pattern`` or ``glob=pattern`` to check error message of expected exception against the pattern.
    Value for `glob` must be a string, `regex` can be either a literal or compiled regex pattern.
    By default, the check will ignore case, if called with `glob` or a literal for `regex`.
    To enforce case sensitive check pass ``match_case=True``.
    Don't specify `match_case` if passing already compiled regex pattern.
    """

    if glob is None and regex is None:
        return tools.assert_raises(exception, *args, **kwargs)

    if glob is not None and regex is not None:
        raise ValueError("You should specify at most one of `glob` and `regex` parameters but not both")

    if glob is not None:
        if not isinstance(glob, str):
            raise ValueError("Glob pattern must be a string")
        pattern = fnmatch.translate(glob)
        # fnmatch adds special character to match the end of the string, so that when used
        # with re.match it, by default, matches the whole string. Here, it's going to be used
        # with re.search so it would be weird to enforce matching the suffix, but not the prefix.
        if pattern[-2:] == r"\Z":
            pattern = pattern[:-2]
    else: # regex is not None
        if match_case is not None and not isinstance(regex, str):
            raise ValueError("Regex must be a string if `match_case` is specified when calling assert_raises_pattern")
        pattern = regex

    if isinstance(pattern, str) and not match_case: # ignore case by default
        pattern = re.compile(pattern, re.IGNORECASE)

    return tools.assert_raises_regex(exception, pattern, *args, **kwargs)


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
