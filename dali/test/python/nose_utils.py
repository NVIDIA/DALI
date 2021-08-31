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


def assert_raises_pattern(exception, pattern, *args, **kwargs):

    """
    There already is an assert_raises_regex function available in nose.tools
    (which is actually a method assertRaisesRegex copied from python's unittest.TestCase dummy instance).
    This is a simple wrapper around the function that allows to:
    * specify if the pattern searching should be case-sensitive (by default it is not, use `match_case`=True to change that)
    * use simple glob pattern instead of regex (specify `use_glob`=False if you want to use regex)

    `exception` can be either a class or list of classes
    """

    match_case = kwargs.pop("match_case", None)
    use_glob = kwargs.pop("use_glob", None)

    if not isinstance(pattern, str): # it could be an already compiled regex
        if use_glob is not None or match_case is not None:
            raise ValueError("Pattern must be a string if `match_case` or `use_glob` is specified when calling assert_raises_pattern")
    else:
        if use_glob is None:
            use_glob = True
        if match_case is None:
            match_case = False

        if use_glob:
            pattern = fnmatch.translate(pattern)
        flags = 0
        if not match_case:
            flags |= re.IGNORECASE
        pattern = re.compile(pattern, flags)

    return tools.assert_raises_regex(exception, pattern, *args, **kwargs)


def raises_pattern(exception, pattern, match_case=None, use_glob=None):

    def decorator(func):

        def new_func(*args, **kwargs):
            with assert_raises_pattern(exception, pattern, match_case=match_case, use_glob=use_glob):
                return func(*args, **kwargs)

        return tools.make_decorator(func)(new_func)

    return decorator
