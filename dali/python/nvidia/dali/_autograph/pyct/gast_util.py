# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Gast compatibility library. Supports 0.2.2 and 0.3.2."""
# TODO(mdan): Remove this file once it's safe to break compatibility.

import functools
import gast

from packaging.version import Version


def convert_to_version(function):
    """Makes sure that returned function value is a Version object"""

    def wrap_function(*args, **kwargs):
        return Version(function(*args, **kwargs))

    return wrap_function


@convert_to_version
def get_gast_version():
    """Gast exports `__version__` from 0.5.3 onwards, we need to look it up in a different way."""
    if hasattr(gast, "__version__"):
        return gast.__version__
    try:
        import pkg_resources

        return pkg_resources.get_distribution("gast").version
    except pkg_resources.DistributionNotFound:
        # Older gast had 'Str', check for the oldest supported version
        if hasattr(gast, "Str"):
            return "0.2"
        else:
            try:
                # Try to call it with 3 arguments, to differentiate between 0.5+ and earlier.
                gast.Assign(None, None, None)
            except AssertionError as e:
                if "Bad argument number for Assign: 3, expecting 2" in str(e):
                    return "0.4"
            return "0.5"


def is_constant(node):
    """Tests whether node represents a Python constant."""
    return isinstance(node, gast.Constant)


def is_literal(node):
    """Tests whether node represents a Python literal."""
    # Normal literals, True/False/None/Etc. in Python3
    if is_constant(node):
        return True

    # True/False/None/Etc. in Python2
    if isinstance(node, gast.Name) and node.id in ["True", "False", "None"]:
        return True

    return False


def is_ellipsis(node):
    """Tests whether node represents a Python ellipsis."""
    return isinstance(node, gast.Constant) and node.value == Ellipsis


def _compat_assign_gast_4(targets, value, type_comment):
    """Wraps around gast.Assign to use same function signature across versions."""
    return gast.Assign(targets=targets, value=value)


def _compat_assign_gast_5(targets, value, type_comment):
    """Wraps around gast.Assign to use same function signature across versions."""
    return gast.Assign(targets=targets, value=value, type_comment=type_comment)


if get_gast_version() < Version("0.5"):
    compat_assign = _compat_assign_gast_4
else:
    compat_assign = _compat_assign_gast_5

Module = functools.partial(gast.Module, type_ignores=None)  # pylint:disable=invalid-name
Name = functools.partial(gast.Name, type_comment=None)  # pylint:disable=invalid-name
Str = functools.partial(gast.Constant, kind=None)  # pylint:disable=invalid-name
