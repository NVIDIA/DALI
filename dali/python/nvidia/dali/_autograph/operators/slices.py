# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Operators specific to slicing operations."""

import collections

from nvidia.dali._autograph.utils import hooks

# TODO(mdan): Support extended slices.


class GetItemOpts(collections.namedtuple("GetItemOpts", ("element_dtype",))):
    pass


def get_item(target, i, opts):
    """The slice read operator (i.e. __getitem__).

    Note: it is unspecified whether target will be mutated or not. In general,
    if target is mutable (like Python lists), it will be mutated.

    Args:
      target: An entity that supports getitem semantics.
      i: Index to read from.
      opts: A GetItemOpts object.

    Returns:
      The read element.

    Raises:
      ValueError: if target is not of a supported type.
    """
    assert isinstance(opts, GetItemOpts)

    if hooks._DISPATCH.detect_overload_get_item(target):
        return hooks._DISPATCH.get_item(target, i)
    else:
        return _py_get_item(target, i)


def _py_get_item(target, i):
    """Overload of get_item that executes a Python list modification."""
    return target[i]


def set_item(target, i, x):
    """The slice write operator (i.e. __setitem__).

    Note: it is unspecified whether target will be mutated or not. In general,
    if target is mutable (like Python lists), it will be mutated.

    Args:
      target: An entity that supports setitem semantics.
      i: Index to modify.
      x: The new element value.

    Returns:
      Same as target, after the update was performed.

    Raises:
      ValueError: if target is not of a supported type.
    """

    if hooks._DISPATCH.detect_overload_set_item(target):
        return hooks._DISPATCH.set_item(target, i, x)
    else:
        return _py_set_item(target, i, x)


def _py_set_item(target, i, x):
    """Overload of set_item that executes a Python list modification."""
    target[i] = x
    return target
