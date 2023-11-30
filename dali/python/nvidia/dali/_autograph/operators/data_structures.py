# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Operators specific to data structures: list append, subscripts, etc."""

import collections

from nvidia.dali._autograph.utils import hooks


# TODO(mdan): Once control flow supports objects, repackage as a class.


def new_list(iterable=None):
    """The list constructor.

    Args:
      iterable: Optional elements to fill the list with.

    Returns:
      A list-like object. The exact return value depends on the initial elements.
    """
    # TODO(klecki): DALI would fail here, as DataNode is explicitly non-convertible to boolean.
    # We probably need to revert this idiom.
    if hooks._DISPATCH.detect_overload_list_new(iterable):
        return hooks._DISPATCH.list_new(iterable)
    if iterable:
        elements = tuple(iterable)
    else:
        elements = ()

    if elements:
        # When the list contains elements, it is assumed to be a "Python" lvalue
        # list.
        return _py_list_new(elements)
    # Empty list creation
    return hooks._DISPATCH.list_new(elements)


def _py_list_new(elements):
    """Overload of new_list that creates a Python list."""
    return list(elements)


def list_append(list_, x):
    """The list append function.

    Note: it is unspecified where list_ will be mutated or not. If list_ is
    a TensorFlow entity, it will not be typically mutated. If list_ is a plain
    list, it will be. In general, if the list is mutated then the return value
    should point to the original entity.

    Args:
      list_: An entity that supports append semantics.
      x: The element to append.

    Returns:
      Same as list_, after the append was performed.

    Raises:
      ValueError: if list_ is not of a known list-like type.
    """
    if hooks._DISPATCH.detect_overload_list_append(list_):
        return hooks._DISPATCH.list_append(list_, x)
    else:
        return _py_list_append(list_, x)


def _py_list_append(list_, x):
    """Overload of list_append that executes a Python list append."""
    # Revert to the original call.
    list_.append(x)
    return list_


class ListPopOpts(collections.namedtuple("ListPopOpts", ("element_dtype", "element_shape"))):
    pass


def list_pop(list_, i, opts):
    """The list pop function.

    Note: it is unspecified where list_ will be mutated or not. If list_ is
    a TensorFlow entity, it will not be typically mutated. If list_ is a plain
    list, it will be. In general, if the list is mutated then the return value
    should point to the original entity.

    Args:
      list_: An entity that supports pop semantics.
      i: Optional index to pop from. May be None.
      opts: A ListPopOpts.

    Returns:
      Tuple (x, out_list_):
        out_list_: same as list_, after the removal was performed.
        x: the removed element value.

    Raises:
      ValueError: if list_ is not of a known list-like type or the operation is
      not supported for that type.
    """
    assert isinstance(opts, ListPopOpts)

    if hooks._DISPATCH.detect_overload_list_pop(list_):
        return hooks._DISPATCH.list_pop(list_, i)
    else:
        return _py_list_pop(list_, i)


def _py_list_pop(list_, i):
    """Overload of list_pop that executes a Python list append."""
    if i is None:
        x = list_.pop()
    else:
        x = list_.pop(i)
    return list_, x


# TODO(mdan): Look into reducing duplication between all these containers.
class ListStackOpts(collections.namedtuple("ListStackOpts", ("element_dtype", "original_call"))):
    pass


# TODO(klecki): Just remove this from code generation? It's TF-specific extension
def list_stack(list_, opts):
    """The list stack function.

    This does not have a direct correspondent in Python. The closest idiom to
    this is tf.append or np.stack. It's different from those in the sense that it
    accepts a Tensor list, rather than a list of tensors. It can also accept
    TensorArray. When the target is anything else, the dispatcher will rely on
    ctx.original_call for fallback.

    Args:
      list_: An entity that supports append semantics.
      opts: A ListStackOpts object.

    Returns:
      The output of the stack operation, typically a Tensor.
    """
    assert isinstance(opts, ListStackOpts)

    if hooks._DISPATCH.detect_overload_list_stack(list_):
        return hooks._DISPATCH.list_stack(list_, opts)
    else:
        return _py_list_stack(list_, opts)


def _py_list_stack(list_, opts):
    """Overload of list_stack that executes a Python list append."""
    # Revert to the original call.
    return opts.original_call(list_)
