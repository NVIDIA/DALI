# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Quantifiers - any, all
"""

from ._batch import _get_batch_size, as_batch
from ._tensor import as_tensor
from ._type import bool as _bool


def all(input, *, axes=None, axis_names=None, keep_dims=False):
    """
    Tests whether all elements along given axes evaluate to ``True``.

    The input is converted to ``bool`` first: any non-zero value (including NaN) is ``True``.
    An empty volume evaluates to ``True``.

    If the input is a batch, then the evaluation is performed on all constituent tensors
    and a batch of reduced tensors is returned.

    Parameters
    ----------
    input : TensorLike or Batch
        The data to test.
    axes : int or list of int, optional, default: None
        Axis or axes along which the reduction is performed. Negative indices are counted
        from the back. If neither `axes` nor `axis_names` is given, all axes are reduced.
    axis_names : str, optional, default: None
        Name(s) of the axis or axes along which the reduction is performed, translated to
        indices using the input layout. This argument cannot be used together with `axes`.
    keep_dims : bool, optional, default: False
        If ``True``, the reduced axes are kept in the output with extent 1.

    Returns
    -------
    Tensor or Batch
        The result of the test, of type ``bool``.
    """
    from . import reductions

    if _get_batch_size(input) is not None:
        return reductions.min(
            as_batch(input, dtype=_bool), axes=axes, axis_names=axis_names, keep_dims=keep_dims
        )
    else:
        return reductions.min(
            as_tensor(input, dtype=_bool), axes=axes, axis_names=axis_names, keep_dims=keep_dims
        )


def any(input, *, axes=None, axis_names=None, keep_dims=False):
    """
    Tests whether any elements along given axes evaluate to ``True``.

    The input is converted to ``bool`` first: any non-zero value (including NaN) is ``True``.
    An empty volume evaluates to ``False``.

    If the input is a batch, then the evaluation is performed on all constituent tensors
    and a batch of reduced tensors is returned.

    Parameters
    ----------
    input : TensorLike or Batch
        The data to test.
    axes : int or list of int, optional, default: None
        Axis or axes along which the reduction is performed. Negative indices are counted
        from the back. If neither `axes` nor `axis_names` is given, all axes are reduced.
    axis_names : str, optional, default: None
        Name(s) of the axis or axes along which the reduction is performed, translated to
        indices using the input layout. This argument cannot be used together with `axes`.
    keep_dims : bool, optional, default: False
        If ``True``, the reduced axes are kept in the output with extent 1.

    Returns
    -------
    Tensor or Batch
        The result of the test, of type ``bool``.
    """
    from . import reductions

    if _get_batch_size(input) is not None:
        return reductions.max(
            as_batch(input, dtype=_bool), axes=axes, axis_names=axis_names, keep_dims=keep_dims
        )
    else:
        return reductions.max(
            as_tensor(input, dtype=_bool), axes=axes, axis_names=axis_names, keep_dims=keep_dims
        )


__all__ = ["any", "all"]
