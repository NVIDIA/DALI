# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import enum
from typing import Any, Protocol, Sequence, TypeAlias, Union


class ArrayInterface(Protocol):
    """
    Protocol representing classes that are compatible with NumPy array interface.
    Such classes can be used to represent multi-dimensional arrays - that is tensors.
    DALI can accept NumPy, PyTorch and Apache MXNet tensor objects as a constant parameter
    to its operators. Such parameter would be broadcast for all samples in the batch.
    """

    def __array__(self) -> Any: ...


class CudaArrayInterface(Protocol):
    """
    Protocol representing classes that are compatible with Numba CUDA Array Interface.
    Such classes can be used to represent multi-dimensional arrays - that is tensors, residing
    on the GPU memory. DALI can accept such objects as data source for External Source operator.
    """

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]: ...


class DLPack(Protocol):
    """
    Protocol representing classes that are compatible with DLPack interface.
    See: https://dmlc.github.io/dlpack/latest/python_spec.html.
    """

    def __dlpack__(
        self,
        *,
        stream: int | Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[Any, int] | None = None,
        copy: bool | None = None,
    ) -> Any: ...  # types.PyCapsule is only available since Python 3.13

    def __dlpack_device__(self) -> tuple[enum.Enum, int]: ...


TensorLikeIn: TypeAlias = Union[ArrayInterface, Sequence[int], Sequence[float], int, float]
"""
Constant input to the operator, that is expressed by a single tensor. Such input represents
one sample that is repeated (broadcast) to form a batch.
"""


TensorLikeArg: TypeAlias = ArrayInterface | DLPack
"""
Constant argument to the operator, that is expressed by a single tensor. Such input represents
one sample that is repeated (broadcast) to form a batch.
"""

TensorLike: TypeAlias = ArrayInterface | CudaArrayInterface | DLPack
"""
Object compatible with ``dali.dynamic.Tensor`` used as input for per-sample
dynamic mode functions.
"""
