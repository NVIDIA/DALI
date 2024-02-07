# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from typing import Sequence
from typing import Any, Union, TypeAlias, Protocol


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
    def __cuda_array_interface__(self) -> Any: ...


TensorLikeIn: TypeAlias = Union[ArrayInterface, Sequence[int], Sequence[float], int, float]
"""
Constant input to the operator, that is expressed by a single tensor. Such input represents
one sample that is repeated (broadcast) to form a batch.
"""


TensorLikeArg: TypeAlias = ArrayInterface
"""
Constant argument to the operator, that is expressed by a single tensor. Such input represents
one sample that is repeated (broadcast) to form a batch.
"""
