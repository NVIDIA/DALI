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

from typing import Any, Optional, Union, List, Sequence, Callable

from nvidia.dali.data_node import DataNode

from nvidia.dali.types import DALIDataType
from nvidia.dali._typing import TensorLikeIn

def numba_function(
    __input_0: Union[DataNode, TensorLikeIn],
    __input_1: Union[DataNode, TensorLikeIn, None] = None,
    __input_2: Union[DataNode, TensorLikeIn, None] = None,
    __input_3: Union[DataNode, TensorLikeIn, None] = None,
    __input_4: Union[DataNode, TensorLikeIn, None] = None,
    __input_5: Union[DataNode, TensorLikeIn, None] = None,
    /,
    *,
    run_fn: Callable[..., None],
    out_types: List[DALIDataType],
    in_types: List[DALIDataType],
    outs_ndim: List[int],
    ins_ndim: List[int],
    setup_fn: Optional[Callable[[Sequence[Sequence[Any]], Sequence[Sequence[Any]], None]]] = None,
    batch_processing: bool = False,
    blocks: Optional[Sequence[int]] = None,
    threads_per_block: Optional[Sequence[int]] = None,
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    preserve: Optional[bool] = False,
    seed: Optional[int] = -1,
    device: Optional[str] = None,
    name: Optional[str] = None,
) -> Union[DataNode, Sequence[DataNode]]:
    """Invokes a njit compiled Numba function.

    The run function should be a Python function that can be compiled in Numba ``nopython`` mode."""
    ...
