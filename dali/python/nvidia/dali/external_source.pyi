# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Union, Optional, overload
from typing import Sequence, Any, Callable, Iterable

from nvidia.dali.data_node import DataNode
from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

class ExternalSource:
    # The `source` parameter represents the Union of types accepted by the `fn.external_source`,
    # check the comment there for the explanation.
    def __init__(
        self,
        source: Optional[
            Union[
                Callable[..., Any],
                Iterable[Any],
                Callable[..., Sequence[Any]],
                Iterable[Sequence[Any]],
            ]
        ] = None,
        num_outputs: Optional[int] = None,
        *,
        batch: Optional[bool] = None,
        batch_info: Optional[bool] = False,
        dtype: Union[Sequence[DALIDataType], DALIDataType, None] = None,
        ndim: Union[Sequence[int], int, None] = None,
        layout: Union[Sequence[str], str, None] = None,
        name: Optional[str] = None,
        device: Optional[str] = "cpu",
        cuda_stream: Optional[Any] = None,
        use_copy_kernel: Optional[bool] = False,
        cycle: Union[str, bool, None] = None,
        repeat_last: Optional[bool] = False,
        parallel: Optional[bool] = False,
        no_copy: Optional[bool] = None,
        prefetch_queue_depth: Optional[int] = 1,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    ) -> None: ...
    def __call__(
        self,
        *,
        source: Optional[
            Union[
                Callable[..., Any],
                Iterable[Any],
                Callable[..., Sequence[Any]],
                Iterable[Sequence[Any]],
            ]
        ] = None,
        batch: Optional[bool] = None,
        batch_info: Optional[bool] = False,
        dtype: Union[Sequence[DALIDataType], DALIDataType, None] = None,
        ndim: Union[Sequence[int], int, None] = None,
        layout: Union[Sequence[str], str, None] = None,
        name: Optional[str] = None,
        cuda_stream: Optional[Any] = None,
        use_copy_kernel: Optional[bool] = False,
        cycle: Union[str, bool, None] = None,
        repeat_last: Optional[bool] = False,
        parallel: Optional[bool] = False,
        no_copy: Optional[bool] = None,
        prefetch_queue_depth: Optional[int] = 1,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    ) -> DataNode: ...

# The overload representing a call without specifying `num_outputs`. It expects a function
# returning a tensor or a batch of tensors directly, corresponding to exactly one DataNode output.
# `Any` can be replaced to represent TensorLike and BatchLike values.
# TODO(klecki): overloads with specific `batch` values can be considered
@overload
def external_source(
    source: Optional[Union[Callable[..., Any], Iterable[Any]]] = None,
    *,
    batch: Optional[bool] = None,
    batch_info: Optional[bool] = False,
    dtype: Union[DALIDataType, Sequence[DALIDataType], None] = None,
    ndim: Union[int, Sequence[int], None] = None,
    layout: Union[str, Sequence[str], None] = None,
    name: Optional[str] = None,
    device: Optional[str] = "cpu",
    cuda_stream: Optional[Any] = None,
    use_copy_kernel: Optional[bool] = False,
    cycle: Union[str, bool, None] = None,
    repeat_last: Optional[bool] = False,
    parallel: Optional[bool] = False,
    no_copy: Optional[bool] = None,
    prefetch_queue_depth: Optional[int] = 1,
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
) -> DataNode: ...

# The overload representing a call with `num_outputs` specified. It expects a function
# returning a tuple/sequence of tensors or batches, corresponding to a tuple of `num_outputs`
# DataNode outputs.
# `Any` can be replaced to represent TensorLike and BatchLike values.
# TODO(klecki): overloads with specific `batch` values can be considered
@overload
def external_source(
    source: Optional[Union[Callable[..., Sequence[Any]], Iterable[Sequence[Any]]]] = None,
    num_outputs: int = ...,
    *,
    batch: Optional[bool] = None,
    batch_info: Optional[bool] = False,
    dtype: Union[Sequence[DALIDataType], DALIDataType, None] = None,
    ndim: Union[Sequence[int], int, None] = None,
    layout: Union[Sequence[str], str, None] = None,
    name: Optional[str] = None,
    device: Optional[str] = "cpu",
    cuda_stream: Optional[Any] = None,
    use_copy_kernel: Optional[bool] = False,
    cycle: Union[str, bool, None] = None,
    repeat_last: Optional[bool] = False,
    parallel: Optional[bool] = False,
    no_copy: Optional[bool] = None,
    prefetch_queue_depth: Optional[int] = 1,
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
) -> Sequence[DataNode]: ...
