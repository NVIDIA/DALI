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

from collections.abc import Callable
from typing import List, Optional, Tuple, Union, overload

import jax
from jax.sharding import Sharding

from nvidia.dali.data_node import DataNode
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from . import fn as fn
from .iterator import DALIGenericIterator as DALIGenericIterator

_PipelineFunction = Callable[..., Union[DataNode, Tuple[DataNode, ...]]]

@overload
def data_iterator(
    pipeline_fn: _PipelineFunction,
    output_map: List[str] = [],
    size: int = -1,
    reader_name: Optional[str] = None,
    auto_reset: Union[str, bool, None] = False,
    last_batch_padded: bool = False,
    last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
    prepare_first_batch: bool = True,
    sharding: Optional[Sharding] = None,
    devices: Optional[List[jax.Device]] = None,
    pmap_compatible: Optional[bool] = None,
) -> Callable[..., DALIGenericIterator]:
    """Decorate a DALI pipeline definition so that it creates a JAX iterator."""
    ...

@overload
def data_iterator(
    pipeline_fn: None = None,
    output_map: List[str] = [],
    size: int = -1,
    reader_name: Optional[str] = None,
    auto_reset: Union[str, bool, None] = False,
    last_batch_padded: bool = False,
    last_batch_policy: LastBatchPolicy = LastBatchPolicy.FILL,
    prepare_first_batch: bool = True,
    sharding: Optional[Sharding] = None,
    devices: Optional[List[jax.Device]] = None,
    pmap_compatible: Optional[bool] = None,
) -> Callable[[_PipelineFunction], Callable[..., DALIGenericIterator]]:
    """Configure a decorator that creates a JAX iterator from a DALI pipeline definition."""
    ...

__all__ = ["DALIGenericIterator", "data_iterator", "fn"]
