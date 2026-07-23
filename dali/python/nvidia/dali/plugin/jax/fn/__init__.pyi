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

from collections.abc import Callable, Sequence
from typing import Optional, Protocol, Tuple, Union, overload

import jax

from nvidia.dali._typing import TensorLikeIn
from nvidia.dali.data_node import DataNode

class JaxCallback(Protocol):
    def __call__(self, *args: jax.Array) -> Optional[Union[jax.Array, Tuple[jax.Array, ...]]]: ...

class DaliCallback(Protocol):
    def __call__(
        self, *args: Union[DataNode, TensorLikeIn]
    ) -> Optional[Union[DataNode, Sequence[DataNode]]]: ...

@overload
def jax_function(
    function: JaxCallback,
    num_outputs: int = 1,
    output_layouts: Union[None, str, Tuple[str, ...]] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
    device: Optional[str] = None,
    preserve: bool = True,
) -> DaliCallback:
    """Transform a JAX callback into a DALI operator callable."""
    ...

@overload
def jax_function(
    function: None = None,
    num_outputs: int = 1,
    output_layouts: Union[None, str, Tuple[str, ...]] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
    device: Optional[str] = None,
    preserve: bool = True,
) -> Callable[[JaxCallback], DaliCallback]:
    """Configure a decorator that transforms a JAX callback into a DALI operator callable."""
    ...
