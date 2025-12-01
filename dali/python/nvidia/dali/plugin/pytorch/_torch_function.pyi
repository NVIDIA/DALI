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

import torch

from typing import Union, Optional
from typing import Callable, Sequence, Tuple

from nvidia.dali.data_node import DataNode
from nvidia.dali._typing import TensorLikeIn

class TorchPythonFunction:
    """Executes a function that is operating on Torch tensors"""

    def __init__(
        self,
        /,
        function: Optional[
            Callable[..., Union[torch.Tensor, Tuple[torch.Tensor, ...], None]]
        ] = None,
        num_outputs: int = 1,
        device: str = "cpu",
        batch_processing: bool = False,
        *,
        output_layouts: Union[Sequence[str], str, None] = None,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        preserve: Optional[bool] = False,
        seed: Optional[int] = -1,
        name: Optional[str] = None,
    ) -> None: ...
    def __call__(
        self,
        /,
        *input: Union[DataNode, TensorLikeIn],
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        preserve: Optional[bool] = False,
        seed: Optional[int] = -1,
        name: Optional[str] = None,
    ) -> Union[DataNode, Sequence[DataNode], None]: ...
