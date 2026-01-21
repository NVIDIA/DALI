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


import nvidia.dali.experimental.dynamic as ndd

import sys

sys.path.append("..")
from ..operator import adjust_input  # noqa: E402


@adjust_input
def horizontal_flip(inpt: ndd.Tensor) -> ndd.Tensor:
    """
    Horizontally flips the given tensor.
    Refer to ``HorizontalFlip`` for more details.
    """
    return ndd.flip(inpt, horizontal=1, vertical=0)


@adjust_input
def vertical_flip(inpt: ndd.Tensor) -> ndd.Tensor:
    """
    Vertically flips the given tensor.
    Refer to ``VerticalFlip`` for more details.
    """
    return ndd.flip(inpt, horizontal=0, vertical=1)
