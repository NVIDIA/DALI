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

from typing import Literal
from .operator import Operator
import nvidia.dali.fn as fn


class RandomFlip(Operator):
    """
    Randomly flips the given image randomly with a given probability.

    Parameters
    ----------
        p : float
            Probability of the image being flipped. Default value is 0.5
        horizontal : int
            Flip the horizontal dimension.
        device : Literal["cpu", "gpu"], optional, default = "cpu"
            Device to use for the flip. Can be ``"cpu"`` or ``"gpu"``.
    """

    def __init__(self, p: float = 0.5, horizontal: int = 1, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device=device)
        self.prob = p
        self.device = device
        self.horizontal = horizontal

    def _kernel(self, data_input):
        if self.horizontal:
            data_input = fn.flip(
                data_input, horizontal=fn.random.coin_flip(probability=self.prob), vertical=0
            )
        else:
            data_input = fn.flip(
                data_input, horizontal=0, vertical=fn.random.coin_flip(probability=self.prob)
            )

        return data_input


class RandomHorizontalFlip(RandomFlip):
    """
    Randomly horizontally flips the given image randomly with a given probability.

    Parameters
    ----------
    p : float
        Probability of the image being flipped. Default value is 0.5
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the flip. Can be ``"cpu"`` or ``"gpu"``.
    """

    def __init__(self, p: float = 0.5, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(p, True, device)


class RandomVerticalFlip(RandomFlip):
    """
    Randomly vertically flips the given image randomly with a given probability.

    Parameters
    ----------
    p : float
        Probability of the image being flipped. Default value is 0.5
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the flip. Can be ``"cpu"`` or ``"gpu"``.
    """

    def __init__(self, p: float = 0.5, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(p, False, device)
