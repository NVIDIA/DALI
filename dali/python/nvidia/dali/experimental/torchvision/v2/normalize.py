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

import numpy as np
from typing import Sequence, Literal
import nvidia.dali.fn as fn


class Normalize:
    """
    Normalize a tensor image or video with mean and standard deviation.

    This transform does not support PIL Image.
    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels,
    this transform will normalize each channel of the input torch.*Tensor
    i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]

    Parameters:

    mean (sequence) – Sequence of means for each channel.
    std (sequence) – Sequence of standard deviations for each channel.
    inplace (bool,optional) – Bool to make this operation in-place.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        self.mean = np.asarray(mean)[None, None, :]
        self.std = np.asarray(std)[None, None, :]
        self.inplace = inplace
        self.device = device

    def __call__(self, data_input):
        if self.device == "gpu":
            data_input = data_input.gpu()

        return fn.normalize(data_input, mean=self.mean, stddev=self.std, device=self.device)
