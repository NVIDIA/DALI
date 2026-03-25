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

import numpy as np
import torch
from typing import Sequence, Literal
import nvidia.dali.fn as fn

from .operator import Operator, _ArgumentValidateRule


class _ValidateStd(_ArgumentValidateRule):
    """
    Verify the standard deviation argument for the Normalize operator.

    Parameters
    ----------
    std : sequence
        Sequence of standard deviations for each channel.
    """

    @classmethod
    def verify(cls, *, std, **_) -> None:
        if not isinstance(std, (int, float, Sequence, torch.Tensor, np.ndarray)) or (
            isinstance(std, Sequence) and isinstance(std, str)
        ):
            raise TypeError(f"Std must be an int, a float or a Sequence, got {type(std)}")
        if np.any(np.array(std) == 0):
            raise ValueError("Std must not be 0")


class _ValidateMean(_ArgumentValidateRule):
    """
    Verify the mean argument for the Normalize operator.

    Parameters
    ----------
    mean : sequence
        Sequence of means for each channel.
    """

    @classmethod
    def verify(cls, *, mean, **_) -> None:
        # This is on-pair validation with Torchvision - no other validation is performed
        _ = torch.as_tensor(mean)


class Normalize(Operator):
    """
    Normalize a tensor image or video with mean and standard deviation.

    This transform does not support PIL Image.
    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels,
    this transform will normalize each channel of the input torch.*Tensor
    i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]

    Parameters
    ----------
    mean : sequence
        Sequence of means for each channel.
    std : sequence
        Sequence of standard deviations for each channel.
    inplace : bool, optional
        Bool to make this operation in-place. Not supported.
    """

    arg_rules = [_ValidateStd, _ValidateMean]
    # TODO: currently not supported
    # input_rules = [VerificationIsTensor]

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(device=device, mean=mean, std=std)

        # self.mean = np.asarray(mean)[:, None, None]
        # self.std = np.asarray(std)[:, None, None]
        self.mean = mean
        self.std = std
        if inplace:
            raise NotImplementedError("inplace is not supported")
        self.inplace = inplace

    def _kernel(self, data_input):

        layout = data_input.property("layout")[-1]

        if layout.cpu() == np.frombuffer(bytes("C", "utf-8"), dtype=np.uint8)[0]:
            mean = np.asarray(self.mean)[None, None, :]
            std = np.asarray(self.std)[None, None, :]
        else:
            mean = np.asarray(self.mean)[:, None, None]
            std = np.asarray(self.std)[:, None, None]

        return fn.normalize(
            data_input,
            mean=mean,
            stddev=std,
            device=self.device,
        )
