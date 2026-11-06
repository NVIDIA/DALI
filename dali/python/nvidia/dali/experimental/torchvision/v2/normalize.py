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

from .operator import Operator, ArgumentValidateRule


def _channel_params(mean, std, channels_last: bool):
    """
    Returns mean and std as float32 arrays that broadcast over the channels of the input.
    Shared by the ``Normalize`` operator and the functional ``normalize``, so both compute the
    same values. Float32 matches torchvision; in the dynamic API, float64 constants would promote
    the output to float64.
    """
    shape = (1, 1, -1) if channels_last else (-1, 1, 1)
    return np.asarray(mean, np.float32).reshape(shape), np.asarray(std, np.float32).reshape(shape)


class _ValidateStd(ArgumentValidateRule):
    """
    Verify the standard deviation argument for the Normalize operator.

    Parameters
    ----------
    std : sequence
        Sequence of standard deviations for each channel.
    """

    @classmethod
    def verify(cls, *, std, **_) -> None:
        if not isinstance(std, (Sequence, torch.Tensor, np.ndarray)) or (
            isinstance(std, Sequence) and isinstance(std, str)
        ):
            raise TypeError(f"Std must be a Sequence, got {type(std)}")

        tensor_std = torch.as_tensor(std)
        if torch.any(tensor_std == 0):
            raise ValueError("Std must not be 0")


class _ValidateMean(ArgumentValidateRule):
    """
    Verify the mean argument for the Normalize operator.

    Parameters
    ----------
    mean : sequence
        Sequence of means for each channel.
    """

    @classmethod
    def verify(cls, *, mean, **_) -> None:
        if not isinstance(mean, (Sequence, torch.Tensor, np.ndarray)) or (
            isinstance(mean, Sequence) and isinstance(mean, str)
        ):
            raise TypeError(f"Mean must be a Sequence, got {type(mean)}")
        # This is on-par validation with Torchvision - no other validation is performed
        _ = torch.as_tensor(mean)


class Normalize(Operator):
    """
    Normalize a tensor image with mean and standard deviation.

    This transform does not support PIL Image.
    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels,
    this transform will normalize each channel of the input torch.*Tensor
    i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]

    .. note::
        Unlike torchvision, which accepts only floating point tensors, this transform also accepts
        integer tensors, e.g. the ``uint8`` output of ``PILToTensor``, and converts them to
        ``float32`` automatically. The values are not rescaled, so ``uint8`` input is normalized
        in the [0, 255] range. To get the result that torchvision gives for the image converted to
        [0, 1] (``ToDtype(torch.float32, scale=True)``), multiply ``mean`` and ``std`` by 255.

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
    # Like torchvision, Normalize does not accept PIL images (see Compose._verify_pil_support)
    supports_pil = False

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(device=device, mean=mean, std=std)

        self.mean = mean
        self.std = std
        if inplace:
            raise NotImplementedError("inplace is not supported")
        self.inplace = inplace

    def _kernel(self, data_input):

        layout = data_input.property("layout")[-1]

        if layout == np.frombuffer(bytes("C", "utf-8"), dtype=np.uint8)[0]:
            mean, std = _channel_params(self.mean, self.std, channels_last=True)
        else:
            mean, std = _channel_params(self.mean, self.std, channels_last=False)

        # Arithmetic broadcasting is right-aligned, so it works for CHW, FCHW, HWC and FHWC.
        # Unlike torchvision, std can expand the channels (e.g. 1 channel with 3-element std), just
        # like mean can: torchvision rejects it only because it divides in place.
        return (data_input - mean) / std
