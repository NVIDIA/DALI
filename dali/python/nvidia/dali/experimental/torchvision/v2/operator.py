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

from abc import ABC, abstractmethod
from typing import Sequence, Literal

from PIL import Image
import torch
import numpy as np

import nvidia.dali.experimental.dynamic as ndd


class DataVerificationRule(ABC):
    """
    Abstract base class for data verification rules

    Implement ``verify`` method in a child class raising an exception in case of failed verification
    """

    @classmethod
    @abstractmethod
    def verify(self, data) -> None:
        pass


class ArgumentVerificationRule(ABC):
    """
    Abstract base class for input verification rules

    Implement ``verify`` method in a child class raising an exception in case of failed verification
    """

    @classmethod
    @abstractmethod
    def verify(cls, **kwargs) -> None:
        pass


class VerificationIsTensor(DataVerificationRule):
    """
    Verify if the data is a ``torch.Tensor``.

    Parameters
    ----------
    data : any
        Data to verify. Should be a ``torch.Tensor``.
    """

    @classmethod
    def verify(cls, data):
        if not isinstance(data, (torch.Tensor)):
            raise TypeError(f"Data should be Tensor. Got {type(data)}")


class VerificationTensorOrImage(DataVerificationRule):
    """
    Verify if the data is a ``torch.Tensor`` or ``PIL.Image``.

    Parameters
    ----------
    data : any
        Data to verify. Should be a ``torch.Tensor`` or ``PIL.Image``.
    """

    @classmethod
    def verify(cls, data):
        if not isinstance(data, (Image.Image, torch.Tensor)):
            raise TypeError(f"inpt should be Tensor or PIL Image. Got {type(data)}")


class VerificationChannelCount(DataVerificationRule):
    """
    Verify the number of channels for the input data.

    Parameters
    ----------
    data : any
        Data to verify in CHW format.
    """

    CHANNELS = [1, 2, 3, 4]

    @classmethod
    def verify(cls, data):
        if (
            isinstance(data, torch.Tensor)
            and data.shape[-3] not in VerificationChannelCount.CHANNELS
        ):
            raise ValueError(
                f"Input should be in CHW if Tensor. \
                  Supported channels: {VerificationChannelCount.CHANNELS} is {data.shape[-3]}"
            )


class VerifyIfPositive(ArgumentVerificationRule):
    """
    Verify if the value is positive.

    Parameters
    ----------
    values : any
        Value to verify. Should be a positive number.
    """

    @classmethod
    def verify(cls, *, values, name, **_) -> None:
        if isinstance(values, (int, float)) and values <= 0:
            raise ValueError(f"Value {name} must be positive, got {values}")
        elif isinstance(values, (list, tuple)) and any(k <= 0 for k in values):
            raise ValueError(f"Values {name} should be positive number, got {values}")


class VerifyIfOrderedPair(ArgumentVerificationRule):
    """
    Verify if the value is an ordered pair.

    Parameters
    ----------
    values : any
        Value to verify. Should be an ordered pair.
    """

    @classmethod
    def verify(cls, *, values, name, **_) -> None:
        if isinstance(values, (list, tuple)) and len(values) == 2 and values[0] > values[1]:
            raise ValueError(f"Values {name} should be ordered, got {values}")


class VerificationSize(ArgumentVerificationRule):
    """
    Verify if the value is an integer or a sequence of length 1 or 2.

    Parameters
    ----------
    size : any
        Value to verify. Should be an integer or a sequence of length 1 or 2.
    """

    @classmethod
    def verify(cls, *, size, **_) -> None:
        if not isinstance(size, (int, list, tuple)):
            raise TypeError(f"Size must be int or sequence, got {type(size)}")
        elif isinstance(size, (list, tuple)) and len(size) > 2:
            raise ValueError(f"Size sequence must have length 1 or 2, got {len(size)}")
        VerifyIfPositive.verify(values=size, name="size")


class Operator(ABC):
    """
    Abstract base class for operator specification

    Implement _kernel for algorithm specific processing

    ``arg_rules`` - a sequence of verification rules for algorithm's arguments.
    ``input_rules`` - a sequence of verification rules for algorithm's input data.
    ``preprocess_data`` - a function to preprocess the input data.

    Parameters
    ----------
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the operator. Can be ``"cpu"`` or ``"gpu"``.
    **kwargs
        Additional keyword arguments for the operator.
    """

    arg_rules: Sequence[ArgumentVerificationRule] = []
    input_rules: Sequence[DataVerificationRule] = []
    preprocess_data = None

    @classmethod
    def verify_args(cls, **kwargs):
        for rule in cls.arg_rules:
            rule.verify(**kwargs)

    @classmethod
    def verify_data(cls, data_input):
        for rule in cls.input_rules:
            rule.verify(data_input)

    def __init__(self, device: Literal["cpu", "gpu"] = "cpu", **kwargs):
        self.device = device
        type(self).verify_args(**kwargs)

    @abstractmethod
    def _kernel(self, data_input):
        """
        Algorithm's processing
        """
        pass

    def __call__(self, data_input):

        if self.device == "gpu":
            data_input = data_input.gpu()

        if type(self).preprocess_data:
            data_input = type(self).preprocess_data(data_input)

        output = self._kernel(data_input)

        return output


def adjust_input(func):
    """
    This decorator transforms the 1st argument of a function to internal DALI representation
    according to the following rules:
    - ``PIL.Image`` -> ``ndd.Tensor(layout = "HWC")``
    - ``torch.Tensor``:
        - ``ndim == 3`` -> ``ndd.Tensor(layout = "CHW")``
        - ``ndim > 3`` -> ``ndd.Batch(layout = "CHW")``

    Note: When new input types are supported this function will be extended.
    """

    def transform_input(inpt) -> ndd.Tensor | ndd.Batch:
        """
        Transforms supported inputs to either DALI tensor or batch
        """
        mode = "RGB"
        if isinstance(inpt, Image.Image):
            _input = ndd.Tensor(np.array(inpt, copy=True), layout="HWC")
            if _input.shape[-1] == 1:
                mode = "L"
            elif _input.shape[-1] == 4:
                mode = "RGBA"
        elif isinstance(inpt, torch.Tensor):
            if inpt.ndim == 3:
                _input = ndd.Tensor(inpt, layout="CHW")
            elif inpt.ndim > 3:
                # The following should work, bug: https://jirasw.nvidia.com/browse/DALI-4566
                # _input = ndd.as_batch(inpt, layout="NCHW")
                # WAR:
                _input = ndd.as_batch(ndd.as_tensor(inpt), layout="CHW")
            else:
                raise TypeError(f"Tensor has < 3 dimensions: {inpt.ndim} / {inpt.shape}")
        else:
            raise TypeError(f"Data type: {type(inpt)} is not supported")

        return _input, mode

    def adjust_output(
        output: ndd.Tensor | ndd.Batch, inpt, mode: str = "RGB"
    ) -> Image.Image | torch.Tensor:
        """
        Adjusts output to match the original input type or operator's result
        """
        if isinstance(inpt, Image.Image):
            if output.shape[-1] == 1:
                output = np.asarray(output).squeeze(2)
                mode = "L"
            return Image.fromarray(np.asarray(output), mode=mode)
        elif isinstance(inpt, torch.Tensor):
            if isinstance(output, ndd.Batch):
                output = ndd.as_tensor(output)
            elif isinstance(output, ndd.Tensor):
                output = output
            else:
                raise TypeError(f"Invalid output type: {type(output)}")

            # This is WAR for DLPpack not supporting pinned memory
            if output.device.device_type == "cpu":
                output = np.asarray(output)

            return torch.as_tensor(output)
        else:
            return output

    def inner_function(inpt, *args, **kwargs):

        _input, mode = transform_input(inpt)
        output = func(_input, *args, **kwargs)

        output = output.evaluate()

        return adjust_output(output, inpt, mode)

    return inner_function
