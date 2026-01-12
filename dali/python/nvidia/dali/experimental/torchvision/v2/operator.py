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

    Implement verify method in a child class raising an exception in case of failed verification
    """

    @classmethod
    @abstractmethod
    def verify(self, data) -> None:
        pass


class ArgumentVerificationRule(ABC):
    """
    Abstract base class for input verification rules

    Implement verify method in a child class raising an exception in case of failed verification
    """

    @classmethod
    @abstractmethod
    def verify(cls, **kwargs) -> None:
        pass


class VerificationIsTensor(DataVerificationRule):
    @classmethod
    def verify(cls, data):
        if not isinstance(data, (torch.Tensor)):
            raise TypeError(f"Data should be Tensor. Got {type(data)}")


class VerificationTensorOrImage(DataVerificationRule):
    @classmethod
    def verify(cls, data):
        if not isinstance(data, (Image.Image, torch.Tensor)):
            raise TypeError(f"inpt should be Tensor or PIL Image. Got {type(data)}")


class VerificationChannelCount(DataVerificationRule):
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
    @classmethod
    def verify(cls, *, values, **_) -> None:
        if isinstance(values, (int, float)) and values <= 0:
            raise ValueError(f"Value must be positive, got {values}")
        elif isinstance(values, (list, tuple)) and any(k <= 0 for k in values):
            raise ValueError(f"Values should be positive number, got {values}")


class VerificationSize(ArgumentVerificationRule):
    @classmethod
    def verify(cls, *, size, **_) -> None:
        if not isinstance(size, (int, list, tuple)):
            raise TypeError(f"Size must be int or sequence, got {type(size)}")
        elif isinstance(size, (list, tuple)) and len(size) > 2:
            raise ValueError(f"Size sequence must have length 1 or 2, got {len(size)}")
        VerifyIfPositive.verify(values=size)


class Operator(ABC):
    """
    Abstract base class for operator specification

    Implement _kernel for algorithm specific processing

    Parameters:
        arg_rules - sequence of verification rules for algorithm's arguments
        input_rules - sequence of verification rules for algorithm's input data
        device - device on which operator will run ("gpu" or "cpu")


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

        # TODO: does not work with pipelines
        # type(self).verify_data(data_input)

        if self.device == "gpu":
            data_input = data_input.gpu()

        if type(self).preprocess_data:
            data_input = type(self).preprocess_data(data_input)

        output = self._kernel(data_input)

        return output


def adjust_input(func):
    def inner_function(inpt, *args, **kwargs):
        VerificationTensorOrImage.verify(inpt)
        _input = inpt
        mode = "RGB"
        if isinstance(inpt, Image.Image):
            _input = ndd.Tensor(np.array(inpt, copy=True), layout="HWC")
            if _input.shape[-1] == 1:
                mode = "L"
            elif _input.shape[-1] == 4:
                mode = "RGBA"
        elif isinstance(inpt, torch.Tensor):
            if inpt.ndim <= 3:
                _input = ndd.Tensor(inpt, layout="CHW")
            else:
                _input = ndd.as_batch(inpt)  # , layout="BCHW")
        else:
            raise ValueError(f"Data type: {type(inpt)} is not supported")
        output = func(_input, *args, **kwargs)

        if isinstance(inpt, Image.Image):
            return Image.fromarray(np.asarray(output.evaluate()), mode=mode)
        elif isinstance(inpt, torch.Tensor):
            """
            if isinstance(output, ndd.Batch):
                return torch.as_tensor(ndd.as_tensor(output))
            elif isinstance(output, ndd.Tensor):
                return torch.as_tensor(output)
            else:
                raise TypeError(f"Invalid output type: {type(output)}")
            """
            if isinstance(output, ndd.Tensor):
                output = output.evaluate()
                if isinstance(output, ndd.Batch):
                    output = ndd.as_tensor(output)

                if output.device.device_type == 'gpu':
                    return torch.from_dlpack(ndd.as_tensor(output)._storage)
                else:
                    return torch.Tensor(np.asarray(output._storage))
        else:
            return output

    return inner_function
