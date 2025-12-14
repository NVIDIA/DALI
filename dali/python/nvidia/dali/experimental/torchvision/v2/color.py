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

from typing import Sequence, Union, Literal, Optional
import nvidia.dali.fn as fn
import nvidia.dali as dali


class ColorJitter:
    """
    Randomly change the brightness, contrast, saturation and hue of an image or video.

    Parameters:
        brightness (float or tuple of python:float (min, max)) – How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of python:float (min, max)) – How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of python:float (min, max)) – How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of python:float (min, max)) – How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. To jitter hue, the pixel
                values of the input image has to be non-negative for conversion to HSV space
    """

    def _create_validated_param(self, param: Union[float, Sequence[float]]):
        if isinstance(param, float):
            validated_param = [max(0, 1 - param), 1 + param]
        else:
            validated_param = param

        if validated_param is not None and (
            len(validated_param) != 2 or validated_param[0] < 0 or validated_param[1] < 0
        ):
            raise ValueError("Parameters must be > 0")
        return validated_param

    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        hue: Optional[Union[float, Sequence[float]]] = None,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        self.brightness = self._create_validated_param(brightness)
        self.contrast = self._create_validated_param(contrast)
        self.saturation = self._create_validated_param(saturation)

        if isinstance(hue, float):
            self.hue = (-hue, hue)
        else:
            self.hue = hue

        if self.hue is not None and (len(self.hue) != 2 or self.hue[0] < -0.5 or self.hue[1] > 0.5):
            raise ValueError(f"hue values should be between (-0.5, 0.5) but got {self.hue}")

        self.device = device

    def __call__(self, data_input):
        """
        Performs the color jitter.
        """

        if self.brightness is None:
            brightness = 1
        else:
            brightness = fn.random.uniform(range=self.brightness)

        if self.contrast is None:
            contrast = 1
        else:
            contrast = fn.random.uniform(range=self.contrast)

        if self.saturation is None:
            saturation = 1
        else:
            saturation = fn.random.uniform(range=self.saturation)

        if self.hue is None:
            hue = 1
        else:
            hue = fn.random.uniform(range=self.hue)

        data_input = fn.color_twist(
            data_input,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            device=self.device,
        )

        return data_input


class Grayscale:
    """
    Convert images or videos to grayscale.

    Parameters:
        num_output_channels (int) – (1 or 3) number of channels desired for output image
    """

    def __init__(self, num_output_channels: int = 1, device: Literal["cpu", "gpu"] = "cpu"):
        if num_output_channels not in [1, 3]:
            raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}.")
        self.num_output_channels = num_output_channels
        self.device = device

    def __call__(self, data_input):
        """
        Converts an image to a grayscale
        """
        if self.device == "gpu":
            data_input = data_input.gpu()
        c = data_input.shape()[-1]
        if self.num_output_channels == 1 and c == 3:  # RGB (TODO: what if it is HSV?)
            return fn.color_space_conversion(
                data_input,
                image_type=dali.types.RGB,
                output_type=dali.types.GRAY,
                device=self.device,
            )
        else:
            return fn.hsv(data_input, saturation=0, device=self.device)
