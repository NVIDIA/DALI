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
from .operator import ArgumentVerificationRule, Operator


class VerificationBCS(ArgumentVerificationRule):
    """
    Verify Brighness, Contrast and Saturation values
    """

    @classmethod
    def _validate_param(cls, param: Union[float, Sequence[float]]):
        if isinstance(param, float):
            param = [max(0, 1 - param), 1 + param]

        if param is not None and (
            len(param) != 2 or param[0] < 0 or param[1] < 0 or param[0] > param[1]
        ):
            raise ValueError("Parameters must be > 0")

    @classmethod
    def verify(cls, *, saturation, brightness, contrast, **_) -> None:
        VerificationBCS._validate_param(brightness)
        VerificationBCS._validate_param(saturation)
        VerificationBCS._validate_param(contrast)


class VerificationHue(ArgumentVerificationRule):
    """
    Verify Hue
    """

    @classmethod
    def verify(cls, *, hue, **_) -> None:
        if isinstance(hue, float):
            hue = (-hue, hue)

        if hue is not None and (len(hue) != 2 or hue[0] < -0.5 or hue[1] > 0.5):
            raise ValueError(f"hue values should be between (-0.5, 0.5) but got {hue}")


def get_BCSH(brightness, contrast, saturation, hue, random_function):
    if brightness[0] == brightness[1]:
        brightness = brightness[0]
    else:
        brightness = random_function(range=brightness)

    if contrast[0] == contrast[1]:
        contrast = contrast[0]
    else:
        contrast = random_function(range=contrast)

    if saturation[0] == saturation[1]:
        saturation = saturation[0]
    else:
        saturation = random_function(range=saturation)

    if hue[0] == hue[1]:
        hue = hue[0]
    else:
        hue = random_function(range=hue)

    return brightness, contrast, saturation, hue


class ColorJitter(Operator):
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

    arg_rules = [VerificationBCS, VerificationHue]

    def _create_param(self, param: Union[float, Sequence[float]]):
        if isinstance(param, float):
            return [max(0, 1 - param), 1 + param]
        else:
            return param

    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = 1.0,
        contrast: Optional[Union[float, Sequence[float]]] = 1.0,
        saturation: Optional[Union[float, Sequence[float]]] = 1.0,
        hue: Optional[Union[float, Sequence[float]]] = 0.0,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(
            device=device,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

        self.brightness = self._create_param(brightness)
        self.contrast = self._create_param(contrast)
        self.saturation = self._create_param(saturation)

        self.hue = hue
        if isinstance(hue, float):
            self.hue = (-hue, hue)

    def _kernel(self, data_input):
        """
        Performs the color jitter.
        """
        brightness, contrast, saturation, hue = get_BCSH(
            self.brightness, self.contrast, self.saturation, self.hue, fn.random.uniform
        )

        data_input = fn.color_twist(
            data_input,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            device=self.device,
        )

        return data_input


class VerificationGSOutputChannels(ArgumentVerificationRule):
    @classmethod
    def verify(cls, *, num_output_channels, **_):
        if num_output_channels not in [1, 3]:
            raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}.")


class Grayscale(Operator):
    """
    Convert images or videos to grayscale.

    Parameters:
        num_output_channels (int) – (1 or 3) number of channels desired for output image
    """

    arg_rules = [VerificationGSOutputChannels]

    def __init__(self, num_output_channels: int = 1, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device=device, num_output_channels=num_output_channels)

        self.num_output_channels = num_output_channels

    def _kernel(self, data_input):
        """
        Converts an image to a grayscale
        """
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
