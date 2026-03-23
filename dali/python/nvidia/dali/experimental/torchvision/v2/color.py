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

from typing import Sequence, Literal, Optional

from .operator import (
    ArgumentVerificationRule,
    DataVerificationRule,
    Operator,
    VerifyIfRange,
    VerifyIfNonNegative,
    get_HWC_from_layout_pipeline,
)

import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn


class VerificationBCS(ArgumentVerificationRule):
    """
    Verify Brightness, Contrast and Saturation values

    Parameters
    ----------
    brightness : float or tuple of float
        How much to jitter brightness.
    contrast : float or tuple of float
        How much to jitter contrast.
    saturation : float or tuple of float
        How much to jitter saturation.
    """

    @classmethod
    def _validate_param(cls, param: float | Sequence[float], name: str):
        if param is None:
            raise ValueError(f"{name} must not be None")

        VerifyIfNonNegative.verify(values=param, name=name)

        if isinstance(param, float):
            param = [max(0, 1 - param), 1 + param]
        else:
            VerifyIfRange.verify(values=param, name=name)

    @classmethod
    def verify(cls, *, saturation, brightness, contrast, **_) -> None:
        VerificationBCS._validate_param(brightness, "brightness")
        VerificationBCS._validate_param(saturation, "saturation")
        VerificationBCS._validate_param(contrast, "contrast")


class VerificationHue(ArgumentVerificationRule):
    """
    Verify Hue

    Parameters
    ----------
    hue : float or tuple of float
        How much to jitter hue.
    """

    @classmethod
    def verify(cls, *, hue, **_) -> None:
        if hue is None:
            raise ValueError("hue must not be None")

        if isinstance(hue, (int, float, Sequence[int], Sequence[float])):
            raise TypeError(
                f"Hue must be int, float or a sequence of ints or floats, got {type(hue)}"
            )

        if hue is not None and (len(hue) != 2 or hue[0] < -0.5 or hue[1] > 0.5):
            raise ValueError(f"hue values should be between [-0.5, 0.5], but got {hue}")


class VerifyGrayscaleInputLayout(DataVerificationRule):
    """
    Verify if grayscale conversion is supported for the current input layout
    """

    @classmethod
    def verify(cls, data_input) -> None:

        layout = data_input.property("layout")[0]
        # If data layout is NHWC or NCHW, check the next character
        if layout == np.frombuffer(bytes("N", "utf-8"), dtype=np.uint8)[0]:
            layout = data_input.property("layout")[1]

        # CHW
        if layout == np.frombuffer(bytes("C", "utf-8"), dtype=np.uint8)[0]:
            raise NotImplementedError(
                "NCHW and CHW layout are not supported for Grayscale, expecting HWC or NHWC"
            )


def _get_BCSH(brightness, contrast, saturation, hue, random_function):
    """
    Gets random: brightness, contrast, saturation and hue.

    Parameters
    ----------
    brightness : float or tuple of float
        How much to jitter brightness.
    contrast : float or tuple of float
        How much to jitter contrast.
    saturation : float or tuple of float
        How much to jitter saturation.
    hue : float or tuple of float
        How much to jitter hue.
    """
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

    Parameters
    ----------
    brightness : float or tuple of float (min, max)
        How much to jitter brightness. brightness_factor is chosen uniformly from
        [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative
        numbers.
    contrast : float or tuple of float (min, max)
        How much to jitter contrast. contrast_factor is chosen uniformly from
        [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non-negative
        numbers.
    saturation : float or tuple of float (min, max)
        How much to jitter saturation. saturation_factor is chosen uniformly from
        [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative
        numbers.
    hue : float or tuple of float (min, max)
        How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given
        [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. To jitter hue,
        the pixel values of the input image has to be non-negative for conversion to HSV space.
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the color jitter. Can be ``"cpu"`` or ``"gpu"``.
    """

    arg_rules = [VerificationBCS, VerificationHue]

    def _create_param(self, param: int | float | Sequence[int] | Sequence[float]):
        if isinstance(param, (int, float)):
            return [max(0.0, 1.0 - param), 1.0 + param]
        elif isinstance(param, Sequence[int]):
            return [float(x) for x in param]
        else:
            return float(param)

    def __init__(
        self,
        brightness: Optional[int | float | Sequence[float]] = 0.0,
        contrast: Optional[int | float | Sequence[int] | Sequence[float]] = 0.0,
        saturation: Optional[int | float | Sequence[int] | Sequence[float]] = 0.0,
        hue: Optional[int | float | Sequence[int] | Sequence[float]] = 0.0,
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
        if isinstance(hue, (int, float)):
            self.hue = (-float(hue), float(hue))

    def _kernel(self, data_input):
        """
        Performs the color jitter using the ``fn.color_twist`` operator.
        """
        brightness, contrast, saturation, hue = _get_BCSH(
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
    """
    Verify the number of output channels for the Grayscale operator.

    Parameters
    ----------
    num_output_channels : int
        Number of channels desired for output image. Should be 1 or 3.
    """

    @classmethod
    def verify(cls, *, num_output_channels, **_):
        if num_output_channels not in [1, 3]:
            raise ValueError(f"num_output_channels must be 1 or 3, got {num_output_channels}.")


class Grayscale(Operator):
    """
    Convert images or videos to grayscale.

    Parameters
    ----------
    num_output_channels : int
        Number of channels desired for output image. Should be 1 or 3.
    """

    arg_rules = [VerificationGSOutputChannels]
    # TODO: it is currently useless since pipeline does not support raising exceptions
    # input_rules = [VerifyGrayscaleInputLayout]
    preprocess_data = get_HWC_from_layout_pipeline

    def __init__(self, num_output_channels: int = 1, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device=device, num_output_channels=num_output_channels)

        self.num_output_channels = num_output_channels

    def _kernel(self, data_input):
        """
        Converts an image to a grayscale using the ``fn.color_space_conversion`` or ``fn.hsv``
        operators.
        """
        _, _, c, output = data_input

        if self.num_output_channels == 1 and c == 3:  # RGB (TODO: what if it is HSV?)
            output = fn.color_space_conversion(
                output,
                image_type=dali.types.RGB,
                output_type=dali.types.GRAY,
                device=self.device,
            )
        elif self.num_output_channels == 1 and c == 1:  # Already handled
            pass
        elif self.num_output_channels == 3 and c == 1:
            output = fn.cat(output, output, output, axis_name="C")
        else:
            output = fn.hsv(output, saturation=0, device=self.device)

        return output
