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

from typing import Optional, Sequence, Literal

from .operator import Operator, ArgumentVerificationRule

import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.types import DALIInterpType

from torchvision.transforms import InterpolationMode
import numpy as np


def get_inputHW(data_input):
    """
    Gets the height and width of the input data.

    Parameters
    ----------
    data_input : Tensor
       Input data to get the height and width of.

    Returns
    -------
    input_height : int
        Height of the input data.
    input_width : int
        Width of the input data.
    """
    layout = data_input.property("layout")[0]

    # If data layout is NHWC or NCHW, check the next character
    if layout == np.frombuffer(bytes("N", "utf-8"), dtype=np.uint8)[0]:
        layout = data_input.property("layout")[1]

    # CHW
    if layout == np.frombuffer(bytes("C", "utf-8"), dtype=np.uint8)[0]:
        input_height = data_input.shape()[-2]
        input_width = data_input.shape()[-1]
    # HWC
    else:
        input_height = data_input.shape()[-3]
        input_width = data_input.shape()[-2]

    return input_height, input_width, data_input


class VerificationSize(ArgumentVerificationRule):
    @classmethod
    def verify(cls, *, size, max_size, interpolation, **_):
        if size is not None and not isinstance(size, int) and not isinstance(size, (tuple, list)):
            raise ValueError(
                "Invalid combination: size must be int, None, or sequence of two ints. "
                "max_size only applies when size is int or None."
            )
        if size is None and max_size is None:
            raise ValueError("Must provide max_size if size is None.")
        if size is not None and max_size is not None and np.min(size) > max_size:
            raise ValueError("max_size should not be smaller than the actual size")
        if isinstance(size, (tuple, list)) and len(size) == 2 and max_size is not None:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller \
                 edge, i.e. size should be an int"
            )
        if interpolation not in Resize.interpolation_modes.keys():
            raise ValueError(f"Interpolation {type(interpolation)} is not supported")


class Resize(Operator):
    """
    Resize the input image to the given size
    If the image is torch Tensor, it is expected to have […, H, W] shape, where … means a maximum
    of two leading dimensions

    Parameters
    ----------
        size:sequence or int
            Desired output size. If size is a sequence like (h, w), output size will be matched
            to this. If size is an int, smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to (size * height / width, size).
        interpolation : InterpolationMode or int
            ``torchvision.transforms.InterpolationMode``. Default is InterpolationMode.BILINEAR.
            If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
        max_size : int, optional
            The maximum allowed for the longer edge of the resized image. If the longer edge of
            the image is greater than max_size after being resized according to size, size will
            be overruled so that the longer edge is equal to max_size. As a result, the smaller
            edge may be shorter than size. This is only supported if size is an int.
        antialias : bool, optional
            Whether to apply antialiasing. If ``True``, antialiasing will be applied. If ``False``,
            antialiasing will not be applied.
        device : Literal["cpu", "gpu"], optional, default = "cpu"
            Device to use for the resize. Can be ``"cpu"`` or ``"gpu"``.
    """

    # 'NEAREST', 'NEAREST_EXACT', 'BILINEAR', 'BICUBIC', 'BOX', 'HAMMING', 'LANCZOS'
    interpolation_modes = {
        InterpolationMode.NEAREST: DALIInterpType.INTERP_NN,
        InterpolationMode.BILINEAR: DALIInterpType.INTERP_LINEAR,
        InterpolationMode.BICUBIC: DALIInterpType.INTERP_CUBIC,
        InterpolationMode.LANCZOS: DALIInterpType.INTERP_LANCZOS3,
        # Not supported, but need to be here to not generate ValueError during VerificationSize
        InterpolationMode.NEAREST_EXACT: DALIInterpType.INTERP_NN,
        InterpolationMode.BOX: DALIInterpType.INTERP_NN,
        InterpolationMode.HAMMING: DALIInterpType.INTERP_NN,
    }

    not_supported_interpolation_modes = [
        InterpolationMode.NEAREST_EXACT,
        InterpolationMode.BOX,
        InterpolationMode.HAMMING,
    ]

    arg_rules = [VerificationSize]
    preprocess_data = get_inputHW

    @classmethod
    def infer_effective_size(
        cls,
        size: Optional[int | Sequence[int]],
        max_size: Optional[int] = None,
    ) -> Optional[int | Sequence[int]]:
        """Normalizes the size parameter. Called once at initialization.

        Returns the size in a canonical form:

        - ``int`` — resize the shorter edge to this value (aspect-ratio preserving)
        - ``None`` — use ``max_size`` only (resize so longer edge equals ``max_size``)
        - ``(h, w)`` tuple/list — resize to the exact target dimensions
        """
        if isinstance(size, (tuple, list)) and len(size) == 1:
            size = size[0]
        return size

    @classmethod
    def calculate_target_size_dynamic_mode(
        cls,
        orig_size: Sequence[int],
        size: Optional[int | Sequence[int]],
        max_size: Optional[int],
    ):
        """Computes the output ``(out_h, out_w)`` compatible with ``torchvision.v2.Resize``.

        Called per resize invocation with the actual input shape.

        Note: This method needs to be called only when in Dynamic Mode
        Unfortunately, both method are needed because of graph creation struggles with proper
        translation of class methods calls
        """
        orig_h = orig_size[0]
        orig_w = orig_size[1]

        if isinstance(size, (tuple, list)):
            # Exact target dimensions — return directly
            return size[0], size[1]

        if size is None:
            # Only max_size given: resize so the longer edge equals max_size
            if orig_h >= orig_w:
                return max_size, int(max_size * orig_w / orig_h)
            else:
                return int(max_size * orig_h / orig_w), max_size

        # size is int: resize the shorter edge to size, maintaining aspect ratio
        s = size
        if orig_h <= orig_w:
            # height is the shorter (or equal) edge
            out_h = s
            out_w = int(s * orig_w / orig_h)
            if max_size is not None and out_w > max_size:
                out_h = int(max_size * out_h / out_w)
                out_w = max_size
        else:
            # width is the shorter edge
            out_h = int(s * orig_h / orig_w)
            out_w = s
            if max_size is not None and out_h > max_size:
                out_w = int(max_size * out_w / out_h)
                out_h = max_size

        return out_h, out_w

    @classmethod
    def calculate_target_size_pipeline_mode(
        cls,
        orig_size: Sequence[int],
        size: Optional[int | Sequence[int]],
        max_size: Optional[int],
    ):
        """Computes the output ``(out_h, out_w)`` compatible with ``torchvision.v2.Resize``.

        Called per resize invocation with the actual input shape.

        Note: This method needs to be called only when in Pipeline Mode
        """
        orig_h = orig_size[0]
        orig_w = orig_size[1]

        if isinstance(size, (tuple, list)):
            # Exact target dimensions — return directly
            return size[0], size[1]

        if size is None:
            # Only max_size given: resize so the longer edge equals max_size
            if orig_h >= orig_w:
                return max_size, fn.cast(
                    dali.math.floor(max_size * orig_w / orig_h), dtype=dali.types.INT32
                )
            else:
                return (
                    fn.cast(dali.math.floor(max_size * orig_h / orig_w), dtype=dali.types.INT32),
                    max_size,
                )

        # size is int: resize the shorter edge to size, maintaining aspect ratio
        s = size
        if orig_h <= orig_w:
            # height is the shorter (or equal) edge
            out_h = s
            out_w = fn.cast(dali.math.floor(s * orig_w / orig_h), dtype=dali.types.INT32)
            if max_size is not None and out_w > max_size:
                out_h = fn.cast(dali.math.floor(max_size * out_h / out_w), dtype=dali.types.INT32)
                out_w = max_size
        else:
            # width is the shorter edge
            out_h = fn.cast(dali.math.floor(s * orig_h / orig_w), dtype=dali.types.INT32)
            out_w = s
            if max_size is not None and out_h > max_size:
                out_w = fn.cast(dali.math.floor(max_size * out_w / out_h), dtype=dali.types.INT32)
                out_h = max_size

        return out_h, out_w

    def __init__(
        self,
        size: Optional[int | Sequence[int]],
        interpolation: InterpolationMode | int = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[bool] = True,
        device: Literal["cpu", "gpu"] = "cpu",
    ):

        super().__init__(
            device=device,
            size=size,
            max_size=max_size,
            interpolation=interpolation,
        )

        self.size = size
        self.max_size = max_size

        if interpolation in Resize.not_supported_interpolation_modes:
            raise NotImplementedError(f"Interpolation mode: {interpolation} is not supported")

        self.interpolation = Resize.interpolation_modes[interpolation]
        self.size_normalized = Resize.infer_effective_size(size, max_size)
        self.antialias = antialias

    def _kernel(self, data_input):
        """
        Performs the resize. The method infers the requested size in compliance
        with ``torchvision.transforms.Resize`` documentation and applies DALI operator on the
        ``data_input``.
        """

        in_h, in_w, data_input = data_input

        target_h, target_w = Resize.calculate_target_size_pipeline_mode(
            (in_h, in_w),
            self.size_normalized,
            self.max_size,
        )

        return fn.resize(
            data_input,
            device=self.device,
            size=fn.stack(
                fn.cast(target_h, dtype=dali.types.FLOAT),
                fn.cast(target_w, dtype=dali.types.FLOAT),
            ),
            interp_type=self.interpolation,
            antialias=self.antialias,
        )
