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

from typing import Optional, Sequence, Union, Literal

from .operator import Operator, ArgumentVerificationRule

import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.types import DALIInterpType

from torchvision.transforms import InterpolationMode
import numpy as np


class VerificationSize(ArgumentVerificationRule):
    @classmethod
    def verify(self, *, size, max_size, interpolation, **_):
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
        InterpolationMode.NEAREST_EXACT: DALIInterpType.INTERP_NN,  # TODO
        InterpolationMode.BILINEAR: DALIInterpType.INTERP_LINEAR,
        InterpolationMode.BICUBIC: DALIInterpType.INTERP_CUBIC,
        InterpolationMode.BOX: DALIInterpType.INTERP_LINEAR,  # TODO:
        InterpolationMode.HAMMING: DALIInterpType.INTERP_GAUSSIAN,  # TODO:
        InterpolationMode.LANCZOS: DALIInterpType.INTERP_LANCZOS3,
    }
    arg_rules = [VerificationSize]

    @classmethod
    def infer_effective_size(
        cls,
        size: Optional[Union[int, Sequence[int]]],
        max_size: Optional[int] = None,
    ) -> Union[int, Sequence[int]]:

        mode = "default"

        if isinstance(size, (tuple, list)) and len(size) == 1:
            size = size[0]

        if isinstance(size, int):
            # If size is an int, smaller edge of the image will be matched to this number.
            # If size is an int: if the longer edge of the image is greater than max_size
            # after being resized according to size, size will be overruled so that the
            # longer edge is equal to max_size. As a result, the smaller edge may be shorter
            # than size.
            mode = "resize_shorter"

            return ((size, size), mode)

        if size is None:
            mode = "not_larger"
            return ((max_size, max_size), mode)

        return size, mode

    @classmethod
    def calculate_target_size(
        cls, orig_size: Sequence[int], effective_size: Sequence[int], max_size: int, no_size: bool
    ):
        orig_h = orig_size[0]
        orig_w = orig_size[1]
        target_h = effective_size[0]
        target_w = effective_size[1]

        # If size is None, then effective_size is max_size
        if no_size:
            if orig_h > orig_w:
                target_w = (max_size * orig_w) / orig_h
            else:
                target_h = (max_size * orig_h) / orig_w

        return target_h, target_w

    def __init__(
        self,
        size: Optional[Union[int, Sequence[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
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
        self.interpolation = Resize.interpolation_modes[interpolation]
        self.effective_size, self.mode = Resize.infer_effective_size(size, max_size)
        self.antialias = antialias

    def _kernel(self, data_input):
        """
        Performs the resize. The method infers the requested size in compliance
        with ``torchvision.transforms.Resize`` documentation and applies DALI operator on the
        ``data_input``.
        """

        target_h, target_w = Resize.calculate_target_size(
            data_input.shape(), self.effective_size, self.max_size, self.size is None
        )

        # Shorter edge limited by max size
        if self.mode == "resize_shorter":
            return fn.resize(
                data_input, device=self.device, resize_shorter=target_h, max_size=self.max_size
            )

        return fn.resize(
            data_input,
            device=self.device,
            size=fn.stack(
                fn.cast(target_h, dtype=dali.types.FLOAT),
                fn.cast(target_w, dtype=dali.types.FLOAT),
            ),
            mode=self.mode,
        )
