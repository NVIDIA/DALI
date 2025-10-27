# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional, Sequence, Union, Tuple

import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.types import DALIInterpType

from torchvision.transforms import InterpolationMode


class Resize:
    """
    Resize the input to the given size.

    If the input is a torch.Tensor or a TVTensor (e.g. Image, Video, BoundingBoxes etc.)
    it can have arbitrary number of leading batch dimensions. For example, the image can
    have [..., C, H, W] shape. A bounding box can have [..., 4] shape.

    Parameters:
        size (sequence or int) – Desired output size of the crop. If size is an int
                               instead of sequence like (h, w), a square crop (size, size)
                               is made. If provided a sequence of length 1, it will be
                               interpreted as (size[0], size[0]).
        interpolation (InterpolationMode or optional) - Desired interpolation enum defined by
                               torchvision.transforms.InterpolationMode. Default is
                               InterpolationMode.BILINEAR.
        max_size (int, optional) - The maximum allowed for the longer edge of the resizedimage.
                               If size is an int: if the longer edge of the image is greater than
                               max_size after being resized according to size, size will be
                               overruled so that the longer edge is equal to max_size. As a result,
                               the smaller edge may be shorter than size. This is only supported if
                               size is an int. If size is None: the longer edge of the image will be
                               matched to max_size. i.e, if height > width, then image will be
                               rescaled to (max_size, max_size * width / height).
        antialias (bool, optional) - Whether to apply antialiasing.
                               True (default): will apply antialiasing
                               False: will not apply antialiasing
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

    def _infer_effective_size(
        self,
        size: Optional[Union[int, Sequence[int]]],
        max_size: Optional[int] = None,
    ) -> Tuple[int, int]:

        self.mode = "default"
        if isinstance(size, int):
            if max_size is None:
                # If size is an int, smaller edge of the image will be matched to this number.
                self.mode = "not_smaller"
            else:
                if size > max_size:
                    raise ValueError
                # If size is an int: if the longer edge of the image is greater than max_size
                # after being resized according to size, size will be overruled so that the
                # longer edge is equal to max_size. As a result, the smaller edge may be shorter
                # than size.
                self.mode = "not_larger"
                # size = max_size

            return (size, size)
        if size is None:
            self.mode = "not_larger"
            return (max_size, max_size)
        if isinstance(size, (tuple, list)) and len(size) == 2:
            if max_size is not None:
                raise ValueError(
                    "max_size should only be passed if size specifies the length of the smaller \
                     edge, i.e. size should be an int"
                )
            return size

    def __init__(
        self,
        size: Optional[Union[int, Sequence[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[bool] = True,
        device: Optional[str] = "cpu",
    ):

        self.size = size
        if (
            not isinstance(self.size, int)
            and not isinstance(self.size, (tuple, list))
            and len(self.size) == 2
        ):
            raise ValueError(
                "Invalid combination: size must be int, None, or sequence of two ints. "
                "max_size only applies when size is int or None."
            )

        self.max_size = max_size

        if self.size is None and self.max_size is None:
            raise ValueError("Must provide max_size if size is None.")

        self.interpolation = Resize.interpolation_modes[interpolation]
        self.effective_size = self._infer_effective_size(size, max_size)
        self.antialias = antialias
        self.device = device

    def __call__(self, data_input):
        """
        Performs the resize. The method infers the requested size in compliance
        with Torchvision resize documentation and applies DALI operator on the data_input.

        """
        orig_size = data_input.shape()
        orig_h = orig_size[0]
        orig_w = orig_size[1]
        target_h = self.effective_size[0]
        target_w = self.effective_size[1]
        if self.device == "gpu":
            data_input = data_input.gpu()

        if self.mode == "no_larger" and self.max_size is not None:
            if orig_h > target_h:
                target_h = self.max_size
            if orig_w > target_w:
                target_w = self.max_size

        return fn.resize(
            data_input,
            device=self.device,
            size=fn.stack(
                fn.cast(target_h, dtype=dali.types.FLOAT),
                fn.cast(target_w, dtype=dali.types.FLOAT),
            ),
            mode=self.mode,
        )
