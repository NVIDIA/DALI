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

from typing import Union, Sequence
from .operator import Operator, VerificationSize
import nvidia.dali.fn as fn


class CenterCrop(Operator):
    """
    Crop the input at the center.

    If the input is a torch.Tensor or a TVTensor (e.g. Image, Video, BoundingBoxes etc.)
    it can have arbitrary number of leading batch dimensions. For example, the image can
    have [..., C, H, W] shape. A bounding box can have [..., 4] shape.

    If image size is smaller than output size along any edge, image is padded with 0
    and then center cropped.

    Parameters:
        size (sequence or int) – Desired output size of the crop. If size is an int
                               instead of sequence like (h, w), a square crop (size, size)
                               is made. If provided a sequence of length 1, it will be
                               interpreted as (size[0], size[0]).
    """

    arg_rules = [VerificationSize]

    @staticmethod
    def adjust_size(size: int | Sequence[int]) -> Sequence[int]:
        if isinstance(size, int):
            return (size, size)
        elif isinstance(size, (list, tuple)):
            if len(size) == 1:
                return (size[0], size[0])
            elif len(size) == 2:
                return tuple(size)

        return size

    def __init__(self, size: Union[int, Sequence[int]], device: str = "cpu"):
        """
        Initialize CenterCrop with the desired output size.

        Args:
            size: Desired output size of the crop. Can be:
                - int: Creates square crop (size, size)
                - sequence of length 1: Interpreted as (size[0], size[0])
                - sequence of length 2: Used as (height, width)

        Raises:
            TypeError: If size is not int or sequence
            ValueError: If size values are not positive integers or sequence length is invalid
        """
        super().__init__(device=device, size=size)

        self.size = CenterCrop.adjust_size(size)

    def _kernel(self, data):
        """
        Apply center crop to the input data.

        If image size is smaller than output size along any edge, image is padded
        with 0 and then center cropped.

        Args:
            data: Input data (image or tensor) to be center cropped

        Returns:
            Center cropped data using DALI operations
        """
        return fn.crop(
            data,
            device=self.device,
            crop=self.size,
            crop_pos_x=0.5,
            crop_pos_y=0.5,
            out_of_bounds_policy="pad",
            fill_values=0,
        )
