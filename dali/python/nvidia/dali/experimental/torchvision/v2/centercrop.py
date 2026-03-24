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

from typing import Sequence, Literal
from .operator import Operator, VerifSizeDescriptor
import nvidia.dali as dali
import nvidia.dali.fn as fn
from .operator import get_HWC_from_layout_pipeline


class CenterCrop(Operator):
    """
    Crop the input at the center.

    If the input is a ``torch.Tensor`` it can have an arbitrary number of leading batch dimensions.
    For example, the image tensor can have [..., C, H, W] shape.

    If image size is smaller than output size along any edge, image is padded with 0 and then
    center cropped.

    Parameters
    ----------
    size : sequence or int
        Desired output size of the crop. If size is an int instead of sequence like (h, w),
        a square crop (size, size) is made. If provided a sequence of length 1, it will be
        interpreted as (size[0], size[0]).
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the crop. Can be ``"cpu"`` or ``"gpu"``.
    """

    arg_rules = [VerifSizeDescriptor]
    preprocess_data = get_HWC_from_layout_pipeline

    @classmethod
    def adjust_size(cls, size: int | Sequence[int]) -> Sequence[int]:
        if isinstance(size, int):
            return (size, size)
        elif isinstance(size, (list, tuple)):
            if len(size) == 0 or len(size) > 2:
                raise ValueError(f"Invalid size length, expected 1 or 2, got {len(size)}")
            elif len(size) == 1:
                return (size[0], size[0])
            else:
                return tuple(size)

        else:
            raise TypeError(f"Invalid size type expected int, list or tuple, got {type(size)}")

    def __init__(self, size: int | Sequence[int], device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device=device, size=size)

        self.size = CenterCrop.adjust_size(size)

    def _kernel(self, data_input):
        """
        Applies the center crop to the input data.

        If image size is smaller than output size along any edge, image is padded with 0 and then
        center cropped.
        """
        in_h, in_w, _, tensor = data_input
        crop_h, crop_w = self.size

        # Slack between image and crop along each axis (may be zero or negative).
        N_h = fn.cast(in_h, dtype=dali.types.INT32) - crop_h
        N_w = fn.cast(in_w, dtype=dali.types.INT32) - crop_w

        # Banker's-rounded (round-half-to-even) half of N, matching Python's round() and
        # torchvision's int(round((dim - crop_dim) / 2.0)).
        #
        # Formula (integer arithmetic, no modulo needed):
        #   floor_half  = floor(N / 2)
        #   floor_quarter = floor(N / 4)
        #   half = floor_half + (floor_half - 2*floor_quarter) * (N - 2*floor_half)
        #        = floor(N/2) + (floor(N/2) % 2) * (N % 2)
        # Adds 1 only when floor(N/2) is odd AND N is odd (i.e. N % 4 == 3).
        floor_half_h = N_h // 2
        floor_quarter_h = N_h // 4
        half_h = floor_half_h + (floor_half_h - 2 * floor_quarter_h) * (N_h - 2 * floor_half_h)

        floor_half_w = N_w // 2
        floor_quarter_w = N_w // 4
        half_w = floor_half_w + (floor_half_w - 2 * floor_quarter_w) * (N_w - 2 * floor_half_w)

        # Compute normalised position for fn.crop:
        #   N > 0  (no padding): crop_pos = half / N  (exact round-trip through fn.crop)
        #   N = 0  (crop = image): position is irrelevant; fn.crop gives 0 regardless
        #   N < 0  (crop > image): use 0.5 so out_of_bounds_policy pads symmetrically
        #
        # Implementation avoids Python conditionals on DALI nodes:
        #   is_pos   = 1.0 if N > 0, else 0.0
        #   N_safe   = max(N, 1)            <- avoids division by zero
        #   crop_pos = is_pos * (half / N_safe) + (1 - is_pos) * 0.5
        is_pos_h = N_h > 0
        is_pos_w = N_w > 0
        N_h_safe = dali.math.max(N_h, 1)
        N_w_safe = dali.math.max(N_w, 1)

        crop_pos_y = is_pos_h * half_h / N_h_safe + (1.0 - is_pos_h) * 0.5
        crop_pos_x = is_pos_w * half_w / N_w_safe + (1.0 - is_pos_w) * 0.5

        return fn.crop(
            tensor,
            device=self.device,
            crop=self.size,
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
            out_of_bounds_policy="pad",
            fill_values=0,
        )
