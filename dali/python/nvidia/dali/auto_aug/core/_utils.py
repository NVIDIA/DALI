# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Optional

from nvidia.dali.data_node import DataNode as _DataNode

from nvidia.dali.auto_aug.core._select import select
from nvidia.dali.auto_aug.core._args import MissingArgException
from nvidia.dali.auto_aug.core._augmentation import Augmentation
import nvidia.dali.auto_aug.augmentations as a


def max_translate_hw(max_translate):
    if isinstance(max_translate, (tuple, list)):
        height, width = max_translate
        return height, width
    return max_translate, max_translate


def parse_validate_offset(
    use_shape,
    max_translate_abs=None,
    max_translate_rel=None,
    default_translate_abs=250,
    default_translate_rel=1.0,
):
    # if one passes DataNode (with shapes for instance), the error message would be very vague
    if not isinstance(use_shape, bool):
        raise Exception(
            f"The `use_shape` is a flag that should be set to either True or False, "
            f"got {use_shape}."
        )
    if use_shape:
        if max_translate_abs is not None:
            raise Exception(
                "The argument `max_translate_abs` cannot be used with image shapes. "
                "You may use `max_translate_rel` instead."
            )
        if max_translate_rel is None:
            max_translate_rel = default_translate_rel
        return max_translate_hw(max_translate_rel)
    else:
        if max_translate_rel is not None:
            raise Exception(
                "The argument `max_translate_rel` cannot be used without image shapes. "
                "You may use `max_translate_abs` instead."
            )
        if max_translate_abs is None:
            max_translate_abs = default_translate_abs
        return max_translate_hw(max_translate_abs)


def get_translations(
    use_shape: bool,
    default_translate_abs: int,
    default_translate_rel: float,
    max_translate_abs: Optional[int] = None,
    max_translate_rel: Optional[float] = None,
):
    max_translate_height, max_translate_width = parse_validate_offset(
        use_shape,
        max_translate_abs=max_translate_abs,
        max_translate_rel=max_translate_rel,
        default_translate_abs=default_translate_abs,
        default_translate_rel=default_translate_rel,
    )
    if use_shape:
        translate_x = a.translate_x.augmentation((0, max_translate_width), True)
        translate_y = a.translate_y.augmentation((0, max_translate_height), True)
        return [translate_x, translate_y]
    else:
        translate_x = a.translate_x_no_shape.augmentation((0, max_translate_width), True)
        translate_y = a.translate_y_no_shape.augmentation((0, max_translate_height), True)
        return [translate_x, translate_y]


def pretty_select(
    augmentations: List[Augmentation],
    aug_ids: _DataNode,
    op_kwargs,
    auto_aug_name: str,
    ref_suite_name: str,
):
    try:
        return select(augmentations, aug_ids, **op_kwargs)
    except MissingArgException as e:
        if e.missing_args != ["shape"] or e.augmentation.op not in [
            a.translate_x.op,
            a.translate_y.op,
        ]:
            raise
        else:
            raise Exception(
                f"The augmentation `{e.augmentation.name}` requires `shape` argument that "
                f"describes image shape (in HWC layout). Please provide it as `shape` argument "
                f"to `{auto_aug_name}` call. You can get the image shape from encoded "
                f"images with `fn.peek_image_shape`. Alternatively, you can use "
                f"`translate_x_no_shape`/`translate_y_no_shape` that does not rely on image "
                f"shape, but uses offset from fixed range: for reference see `{ref_suite_name}` "
                f"and its `use_shape` argument. "
            )
