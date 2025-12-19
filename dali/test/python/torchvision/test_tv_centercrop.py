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

from typing import Sequence, Union, Literal

import torch
import numpy as np

from nose2.tools import params
from nose_utils import assert_raises
from nvidia.dali.experimental.torchvision import Compose, CenterCrop
import torchvision.transforms.v2 as transforms


def build_centercrop_transform(
    size: Union[int, Sequence[int]], batch_size: int = 1, device: Literal["cpu", "gpu"] = "cpu"
):
    t = transforms.Compose([transforms.CenterCrop(size=size)])

    td = Compose([CenterCrop(size=size, device=device)], batch_size=batch_size)

    return t, td


def make_img(h, w, c=3):
    arr = np.arange(h * w * c, dtype=np.uint8).reshape((c, h, w))
    return torch.tensor(arr)


@params(
    (6),
    (10),
    (32),
    (128),
    (256),
    ([4, 6]),
    ([12, 10]),
    ([32, 64]),
    (
        [
            128,
        ]
    ),
)
def test_center_crop_square_int(size):
    img = make_img(256, 256)
    t, td = build_centercrop_transform(size)

    out_tv = t(img)
    out_dali_tv = td(img).cpu()

    assert (
        out_tv.shape[-3:] == out_dali_tv.shape[-3:]
    ), f"Is: {out_dali_tv.shape} should be: {out_tv.shape}"


def test_center_crop_equal_size():
    img = make_img(256, 256)
    t, td = build_centercrop_transform((256, 256))

    out_tv = t(img)
    out_dali_tv = td(img).cpu()

    assert torch.equal(img, out_tv)
    assert torch.equal(img, out_dali_tv)


@params((512), (128))
def test_center_crop_larger_than_image(size):
    img = make_img(size // 2, size // 2)
    t, td = build_centercrop_transform((size, size))

    out_tv = t(img)
    out_dali_tv = td(img).cpu()

    assert out_tv.shape == out_dali_tv.shape, f"Is: {out_dali_tv.shape} should be {out_tv.shape}"

    center = out_tv[:, size // 4 : size - size // 4, size // 4 : size - size // 4]
    assert torch.equal(center, img)
    center = out_dali_tv[:, size // 4 : size - size // 4, size // 4 : size - size // 4]
    assert torch.equal(center, img)


@params(
    ({"bad": "value"},),
)
def test_invalid_type(size):
    with assert_raises(TypeError):
        _ = Compose([CenterCrop(size=size)])


@params(
    ([3, 5, 6, 7, 8],),
    ([0, 5],),
    ([5, 0],),
    ([0, 0],),
    (-2,),
    ([127, -1],),
)
def test_value_error(size):
    with assert_raises(ValueError):
        _ = Compose([CenterCrop(size=size)])


@params(
    (5, "cpu", [8, 8]),
    (1, "gpu", [5, 3]),
    (16, "gpu", [3, 9]),
)
def test_batched_input_shape(batch_size, device, size):
    batched = torch.randn(batch_size, 3, 12, 14)

    t, td = build_centercrop_transform(size, batch_size=batch_size, device=device)
    out_tv = t(batched)
    out_dali_tv = td(batched).cpu()

    assert out_tv.shape == out_dali_tv.shape
