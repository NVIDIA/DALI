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

from typing import Sequence, Union

import torch
import numpy as np

from nose2.tools import params
from nose_utils import assert_raises
from nvidia.dali.experimental.torchvision import Compose, CenterCrop
import torchvision.transforms.v2 as transforms


def build_centercrop_transform(size: Union[int, Sequence[int]]):
    t = transforms.Compose([transforms.CenterCrop(size=size)])

    td = Compose([CenterCrop(size=size)])

    return t, td


def make_img(h, w, c=3):
    arr = np.arange(h * w * c, dtype=np.uint8).reshape((c, h, w))
    arr_dali = np.arange(h * w * c, dtype=np.uint8).reshape((h, w, c))
    return torch.tensor(arr), torch.tensor(arr_dali).unsqueeze(0)


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
    img, img_dali = make_img(256, 256)
    t, td = build_centercrop_transform(size)

    out_tv = t(img)
    out_dali_tv = td(img_dali).cpu()

    if isinstance(size, list) and len(size) == 1:
        size = size[0]
    if isinstance(size, int):
        assert out_tv.shape[-2:] == (size, size)
        assert out_dali_tv.shape[1:3] == (size, size)
    else:
        assert out_tv.shape[-2:] == tuple(size)
        assert out_dali_tv.shape[1:3] == tuple(size)


def test_center_crop_equal_size():
    img, img_dali = make_img(256, 256)
    t, td = build_centercrop_transform((256, 256))

    out_tv = t(img)
    out_dali_tv = td(img_dali).cpu()

    assert torch.equal(img, out_tv)
    assert torch.equal(img_dali, out_dali_tv)


@params((512), (128))
def test_center_crop_larger_than_image(size):
    img, img_dali = make_img(size // 2, size // 2)
    t, td = build_centercrop_transform((size, size))

    out_tv = t(img)
    out_dali_tv = td(img_dali).cpu()

    assert out_tv.shape[-2:] == (size, size)
    assert out_dali_tv.shape[1:3] == (size, size)

    center = out_tv[:, size // 4 : size - size // 4, size // 4 : size - size // 4]
    assert torch.equal(center, img)
    center = out_dali_tv[:, size // 4 : size - size // 4, size // 4 : size - size // 4, :]
    assert torch.equal(center, img_dali)


@params(
    ({"bad": "value"},),
)
def test_invalid_type(size):
    with assert_raises(TypeError):
        td = Compose([CenterCrop(size=size)])
        del td


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
        td = Compose([CenterCrop(size=size)])
        del td


"""
TODO: decide if this is our use case
def test_batched_input_shape():
    batched = torch.randn(2, 3, 12, 14)
    crop = CenterCrop((8, 8))
    out = crop(batched)
    assert out.shape == (2, 3, 8, 8)
"""
