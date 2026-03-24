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

import os
from typing import Sequence, Literal
import unittest

from PIL import Image
import torch
import numpy as np

from nose2.tools import params
from nose_utils import assert_raises
from nvidia.dali.experimental.torchvision import Compose, CenterCrop
from nvidia.dali.experimental.torchvision.v2.functional import center_crop
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as tv_fn

dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]


def build_centercrop_transform(
    size: int | Sequence[int], batch_size: int = 1, device: Literal["cpu", "gpu"] = "cpu"
):
    t = transforms.Compose([transforms.CenterCrop(size=size)])

    td = Compose([CenterCrop(size=size, device=device)], batch_size=batch_size)

    return t, td


def make_tensor(h, w, c=3):
    arr = np.arange(h * w * c, dtype=np.uint8).reshape((c, h, w))
    return torch.tensor(arr)


def _test_core(t, td, inpt, size):
    out_tv = t(inpt)
    out_dali_tv = td(inpt)
    out_tf = tv_fn.center_crop(inpt, size)
    out_dali_tf = center_crop(inpt, size)

    if isinstance(inpt, Image.Image):
        out_tv = tv_fn.pil_to_tensor(out_tv)
        out_dali_tv = tv_fn.pil_to_tensor(out_dali_tv)
        out_tf = tv_fn.pil_to_tensor(out_tf)
        out_dali_tf = tv_fn.pil_to_tensor(out_dali_tf)

    assert (
        out_tv.shape == out_dali_tv.shape
    ), f"Size mismatch expected: {out_tv.shape}, got {out_dali_tv.shape}"
    assert (
        out_tf.shape == out_dali_tf.shape
    ), f"Size mismatch expected: {out_tf.shape}, got {out_dali_tf.shape}"
    assert torch.equal(out_dali_tv.cpu(), out_tv.cpu()), f"Value mismatch for size={size}"
    assert torch.equal(
        out_dali_tf.cpu(), out_tf.cpu()
    ), f"Functional value mismatch for size={size}"


def _test_images(size, device):
    t, td = build_centercrop_transform(size, device=device)
    for filename in test_files:
        img = Image.open(filename)
        _test_core(t, td, img, size)


@params(5, 10, 32, [40, 60], [256, 256], [257, 256], [12, 234], [235, 9])
def test_center_crop_imgs_cpu(size):
    _test_images(size, "cpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
@params(5, 10, 32, [40, 60], [256, 256], [257, 256], [12, 234], [235, 9])
def test_center_crop_imgs_gpu(size):
    _test_images(size, "gpu")


@params(
    6,
    10,
    32,
    128,
    256,
    [4, 6],
    [12, 10],
    [32, 64],
    [
        128,
    ],
)
def test_center_crop_square_int(size):
    tens = make_tensor(256, 256)
    t, td = build_centercrop_transform(size)

    _test_core(t, td, tens, size)


@params((512), (128))
def test_center_crop_larger_than_tensor(size):
    tens = make_tensor(size // 2, size // 2)
    t, td = build_centercrop_transform((size, size))

    out_fn = tv_fn.center_crop(tens, (size, size))
    out_dali_fn = center_crop(tens, (size, size))

    out_tv = t(tens)
    out_dali_tv = td(tens)

    assert out_tv.shape == out_dali_tv.shape, f"Is: {out_dali_tv.shape} should be {out_tv.shape}"
    assert out_fn.shape == out_dali_fn.shape, f"Is: {out_dali_tv.shape} should be {out_tv.shape}"

    center = out_tv[:, size // 4 : size - size // 4, size // 4 : size - size // 4]
    assert torch.equal(center, tens)

    center = out_fn[:, size // 4 : size - size // 4, size // 4 : size - size // 4]
    assert torch.equal(center, tens)

    center = out_dali_tv[:, size // 4 : size - size // 4, size // 4 : size - size // 4]
    assert torch.equal(center, tens)

    center = out_dali_fn[:, size // 4 : size - size // 4, size // 4 : size - size // 4]
    assert torch.equal(center, tens)

    assert torch.equal(out_dali_tv, out_tv), f"Value mismatch for size={size}"
    assert torch.equal(out_dali_fn, out_fn), f"Functional value mismatch for size={size}"


@params(
    ({"bad": "value"},),
)
def test_invalid_type(size):
    with assert_raises(TypeError):
        _ = Compose([CenterCrop(size=size)])

    with assert_raises(TypeError):
        _ = center_crop(torch.ones((3, 256, 256)), output_size=size)


@params(
    [3, 5, 6, 7, 8],
    [0, 5],
    [5, 0],
    [0, 0],
    -2,
    [127, -1],
)
def test_value_error(size):
    with assert_raises(ValueError):
        _ = Compose([CenterCrop(size=size)])

    with assert_raises(ValueError):
        _ = center_crop(torch.ones((3, 256, 256)), output_size=size)


def _test_batched_input_shape(batch_size, device, size):
    batched = torch.randn(batch_size, 3, 12, 14)

    if device == "gpu":
        batched = batched.cuda()

    t, td = build_centercrop_transform(size, batch_size=batch_size, device=device)
    _test_core(t, td, batched, size)


@params(
    (5, "cpu", [8, 8]),
    (1, "cpu", [5, 3]),
    (1, "cpu", [3, 9]),
)
def test_batched_input_shape_cpu(batch_size, device, size):
    _test_batched_input_shape(batch_size, device, size)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
@params(
    (1, "gpu", [5, 3]),
    (16, "gpu", [3, 9]),
)
def test_batched_input_shape_gpu(batch_size, device, size):
    _test_batched_input_shape(batch_size, device, size)
