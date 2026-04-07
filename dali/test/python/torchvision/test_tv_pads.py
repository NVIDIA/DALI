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

import math
import os
import unittest

import torch
from PIL import Image
import torchvision.transforms.v2.functional as tv_fn

from nvidia.dali.experimental.torchvision import Compose, Pad
from nvidia.dali.experimental.torchvision.v2.functional import pad

import torchvision.transforms.v2 as tv

from nose2.tools import cartesian_params

dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg_113 = os.path.join(dali_extra, "db", "single", "jpeg", "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]


def make_test_tensor(shape=(1, 1, 10, 10)):
    return torch.arange(math.prod(shape)).reshape(shape).to(dtype=torch.uint8)


def _build_pad_transforms(mode, padding, fill, device):
    if mode == "constant":
        transform = Compose([Pad(padding=padding, fill=fill, padding_mode=mode, device=device)])
        transform_tv = tv.Pad(padding, fill=fill, padding_mode=mode)
    else:
        transform = Compose([Pad(padding=padding, padding_mode=mode, device=device)])
        transform_tv = tv.Pad(padding, padding_mode=mode)
    return transform, transform_tv


def test_pad_core(mode: str, device: str, padding: list, fill=0):
    img = make_test_tensor(shape=(1, 1, 5, 5))
    transform, transform_tv = _build_pad_transforms(mode, padding, fill, device)

    out = transform(img)
    out_tv = transform_tv(img)
    out_tv_fn = tv.functional.pad(img, padding=padding, padding_mode=mode)
    out_fn = pad(img, padding=padding, padding_mode=mode, device=device)

    if device == "gpu":
        out = out.cpu()
        out_fn = out_fn.cpu()

    assert torch.equal(out, out_tv), f"DALI shape: {out.shape}, TV shape: {out_tv.shape}"
    assert torch.equal(
        out_fn, out_tv_fn
    ), f"DALI shape: {out_fn.shape}, TV shape: {out_tv_fn.shape}"


def _test_pad_pil_images(mode, padding, device, fill=0):
    transform, transform_tv = _build_pad_transforms(mode, padding, fill, device)

    for path in test_files:
        img = Image.open(path)

        out = tv_fn.pil_to_tensor(transform(img))
        out_tv = tv_fn.pil_to_tensor(transform_tv(img))
        out_fn = tv_fn.pil_to_tensor(pad(img, padding=padding, padding_mode=mode, device=device))
        out_tv_fn = tv_fn.pil_to_tensor(tv.functional.pad(img, padding=padding, padding_mode=mode))

        if device == "gpu":
            out = out.cpu()
            out_fn = out_fn.cpu()

        assert torch.equal(out, out_tv), f"Compose mismatch: {path}"
        assert torch.equal(out_fn, out_tv_fn), f"functional mismatch: {path}"


@cartesian_params(
    ("constant", "reflect", "symmetric", "edge"),
    (
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 2, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 3],
    ),
)
def test_pad_symmetric_single_cpu(mode, padding):
    test_pad_core(mode, "cpu", padding)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
@cartesian_params(
    ("constant", "reflect", "symmetric", "edge"),
    (
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 2, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 3],
    ),
)
def test_pad_symmetric_single_gpu(mode, padding):
    test_pad_core(mode, "gpu", padding)


@cartesian_params(
    ("constant", "reflect", "symmetric", "edge"),
    (
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [0, 1, 3, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2],
        [3, 0],
        [0, 3],
    ),
)
def test_pad_multi_cpu(mode: str, padding: list):
    test_pad_core(mode, "cpu", padding)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
@cartesian_params(
    ("constant", "reflect", "symmetric", "edge"),
    (
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [0, 1, 3, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2],
        [3, 0],
        [0, 3],
    ),
)
def test_pad_multi_gpu(mode: str, padding: list):
    test_pad_core(mode, "gpu", padding)


@cartesian_params(
    ([2, 2, 2, 2], [5, 5, 5, 5]),
    (2.0,),
)
def test_constant_pad_cpu(padding: list, fill):
    test_pad_core("constant", "cpu", padding, fill)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
@cartesian_params(
    ([2, 2, 2, 2], [5, 5, 5, 5]),
    (2.0,),
)
def test_constant_pad_gpu(padding: list, fill):
    test_pad_core("constant", "gpu", padding, fill)


@cartesian_params(
    ("constant", "reflect", "symmetric", "edge"),
    ([1, 2, 3, 4], [2, 2, 2, 2], [3, 0]),
)
def test_pad_pil_images_cpu(mode, padding):
    _test_pad_pil_images(mode, padding, "cpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
@cartesian_params(
    ("constant", "reflect", "symmetric", "edge"),
    ([1, 2, 3, 4], [2, 2, 2, 2], [3, 0]),
)
def test_pad_pil_images_gpu(mode, padding):
    _test_pad_pil_images(mode, padding, "gpu")
