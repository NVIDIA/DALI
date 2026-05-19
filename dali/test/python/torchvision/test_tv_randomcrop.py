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
import unittest

from nose2.tools import cartesian_params, params
from nose_utils import assert_raises
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as tv_fn

from nvidia.dali.experimental.torchvision import Compose, RandomCrop
from nvidia.dali.experimental.torchvision.v2.operator import Operator


def make_tensor(shape=(3, 8, 10), dtype=torch.uint8):
    return torch.arange(math.prod(shape), dtype=dtype).reshape(shape)


def make_pil_image(mode="RGB", h=8, w=10, seed=42):
    rng = np.random.default_rng(seed)
    if mode == "L":
        data = rng.integers(0, 256, (h, w), dtype=np.uint8)
    elif mode == "RGB":
        data = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif mode == "RGBA":
        data = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return Image.fromarray(data, mode=mode)


def _to_tensor(inpt):
    if isinstance(inpt, Image.Image):
        return tv_fn.pil_to_tensor(inpt)
    return inpt


def _assert_equal_to_torchvision(inpt, dali_transform, tv_transform, device="cpu"):
    out = dali_transform(inpt)
    tv_out = tv_transform(inpt)

    out = _to_tensor(out)
    tv_out = _to_tensor(tv_out)
    if device == "gpu":
        out = out.cpu()
        if isinstance(tv_out, torch.Tensor):
            tv_out = tv_out.cpu()

    assert out.shape == tv_out.shape, f"Shape mismatch: {out.shape} != {tv_out.shape}"
    assert torch.equal(out, tv_out), "DALI RandomCrop output differs from torchvision"


def _build_dali_random_crop(**kwargs):
    batch_size = kwargs.pop("batch_size", 1)
    return Compose([RandomCrop(**kwargs)], batch_size=batch_size)


def _skip_if_gpu_unavailable(device):
    if device == "gpu" and not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")


def _move_tensor_to_device(inpt, device):
    if device == "gpu" and isinstance(inpt, torch.Tensor):
        return inpt.cuda()
    return inpt


def test_random_crop_is_operator():
    assert issubclass(RandomCrop, Operator)


@cartesian_params(
    ("cpu", "gpu"),
    (
        ("tensor", (3, 8, 10), (8, 10)),
        ("tensor", (4, 3, 8, 10), (8, 10)),
        ("pil", "L", (8, 10)),
        ("pil", "RGB", (8, 10)),
        ("pil", "RGBA", (8, 10)),
    ),
)
def test_random_crop_identity_matches_torchvision(device, input_case):
    _skip_if_gpu_unavailable(device)
    input_type, input_arg, size = input_case
    inpt = make_pil_image(input_arg) if input_type == "pil" else make_tensor(shape=input_arg)
    inpt = _move_tensor_to_device(inpt, device)
    batch_size = inpt.shape[0] if isinstance(inpt, torch.Tensor) and inpt.ndim > 3 else 1
    _assert_equal_to_torchvision(
        inpt,
        _build_dali_random_crop(size=size, device=device, batch_size=batch_size),
        transforms.RandomCrop(size=size),
        device=device,
    )


@cartesian_params(
    ("cpu", "gpu"),
    (
        (None, 0, "constant"),
        (1, 0, "constant"),
        ([1], 0, "constant"),
        ([1, 1], 0, "constant"),
        ([1, 1, 1, 1], 0, "constant"),
        (1, 7, "constant"),
        (1, (1, 2, 3), "constant"),
        (1, None, "constant"),
        (1, 0, "edge"),
        (1, 0, "reflect"),
        (1, 0, "symmetric"),
    ),
)
def test_random_crop_padding_matches_torchvision_tensor(device, padding_case):
    _skip_if_gpu_unavailable(device)
    padding, fill, padding_mode = padding_case
    tensor = _move_tensor_to_device(make_tensor(shape=(3, 4, 5)), device)
    size = (4, 5) if padding is None else (6, 7)

    _assert_equal_to_torchvision(
        tensor,
        _build_dali_random_crop(
            size=size,
            padding=padding,
            fill=fill,
            padding_mode=padding_mode,
            device=device,
        ),
        transforms.RandomCrop(
            size=size,
            padding=padding,
            fill=fill,
            padding_mode=padding_mode,
        ),
        device=device,
    )


@cartesian_params(("cpu", "gpu"), ("L", "RGB", "RGBA"))
def test_random_crop_padding_matches_torchvision_pil(device, mode):
    _skip_if_gpu_unavailable(device)
    img = make_pil_image(mode=mode, h=4, w=5)
    _assert_equal_to_torchvision(
        img,
        _build_dali_random_crop(size=(6, 7), padding=1, fill=3, device=device),
        transforms.RandomCrop(size=(6, 7), padding=1, fill=3),
        device=device,
    )


"""
# TODO: Fill using dictionary pattern is currently not supported
def test_random_crop_fill_dict_matches_torchvision_tensor():
    tensor = make_tensor(shape=(3, 4, 5))
    fill = {torch.Tensor: 9}
    _assert_equal_to_torchvision(
        tensor,
        _build_dali_random_crop(size=(6, 7), padding=1, fill=fill),
        transforms.RandomCrop(size=(6, 7), padding=1, fill=fill),
    )

def test_random_crop_fill_dict_matches_torchvision_pil():
    img = make_pil_image(mode="RGB", h=4, w=5)
    fill = {Image.Image: (1, 2, 3)}
    _assert_equal_to_torchvision(
        img,
        _build_dali_random_crop(size=(6, 7), padding=1, fill=fill),
        transforms.RandomCrop(size=(6, 7), padding=1, fill=fill),
    )
"""


@cartesian_params(
    ("cpu", "gpu"),
    (
        (4, (4, 4)),
        ([4, 5], (4, 5)),
    ),
)
def test_random_crop_tensor_shape(device, shape_case):
    _skip_if_gpu_unavailable(device)
    size, expected_hw = shape_case
    tensor = _move_tensor_to_device(make_tensor(), device)
    out = _build_dali_random_crop(size=size, device=device)(tensor)

    assert out.shape == (3, *expected_hw)


@params("cpu", "gpu")
def test_random_crop_pad_if_needed_shape(device):
    if device == "gpu" and not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")

    tensor = make_tensor(shape=(3, 4, 5))
    if device == "gpu":
        tensor = tensor.cuda()
    out = _build_dali_random_crop(size=(6, 7), pad_if_needed=True, device=device)(tensor)

    assert out.shape == (3, 6, 7)


@params(
    [],
    [0, 5],
    [5, 0],
    [1.0, 2],
    [1, 2, 3],
    -1,
    1.0,
    {"bad": "value"},
)
def test_random_crop_invalid_size(size):
    with assert_raises((TypeError, ValueError)):
        _ = RandomCrop(size=size)


@params(
    -1,
    [1, -1],
    [1, 2, 3],
    [1.0],
    "bad",
)
def test_random_crop_invalid_padding(padding):
    with assert_raises((TypeError, ValueError)):
        _ = RandomCrop(size=3, padding=padding)


def test_random_crop_invalid_pad_if_needed():
    with assert_raises(TypeError):
        _ = RandomCrop(size=3, pad_if_needed="yes")


@params(
    object(),
    "bad",
    [1, object()],
    {object(): 1},
    {torch.Tensor: object()},
)
def test_random_crop_invalid_fill(fill):
    with assert_raises(TypeError):
        _ = RandomCrop(size=3, padding=1, fill=fill)


def test_random_crop_invalid_padding_mode_when_padding_is_used():
    with assert_raises(ValueError):
        _ = RandomCrop(size=3, padding=1, padding_mode="bad")


def test_random_crop_invalid_padding_mode_when_pad_if_needed_is_used():
    with assert_raises(ValueError):
        _ = RandomCrop(size=3, pad_if_needed=True, padding_mode="bad")
