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

import nvidia.dali.experimental.torchvision.v2.randomcrop as randomcrop_module
from nvidia.dali.experimental.torchvision import Compose, RandomResizedCrop
from nvidia.dali.experimental.torchvision.v2.operator import Operator
from nvidia.dali.experimental.torchvision.v2.resize import Resize


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


def _build_dali_random_resized_crop(**kwargs):
    batch_size = kwargs.pop("batch_size", 1)
    return Compose([RandomResizedCrop(**kwargs)], batch_size=batch_size)


def _skip_if_gpu_unavailable(device):
    if device == "gpu" and not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")


def _move_tensor_to_device(inpt, device):
    if device == "gpu" and isinstance(inpt, torch.Tensor):
        return inpt.cuda()
    return inpt


def _assert_allclose_to_torchvision(inpt, dali_transform, tv_transform, device="cpu", atol=1):
    out = dali_transform(inpt)
    tv_out = tv_transform(inpt)

    out = _to_tensor(out)
    tv_out = _to_tensor(tv_out)
    if device == "gpu":
        out = out.cpu()
        if isinstance(tv_out, torch.Tensor):
            tv_out = tv_out.cpu()

    assert out.shape == tv_out.shape, f"Shape mismatch: {out.shape} != {tv_out.shape}"
    torch.testing.assert_close(
        out,
        tv_out,
        rtol=0,
        atol=atol,
        check_stride=True,
        msg="DALI output differs from torchvision",
    )


def test_random_resized_crop_is_operator():
    assert issubclass(RandomResizedCrop, Operator)


def test_random_resized_crop_uses_dali_operator():
    transform = RandomResizedCrop(
        size=(4, 5),
        scale=(0.5, 1.0),
        ratio=(0.75, 1.25),
        interpolation=transforms.InterpolationMode.NEAREST,
        antialias=False,
    )
    calls = []

    def fake_random_resized_crop(tensor, **kwargs):
        calls.append((tensor, kwargs))
        return "cropped"

    old_random_resized_crop = randomcrop_module.fn.random_resized_crop
    try:
        randomcrop_module.fn.random_resized_crop = fake_random_resized_crop
        out = transform._kernel((8, 10, 3, "input"))
    finally:
        randomcrop_module.fn.random_resized_crop = old_random_resized_crop

    assert out == "cropped"
    assert len(calls) == 1
    tensor, kwargs = calls[0]
    assert tensor == "input"
    assert kwargs["size"] == (4, 5)
    assert kwargs["random_area"] == (0.5, 1.0)
    assert kwargs["random_aspect_ratio"] == (0.75, 1.25)
    assert kwargs["interp_type"] == Resize.interpolation_modes[transforms.InterpolationMode.NEAREST]
    assert kwargs["antialias"] is False
    assert kwargs["num_attempts"] == 10


@cartesian_params(
    ("cpu", "gpu"),
    (
        ("tensor", (3, 8, 10)),
        ("tensor", (4, 3, 8, 10)),
        ("pil", "L"),
        ("pil", "RGB"),
        ("pil", "RGBA"),
    ),
)
def test_random_resized_crop_identity_matches_torchvision(device, input_case):
    _skip_if_gpu_unavailable(device)
    input_type, input_arg = input_case
    inpt = make_pil_image(input_arg) if input_type == "pil" else make_tensor(shape=input_arg)
    inpt = _move_tensor_to_device(inpt, device)
    batch_size = inpt.shape[0] if isinstance(inpt, torch.Tensor) and inpt.ndim > 3 else 1

    kwargs = {
        "size": (8, 10),
        "scale": (1.0, 1.0),
        "ratio": (10.0 / 8.0, 10.0 / 8.0),
        "interpolation": transforms.InterpolationMode.NEAREST,
        "antialias": False,
    }

    _assert_allclose_to_torchvision(
        inpt,
        _build_dali_random_resized_crop(**kwargs, device=device, batch_size=batch_size),
        transforms.RandomResizedCrop(**kwargs),
        device=device,
        atol=0,
    )


@cartesian_params(
    ("cpu", "gpu"),
    (
        (4, (4, 4)),
        ([4], (4, 4)),
        ([4, 5], (4, 5)),
        ((5, 4), (5, 4)),
    ),
)
def test_random_resized_crop_tensor_shape(device, shape_case):
    _skip_if_gpu_unavailable(device)
    size, expected_hw = shape_case
    tensor = _move_tensor_to_device(make_tensor(), device)
    out = _build_dali_random_resized_crop(
        size=size,
        scale=(0.5, 1.0),
        ratio=(0.75, 1.3333333333333333),
        device=device,
    )(tensor)

    assert out.shape == (3, *expected_hw)


@cartesian_params(("cpu", "gpu"))
def test_random_resized_crop_batched_tensor_shape(device):
    _skip_if_gpu_unavailable(device)
    tensor = _move_tensor_to_device(make_tensor(shape=(4, 3, 8, 10)), device)
    out = _build_dali_random_resized_crop(
        size=(4, 5),
        scale=(0.5, 1.0),
        ratio=(0.75, 1.3333333333333333),
        device=device,
        batch_size=4,
    )(tensor)

    assert out.shape == (4, 3, 4, 5)


@cartesian_params(("cpu", "gpu"))
def test_random_resized_crop_samples_different_crops(device):
    _skip_if_gpu_unavailable(device)
    tensor = _move_tensor_to_device(make_tensor(shape=(3, 32, 40)), device)
    transform = _build_dali_random_resized_crop(
        size=(8, 10),
        scale=(0.2, 0.8),
        ratio=(0.75, 1.3333333333333333),
        interpolation=transforms.InterpolationMode.NEAREST,
        antialias=False,
        device=device,
    )

    outputs = {bytes(transform(tensor).cpu().numpy().tobytes()) for _ in range(20)}

    assert len(outputs) > 1, "RandomResizedCrop produced the same crop for every run"


@cartesian_params(
    (
        transforms.InterpolationMode.NEAREST,
        transforms.InterpolationMode.BILINEAR,
        transforms.InterpolationMode.BICUBIC,
    ),
    ("cpu", "gpu"),
)
def test_random_resized_crop_interpolation_shape(interpolation, device):
    _skip_if_gpu_unavailable(device)
    tensor = _move_tensor_to_device(make_tensor(), device)
    out = _build_dali_random_resized_crop(
        size=(4, 5),
        scale=(1.0, 1.0),
        ratio=(10.0 / 8.0, 10.0 / 8.0),
        interpolation=interpolation,
        antialias=False,
        device=device,
    )(tensor)

    assert out.shape == (3, 4, 5)


def test_random_resized_crop_unsupported_interpolation():
    with assert_raises(NotImplementedError, glob="*Interpolation mode*"):
        _ = RandomResizedCrop(size=3, interpolation=transforms.InterpolationMode.NEAREST_EXACT)


@cartesian_params((True, False), ("cpu", "gpu"))
def test_random_resized_crop_antialias_shape(antialias, device):
    _skip_if_gpu_unavailable(device)
    tensor = _move_tensor_to_device(make_tensor(), device)
    out = _build_dali_random_resized_crop(
        size=(4, 5),
        scale=(1.0, 1.0),
        ratio=(10.0 / 8.0, 10.0 / 8.0),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=antialias,
        device=device,
    )(tensor)

    assert out.shape == (3, 4, 5)


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
def test_random_resized_crop_invalid_size(size):
    with assert_raises((TypeError, ValueError), glob="*size*"):
        _ = RandomResizedCrop(size=size)


@params(
    ("scale", object()),
    ("scale", "bad"),
    ("scale", [1]),
    ("scale", [1, 2, 3]),
    ("scale", [1, object()]),
    ("scale", [-1, 1]),
    ("scale", [1, 0.5]),
    ("ratio", object()),
    ("ratio", "bad"),
    ("ratio", [1]),
    ("ratio", [1, 2, 3]),
    ("ratio", [1, object()]),
    ("ratio", [0, 1]),
    ("ratio", [1, 0.5]),
)
def test_random_resized_crop_invalid_scale_ratio(name, value):
    kwargs = {name: value}
    with assert_raises((TypeError, ValueError), glob=f"*{name}*"):
        _ = RandomResizedCrop(size=3, **kwargs)


def test_random_resized_crop_invalid_interpolation():
    with assert_raises(ValueError, glob="*Interpolation*"):
        _ = RandomResizedCrop(size=3, interpolation="bad")


def test_random_resized_crop_int_interpolation_normalizes_to_enum():
    transform = RandomResizedCrop(size=(4, 5), interpolation=2)
    expected = Resize.interpolation_modes[transforms.InterpolationMode.BILINEAR]
    assert transform.interpolation == expected


def test_random_resized_crop_invalid_int_interpolation():
    with assert_raises(ValueError, glob="*PIL code*"):
        _ = RandomResizedCrop(size=3, interpolation=99)
