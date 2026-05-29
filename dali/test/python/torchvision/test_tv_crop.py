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
import torchvision.transforms.v2.functional as tv_fn

from nvidia.dali.experimental.torchvision.v2.functional import crop


def make_test_tensor(shape=(3, 8, 10), dtype=torch.uint8):
    return torch.arange(math.prod(shape), dtype=dtype).reshape(shape)


def _make_pil_image(mode, h=8, w=10, seed=42):
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


def _assert_crop_matches_torchvision(inpt, top, left, height, width, device="cpu"):
    dali_out = crop(inpt, top, left, height, width, device=device)
    tv_out = tv_fn.crop(inpt, top, left, height, width)

    if device == "gpu" and not isinstance(dali_out, Image.Image):
        dali_out = dali_out.cpu()
        if isinstance(tv_out, torch.Tensor):
            tv_out = tv_out.cpu()

    if isinstance(inpt, Image.Image):
        assert isinstance(dali_out, Image.Image), f"Expected PIL Image, got {type(dali_out)}"
        assert dali_out.mode == tv_out.mode, f"Expected mode {tv_out.mode}, got {dali_out.mode}"
        dali_out = tv_fn.pil_to_tensor(dali_out)
        tv_out = tv_fn.pil_to_tensor(tv_out)

    assert dali_out.shape == tv_out.shape, f"Shape mismatch: {dali_out.shape} != {tv_out.shape}"
    assert torch.equal(dali_out, tv_out), "DALI crop output differs from torchvision"


def _skip_if_gpu_unavailable(device):
    if device == "gpu" and not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")


def _move_tensor_to_device(tensor, device):
    if device == "gpu":
        return tensor.cuda()
    return tensor


@cartesian_params(
    ("cpu", "gpu"),
    (
        (1, 2, 4, 5),
        (0, 0, 8, 10),
        (3, 4, 2, 3),
        (-1, -2, 6, 8),
        (6, 8, 5, 6),
        (0, 0, 12, 14),
        (np.int32(5), np.int32(2), np.int32(2), np.int32(3)),
        (np.int64(-1), np.int64(-2), np.uint16(6), np.uint16(8)),
    ),
)
def test_crop_tensor(device, crop_case):
    _skip_if_gpu_unavailable(device)
    tensor = _move_tensor_to_device(make_test_tensor(), device)
    _assert_crop_matches_torchvision(tensor, *crop_case, device=device)


@cartesian_params(
    ("cpu", "gpu"),
    ("L", "RGB", "RGBA"),
    (
        (1, 2, 4, 5),
        (-2, -3, 12, 14),
        (1.5, 2.5, 4.0, 5.0),
        (np.float32(1.5), np.float64(2.5), np.float32(4.5), np.float64(5.5)),
    ),
)
def test_crop_pil(device, mode, crop_case):
    _skip_if_gpu_unavailable(device)
    _assert_crop_matches_torchvision(_make_pil_image(mode), *crop_case, device=device)


@cartesian_params(("cpu", "gpu"), ((2, 3, 4, 5), (np.int32(2), np.int64(3), 4, 5)))
def test_crop_batched_tensor(device, crop_case):
    _skip_if_gpu_unavailable(device)
    tensor = _move_tensor_to_device(make_test_tensor(shape=(4, 3, 8, 10)), device)
    _assert_crop_matches_torchvision(tensor, *crop_case, device=device)


@params(torch.float32, torch.int16, torch.int32)
def test_crop_preserves_tensor_dtype(dtype):
    tensor = make_test_tensor(dtype=dtype)
    dali_out = crop(tensor, top=1, left=1, height=4, width=5)
    tv_out = tv_fn.crop(tensor, top=1, left=1, height=4, width=5)

    assert dali_out.dtype == tv_out.dtype, f"Expected dtype {tv_out.dtype}, got {dali_out.dtype}"
    assert torch.equal(dali_out, tv_out), "DALI crop output differs from torchvision"


@params(
    dict(top=1.0, left=0, height=1, width=1),
    dict(top=0, left=1.0, height=1, width=1),
    dict(top=0, left=0, height=1.0, width=1),
    dict(top=0, left=0, height=1, width=1.0),
)
def test_crop_tensor_rejects_float_parameters(crop_kwargs):
    with assert_raises(TypeError, glob="*integer*"):
        _ = tv_fn.crop(make_test_tensor(), **crop_kwargs)
    with assert_raises(TypeError, glob="*integer*"):
        _ = crop(make_test_tensor(), **crop_kwargs)


@params(
    dict(top="0", left=0, height=1, width=1),
    dict(top=0, left="0", height=1, width=1),
    dict(top=0, left=0, height="1", width=1),
    dict(top=0, left=0, height=1, width="1"),
)
def test_crop_pil_rejects_non_numeric_parameters(crop_kwargs):
    pil_image = _make_pil_image("RGB")
    with assert_raises(TypeError, glob="*str*"):
        _ = tv_fn.crop(pil_image, **crop_kwargs)
    with assert_raises(TypeError, glob="*real numbers*"):
        _ = crop(pil_image, **crop_kwargs)


def test_crop_invalid_input_type():
    with assert_raises(TypeError, glob="*support*"):
        _ = tv_fn.crop([1, 2, 3], top=0, left=0, height=1, width=1)
    with assert_raises(TypeError, glob="*support*"):
        _ = crop([1, 2, 3], top=0, left=0, height=1, width=1)


@params(
    (0, 1),
    (1, 0),
    (-1, 1),
    (1, -1),
    (1.0, 1),
    (1, 1.0),
)
def test_crop_invalid_output_size(height, width):
    with assert_raises((TypeError, ValueError), glob="*must be*"):
        _ = crop(make_test_tensor(), top=0, left=0, height=height, width=width)


@params(
    (0.5, 0),
    ("0", 0),
    (0, 0.5),
    (0, "0"),
)
def test_crop_invalid_coordinates(top, left):
    with assert_raises(TypeError, glob="*int*"):
        _ = tv_fn.crop(make_test_tensor(), top=top, left=left, height=1, width=1)
    with assert_raises(TypeError, glob="*int*"):
        _ = crop(make_test_tensor(), top=top, left=left, height=1, width=1)
