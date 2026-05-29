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

import unittest

from nose2.tools import cartesian_params, params
from nose_utils import assert_raises
from PIL import Image
import torch
from torchvision import tv_tensors
import torchvision.transforms.v2.functional as fn_tv

from nvidia.dali.experimental.torchvision.v2.functional import (
    get_image_size,
    get_dimensions,
    get_size,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tv_get_image_size(inpt):
    """Call torchvision get_size and adjust the output to be aligned with get_image_size."""
    w, h = fn_tv.get_size(inpt)
    return [h, w]


def _skip_if_gpu_unavailable(device):
    if device == "gpu" and not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")


def _move_tensor_to_device(tensor, device):
    if device == "gpu":
        return tensor.cuda()
    return tensor


def _make_compatibility_input(input_kind, shape):
    tensor = torch.zeros(*shape)
    if input_kind == "tensor":
        return tensor
    if input_kind == "tv_image":
        return tv_tensors.Image(tensor)
    raise ValueError(f"Unsupported input kind: {input_kind}")


# PIL images with known exact dimensions (W x H)
PIL_CASES = [
    Image.new("RGB", (320, 240)),  # 3 channels
    Image.new("L", (100, 50)),  # 1 channel, non-square
    Image.new("RGBA", (64, 32)),  # 4 channels
    Image.new("RGB", (1, 1)),  # minimal
    Image.new("L", (512, 1)),  # extreme aspect ratio
]

# Tensors in CHW / NCHW layout — deliberately use H≠W to catch W/H swap bugs
TENSOR_CASES = [
    torch.zeros(3, 240, 320),  # CHW
    torch.zeros(1, 3, 240, 320),  # NCHW, N=1
    torch.zeros(8, 3, 240, 320),  # NCHW, N=8
    torch.zeros(1, 50, 100),  # CHW, 1 channel
    torch.zeros(4, 32, 64),  # CHW, 4 channels
    torch.zeros(10, 11, 12, 8, 3, 240, 320),  # ...NCHW, N=8
]

TORCHVISION_COMPATIBILITY_CASES = [
    ("tensor", (240, 320)),  # HW, implicit single channel
    ("tensor", (3, 240, 320)),  # CHW
    ("tensor", (8, 3, 240, 320)),  # NCHW
    ("tv_image", (240, 320)),  # torchvision Image converts HW to 1HW
    ("tv_image", (3, 240, 320)),  # torchvision Image, CHW
]


# ---------------------------------------------------------------------------
# get_(image_)size — PIL
# ---------------------------------------------------------------------------


@params(*PIL_CASES)
def test_get_image_size_pil(img):
    expected = _tv_get_image_size(img)
    assert (
        get_image_size(img) == expected
    ), f"mode={img.mode} size={img.size}: got {get_image_size(img)}, expected {expected}"


@params(*PIL_CASES)
def test_get_size_pil(img):
    expected = fn_tv.get_size(img)
    assert (
        get_size(img) == expected
    ), f"mode={img.mode} size={img.size}: got {get_size(img)}, expected {expected}"


# ---------------------------------------------------------------------------
# get_(image_)size — tensors
# ---------------------------------------------------------------------------


@cartesian_params(("cpu", "gpu"), TENSOR_CASES)
def test_get_image_size_tensor(device, t):
    _skip_if_gpu_unavailable(device)
    t = _move_tensor_to_device(t, device)
    expected = _tv_get_image_size(t)
    assert (
        get_image_size(t) == expected
    ), f"device={device} shape={t.shape}: got {get_image_size(t)}, expected {expected}"


@cartesian_params(("cpu", "gpu"), TENSOR_CASES)
def test_get_size_tensor(device, t):
    _skip_if_gpu_unavailable(device)
    t = _move_tensor_to_device(t, device)
    expected = fn_tv.get_size(t)
    assert (
        get_size(t) == expected
    ), f"device={device} shape={t.shape}: got {get_size(t)}, expected {expected}"


# ---------------------------------------------------------------------------
# get_dimensions — PIL
# ---------------------------------------------------------------------------


@params(*PIL_CASES)
def test_get_dimensions_pil(img):
    expected = fn_tv.get_dimensions(img)
    assert (
        get_dimensions(img) == expected
    ), f"mode={img.mode} size={img.size}: got {get_dimensions(img)}, expected {expected}"


# ---------------------------------------------------------------------------
# get_dimensions — tensors
# ---------------------------------------------------------------------------


@cartesian_params(("cpu", "gpu"), TENSOR_CASES)
def test_get_dimensions_tensor(device, t):
    _skip_if_gpu_unavailable(device)
    t = _move_tensor_to_device(t, device)
    expected = fn_tv.get_dimensions(t)
    assert (
        get_dimensions(t) == expected
    ), f"device={device} shape={t.shape}: got {get_dimensions(t)}, expected {expected}"


# ---------------------------------------------------------------------------
# Torchvision compatibility
# ---------------------------------------------------------------------------


@params(*PIL_CASES)
def test_image_metadata_pil_matches_torchvision(img):
    assert get_size(img) == fn_tv.get_size(img)
    assert get_image_size(img) == _tv_get_image_size(img)
    assert get_dimensions(img) == fn_tv.get_dimensions(img)


@cartesian_params(("cpu", "gpu"), TORCHVISION_COMPATIBILITY_CASES)
def test_image_metadata_tensor_inputs_match_torchvision(device, input_case):
    _skip_if_gpu_unavailable(device)
    input_kind, shape = input_case
    inpt = _move_tensor_to_device(_make_compatibility_input(input_kind, shape), device)

    assert get_size(inpt) == fn_tv.get_size(inpt)
    assert get_image_size(inpt) == _tv_get_image_size(inpt)
    assert get_dimensions(inpt) == fn_tv.get_dimensions(inpt)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_get_image_size_1d_tensor_raises():
    with assert_raises(TypeError):
        get_image_size(torch.zeros(10))
    with assert_raises(TypeError):
        get_size(torch.zeros(10))


def test_get_dimensions_1d_tensor_raises():
    with assert_raises(TypeError):
        get_dimensions(torch.zeros(10))


def test_get_image_size_unsupported_type_raises():
    with assert_raises(TypeError):
        get_image_size("not_an_image")
    with assert_raises(TypeError):
        get_size("not_an_image")


def test_get_dimensions_unsupported_type_raises():
    with assert_raises(TypeError):
        get_dimensions("not_an_image")
