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
from typing import List, Literal

import torch
from nose2.tools import cartesian_params, params
from nose_utils import assert_raises
from PIL import Image
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as fn_tv

import nvidia.dali.experimental.torchvision.v2.functional as fn_dali

dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]


def _run_one(
    inpt,
    top: int,
    left: int,
    height: int,
    width: int,
    size: int | List[int],
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = True,
    device: Literal["cpu", "gpu"] = "cpu",
):
    """Run torchvision and DALI on the same input and assert shape / pixel agreement."""
    out_tv = fn_tv.resized_crop(
        inpt, top, left, height, width, size, interpolation=interpolation, antialias=antialias
    )
    out_dali = fn_dali.resized_crop(
        inpt,
        top,
        left,
        height,
        width,
        size,
        interpolation=interpolation,
        antialias=antialias,
        device=device,
    )

    if isinstance(inpt, Image.Image):
        # Shape comparison only — PIL round-trips introduce additional rounding
        out_tv_t = transforms.functional.pil_to_tensor(out_tv)
        out_dali_t = transforms.functional.pil_to_tensor(out_dali)
        assert (
            out_tv_t.shape == out_dali_t.shape
        ), f"Shape mismatch: tv={out_tv_t.shape} dali={out_dali_t.shape}"
    else:
        if out_tv.device != out_dali.device:
            out_dali = out_dali.to(out_tv.device)
        assert torch.allclose(
            out_tv, out_dali, rtol=0, atol=1
        ), f"Pixel mismatch: max diff={(out_tv.int() - out_dali.int()).abs().max().item()}"


def loop_images_test(
    top: int,
    left: int,
    height: int,
    width: int,
    size: int | List[int],
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = True,
    device: Literal["cpu", "gpu"] = "cpu",
):
    for filepath in test_files:
        img = Image.open(filepath)
        _run_one(img, top, left, height, width, size, interpolation, antialias, device)


def build_tensors(h: int = 256, w: int = 320, channels: int = 3):
    """Return a variety of CHW and NCHW tensors with the given spatial dimensions."""
    return [
        torch.randint(0, 256, (channels, h, w), dtype=torch.uint8),
        torch.randint(0, 256, (1, channels, h, w), dtype=torch.uint8),
        torch.randint(0, 256, (4, channels, h, w), dtype=torch.uint8),
    ]


def loop_tensors_test(
    top: int,
    left: int,
    height: int,
    width: int,
    size: int | List[int],
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
    tensor_h: int = 256,
    tensor_w: int = 320,
    channels: int = 3,
):
    # antialias=False by default: antialiased downscaling differs between DALI and
    # torchvision filter implementations and is not expected to be pixel-exact.
    for tn in build_tensors(h=tensor_h, w=tensor_w, channels=channels):
        _run_one(tn, top, left, height, width, size, interpolation, antialias, device)


# ---------------------------------------------------------------------------
# Output size variants
# ---------------------------------------------------------------------------


@cartesian_params((112, 224, [224, 224], [112, 336]), ("cpu", "gpu"))
def test_resized_crop_sizes_images(size, device):
    # Crop a 400×600 window from the top-left area, then resize
    loop_images_test(top=10, left=10, height=400, width=600, size=size, device=device)


@cartesian_params((64, 128, [128, 128], [64, 96]), ("cpu", "gpu"))
def test_resized_crop_sizes_tensors(size, device):
    # Tensors are 256×320; crop a 128×160 window from (32, 40)
    loop_tensors_test(top=32, left=40, height=128, width=160, size=size, device=device)


# ---------------------------------------------------------------------------
# Crop position and window variants
# ---------------------------------------------------------------------------


@cartesian_params(
    ((0, 0, 200, 200), (50, 100, 300, 400), (10, 10, 400, 600)),
    ("cpu", "gpu"),
)
def test_resized_crop_crop_positions_images(crop, device):
    top, left, height, width = crop
    loop_images_test(top=top, left=left, height=height, width=width, size=224, device=device)


@cartesian_params(
    ((0, 0, 100, 100), (10, 20, 128, 160), (50, 50, 64, 64)),
    ("cpu", "gpu"),
)
def test_resized_crop_crop_positions_tensors(crop, device):
    top, left, height, width = crop
    loop_tensors_test(top=top, left=left, height=height, width=width, size=128, device=device)


@cartesian_params(
    ((-1, -2, 6, 8, [6, 8]), (6, 8, 5, 6, [5, 6]), (0, 0, 12, 14, [12, 14])),
    ("cpu", "gpu"),
)
def test_resized_crop_crop_padding_tensors(crop_case, device):
    top, left, height, width, size = crop_case
    loop_tensors_test(
        top=top,
        left=left,
        height=height,
        width=width,
        size=size,
        interpolation=transforms.InterpolationMode.NEAREST,
        antialias=False,
        device=device,
        tensor_h=8,
        tensor_w=10,
    )


# ---------------------------------------------------------------------------
# Interpolation modes
# ---------------------------------------------------------------------------


@cartesian_params(
    (
        transforms.InterpolationMode.NEAREST,
        transforms.InterpolationMode.NEAREST_EXACT,
        transforms.InterpolationMode.BILINEAR,
        transforms.InterpolationMode.BICUBIC,
    ),
    ("cpu", "gpu"),
)
def test_resized_crop_interpolation(interpolation, device):
    if interpolation == transforms.InterpolationMode.NEAREST_EXACT:
        with assert_raises(NotImplementedError):
            loop_images_test(
                top=10,
                left=10,
                height=400,
                width=600,
                size=224,
                interpolation=interpolation,
                device=device,
            )
    else:
        loop_images_test(
            top=10,
            left=10,
            height=400,
            width=600,
            size=224,
            interpolation=interpolation,
            device=device,
        )


# ---------------------------------------------------------------------------
# Antialias
# ---------------------------------------------------------------------------


@cartesian_params((True, False), ("cpu", "gpu"))
def test_resized_crop_antialias_images(antialias, device):
    loop_images_test(
        top=10, left=10, height=400, width=600, size=224, antialias=antialias, device=device
    )


@cartesian_params((True, False), ("cpu", "gpu"))
def test_resized_crop_antialias_tensors(antialias, device):
    # Antialiased downscaling differs between DALI and torchvision filter implementations,
    # so only assert shape agreement (not pixel values) — same approach as test_tv_resize.py.
    out_tv = fn_tv.resized_crop(build_tensors()[0], 32, 40, 128, 160, 128, antialias=antialias)
    out_dali = fn_dali.resized_crop(
        build_tensors()[0], 32, 40, 128, 160, 128, antialias=antialias, device=device
    )
    out_dali = out_dali.cpu()
    assert (
        out_tv.shape == out_dali.shape
    ), f"Shape mismatch: tv={out_tv.shape} dali={out_dali.shape}"


# ---------------------------------------------------------------------------
# Validation and exports
# ---------------------------------------------------------------------------


def test_resized_crop_exported_from_crop_module():
    from nvidia.dali.experimental.torchvision.v2.functional.crop import resized_crop

    assert fn_dali.resized_crop is resized_crop


@params(
    (0.5, 0),
    ("0", 0),
    (0, 0.5),
    (0, "0"),
)
def test_resized_crop_invalid_coordinates(top, left):
    with assert_raises(TypeError, glob="*must be an integer*"):
        _ = fn_dali.resized_crop(
            build_tensors(h=8, w=10)[0], top=top, left=left, height=1, width=1, size=1
        )


@params(
    (0, 1),
    (1, 0),
    (-1, 1),
    (1, -1),
    (1.0, 1),
    (1, 1.0),
)
def test_resized_crop_invalid_crop_size(height, width):
    with assert_raises((TypeError, ValueError), glob="*must be*"):
        _ = fn_dali.resized_crop(
            build_tensors(h=8, w=10)[0], top=0, left=0, height=height, width=width, size=1
        )
