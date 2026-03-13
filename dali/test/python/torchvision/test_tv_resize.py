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
from typing import Sequence, Literal, Union

import numpy as np
from nose2.tools import params, cartesian_params
from nose_utils import assert_raises
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as fn_tv

from nvidia.dali.experimental.torchvision import Resize, Compose
import nvidia.dali.experimental.torchvision.v2.functional as fn_dali


def read_filepath(path):
    return np.frombuffer(path.encode(), dtype=np.int8)


dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]
test_input_filenames = [read_filepath(fname) for fname in test_files]


def build_resize_transform(
    resize: int | Sequence[int],
    max_size: int = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
):
    t = transforms.Compose(
        [
            transforms.Resize(
                size=resize, max_size=max_size, interpolation=interpolation, antialias=antialias
            ),
        ]
    )
    td = Compose(
        [
            Resize(
                size=resize,
                max_size=max_size,
                interpolation=interpolation,
                antialias=antialias,
                device=device,
            ),
        ]
    )
    return t, td


def _internal_loop(
    input_data: Union[Image.Image, torch.Tensor],
    t: transforms.Resize,
    td: Resize,
    resize: int | Sequence[int],
    max_size: int = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
):
    out_fn = fn_tv.resize(
        input_data,
        size=resize,
        max_size=max_size,
        interpolation=interpolation,
        antialias=antialias,
    )
    out_dali_fn = fn_dali.resize(
        input_data,
        size=resize,
        max_size=max_size,
        interpolation=interpolation,
        antialias=antialias,
        device=device,
    )
    out_tv = t(input_data)
    out_dali_tv = td(input_data)

    if isinstance(input_data, Image.Image):
        out_tv = transforms.functional.pil_to_tensor(out_tv).unsqueeze(0).permute(0, 2, 3, 1)
        out_dali_tv = (
            transforms.functional.pil_to_tensor(out_dali_tv).unsqueeze(0).permute(0, 2, 3, 1)
        )
        out_fn = transforms.functional.pil_to_tensor(out_fn)
        out_dali_fn = transforms.functional.pil_to_tensor(out_dali_fn)

    assert (
        out_tv.shape[1:3] == out_dali_tv.shape[1:3]
    ), f"Should be:{out_tv.shape} is:{out_dali_tv.shape}"
    assert (
        out_fn.shape[1:3] == out_dali_fn.shape[1:3]
    ), f"Should be:{out_fn.shape} is:{out_dali_fn.shape}"

    if not isinstance(input_data, Image.Image):
        if out_tv.device != out_dali_tv.device:
            out_dali_tv = out_dali_tv.to(out_tv.device)
        if out_fn.device != out_dali_fn.device:
            out_dali_fn = out_dali_fn.to(out_fn.device)

        assert torch.allclose(out_tv, out_dali_tv, rtol=0, atol=1)
        assert torch.allclose(out_fn, out_dali_fn, rtol=0, atol=1)


def loop_images_test_no_build(
    t: transforms.Resize,
    td: Resize,
    resize: int | Sequence[int],
    max_size: int = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
):
    for fn in test_files:
        img = Image.open(fn)
        _internal_loop(img, t, td, resize, max_size, interpolation, antialias, device=device)


def build_tensors(max_size: int = 512, channels: int = 3, seed=12345):

    torch.manual_seed(seed)

    h = torch.randint(10, max_size, (1,)).item()
    w = torch.randint(10, max_size, (1,)).item()
    tensors = [
        torch.ones((channels, max_size, max_size)),
        torch.ones((1, channels, max_size, max_size)),
        torch.ones((10, channels, max_size, max_size)),
        torch.ones((channels, max_size // 2, max_size)),
        torch.ones((1, channels, max_size // 2, max_size)),
        torch.ones((10, channels, max_size // 2, max_size)),
        torch.ones((channels, max_size, max_size // 2)),
        torch.ones((1, channels, max_size, max_size // 2)),
        torch.ones((10, channels, max_size, max_size // 2)),
        torch.ones((channels, h, w)),
        torch.ones((1, channels, h, w)),
        torch.ones((10, channels, h, w)),
    ]

    return tensors


def loop_tensors_test(
    resize: int | Sequence[int],
    max_size: int = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
):
    t, td = build_resize_transform(resize, max_size, interpolation, antialias, device)
    tensors = build_tensors()

    for tn in tensors:
        _internal_loop(tn, t, td, resize, max_size, interpolation, antialias, device=device)


def loop_images_test(
    resize: int | Sequence[int],
    max_size: int = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
    device: Literal["cpu", "gpu"] = "cpu",
):
    t, td = build_resize_transform(resize, max_size, interpolation, antialias, device)
    loop_images_test_no_build(t, td, resize, max_size, interpolation, antialias, device)


@cartesian_params((4, 5, 8), ("cpu", "gpu"))
def test_resize_sizes_tensors_mini(resize, device):
    loop_tensors_test(resize=resize, max_size=10, device=device)


@cartesian_params((512, 1125, 2048, ([512, 512]), ([2048, 2048])), ("cpu", "gpu"))
def test_resize_sizes_images(resize, device):
    # Resize with single int (preserve aspect ratio)
    loop_images_test(resize=resize, device=device)


@cartesian_params((512, 1125, 2048, ([512, 512]), ([2048, 2048])), ("cpu", "gpu"))
def test_resize_sizes_tensors(resize, device):
    # Resize with single int (preserve aspect ratio)
    loop_tensors_test(resize=resize, device=device)


@params((480, 512), (100, 124), (None, 512), (1024, 512), ([256, 256], 512), (None, None))
def test_resize_max_sizes(resize, max_size):
    # Resize with single int (preserve aspect ratio)
    if resize is not None and max_size is not None and np.min(np.array(resize, int)) > max_size:

        """
        with assert_raises(ValueError):
            _ = transforms.Resize(resize, max_size)
        This exception is called later - when executing the operation
        """

        with assert_raises(ValueError):
            _ = Compose(
                [
                    Resize(resize, max_size=max_size),
                ]
            )
        return
    if resize is None and max_size is None:
        with assert_raises(ValueError):
            _ = transforms.Resize(resize, max_size)

        with assert_raises(ValueError):
            _ = Compose(
                [
                    Resize(resize, max_size=max_size),
                ]
            )
        return

    if isinstance(resize, Sequence) and len(resize) != 1 and max_size is not None:
        """
        with assert_raises(ValueError):
            _ = transforms.Resize(resize, max_size)
        This exception is called later - when executing the operation
        """

        with assert_raises(ValueError):
            _ = Compose(
                [
                    Resize(resize, max_size=max_size),
                ]
            )
        return

    loop_images_test(resize=resize, max_size=max_size)


@cartesian_params(
    (
        640,
        768,
        1024,
        ([512, 512]),
        ([256, 256]),
    ),
    (
        transforms.InterpolationMode.NEAREST,
        transforms.InterpolationMode.NEAREST_EXACT,
        transforms.InterpolationMode.BILINEAR,
        transforms.InterpolationMode.BICUBIC,
    ),
    ("cpu", "gpu"),
)
def test_resize_interpolation(resize, interpolation, device):
    if interpolation == transforms.InterpolationMode.NEAREST_EXACT:
        with assert_raises(NotImplementedError):
            loop_images_test(resize=resize, interpolation=interpolation, device=device)
    else:
        loop_images_test(resize=resize, interpolation=interpolation, device=device)


@cartesian_params((512, 768, 2048, ([512, 512]), ([2048, 2048])), (True, False), ("cpu", "gpu"))
def test_resize_antialiasing(resize, antialiasing, device):
    loop_images_test(resize=resize, antialias=antialiasing, device=device)


@cartesian_params((8192, 8193, 10243), ("cpu", "gpu"))
def test_large_sizes_images(resize, device):
    loop_images_test(resize=resize, device=device)


"""
These tests are too heavy they would cause timeouts

@cartesian_params((8192, 8193, 10243), ("cpu", "gpu"))
def test_large_sizes_tensors(resize, device):
    # Resize with single int (preserve aspect ratio)
    loop_tensors_test(resize=resize, device=device)
"""
