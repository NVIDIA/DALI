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

import os
from typing import Sequence, Union

import numpy as np
from nose2.tools import params, cartesian_params
from nose_utils import assert_raises
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as fn_tv

from nvidia.dali.experimental.torchvision import Resize, Compose, ToTensor
import nvidia.dali.experimental.torchvision.v2.functional as fn_dali


def read_file(path):
    return np.fromfile(path, dtype=np.uint8)


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
    resize: Union[int, Sequence[int]],
    max_size: int = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
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
                size=resize, max_size=max_size, interpolation=interpolation, antialias=antialias
            ),
        ]
    )
    return t, td


def loop_images_test(
    resize: Union[int, Sequence[int]],
    max_size: int = None,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    antialias: bool = False,
):
    t, td = build_resize_transform(resize, max_size, interpolation, antialias)

    for fn in test_files:
        img = Image.open(fn)
        out_fn = fn_tv.resize(
            img, size=resize, max_size=max_size, interpolation=interpolation, antialias=antialias
        )
        out_dali_fn = transforms.functional.pil_to_tensor(
            fn_dali.resize(
                img,
                size=resize,
                max_size=max_size,
                interpolation=interpolation,
                antialias=antialias,
            )
        )

        out_tv = transforms.functional.pil_to_tensor(t(img)).unsqueeze(0).permute(0, 2, 3, 1)
        out_dali_tv = transforms.functional.pil_to_tensor(td(img)).unsqueeze(0).permute(0, 2, 3, 1)
        tv_shape_lower = torch.Size([out_tv.shape[1] - 1, out_tv.shape[2] - 1])
        tv_shape_upper = torch.Size([out_tv.shape[1] + 1, out_tv.shape[2] + 1])
        assert (
            tv_shape_lower[0] <= out_dali_tv.shape[1] <= tv_shape_upper[0]
        ), f"Should be:{out_tv.shape} is:{out_dali_tv.shape}"
        assert (
            tv_shape_lower[1] <= out_dali_tv.shape[2] <= tv_shape_upper[1]
        ), f"Should be:{out_tv.shape} is:{out_dali_tv.shape}"

        assert (
            tv_shape_lower[0] <= out_dali_fn.shape[1] <= tv_shape_upper[0]
        ), f"Should be:{out_tv.shape} is:{out_dali_fn.shape}"
        assert (
            tv_shape_lower[1] <= out_dali_fn.shape[2] <= tv_shape_upper[1]
        ), f"Should be:{out_tv.shape} is:{out_dali_fn.shape}"
        # assert torch.equal(out_tv, out_dali_tv)


@cartesian_params(
    (512, 2048, ([512, 512]), ([2048, 2048])),
    ("cpu", "gpu"),
)
def test_resize_and_tensor(resize, device):
    # Resize with single int (preserve aspect ratio)
    td = Compose(
        [
            Resize(size=resize, device=device),
            ToTensor(),
        ]
    )

    img = Image.open(test_files[0])
    out = td(img)

    assert isinstance(out, torch.Tensor), f"Should be torch.Tensor type is {type(out)}"
    assert torch.all(out <= 1).item(), "Tensor elements should be <0;1>"


@params(512, 2048, ([512, 512]), ([2048, 2048]))
def test_resize_sizes(resize):
    # Resize with single int (preserve aspect ratio)
    loop_images_test(resize=resize)


@params((480, 512), (100, 124), (None, 512), (1024, 512), ([256, 256], 512))
def test_resize_max_sizes(resize, max_size):
    # Resize with single int (preserve aspect ratio)
    if resize is not None and (
        (isinstance(resize, int) and max_size is not None and max_size < resize)
        or (not isinstance(resize, int))
    ):
        with assert_raises(ValueError):
            _ = Compose(
                [
                    Resize(resize, max_size=max_size),
                ]
            )
        return

    loop_images_test(resize=resize, max_size=max_size)


@params(
    ([512, 512], transforms.InterpolationMode.NEAREST),
    (1024, transforms.InterpolationMode.NEAREST_EXACT),
    ([256, 256], transforms.InterpolationMode.BILINEAR),
    (640, transforms.InterpolationMode.BICUBIC),
)
def test_resize_interploation(resize, interpolation):
    loop_images_test(resize=resize, interpolation=interpolation)


@params((512, True), (2048, True), ([512, 512], True), ([2048, 2048], True))
def test_resize_antialiasing(resize, antialiasing):
    loop_images_test(resize=resize, antialias=antialiasing)
