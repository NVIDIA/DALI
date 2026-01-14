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

import torch

from nvidia.dali.experimental.torchvision import Compose, Pad
from nvidia.dali.experimental.torchvision.v2.functional import pad

import torchvision.transforms.v2 as tv

from nose2.tools import cartesian_params


def make_test_tensor(shape=(1, 1, 10, 10)):
    total = 1
    for s in shape:
        total *= s
    return torch.arange(total).reshape(shape).to(dtype=torch.uint8)


def test_pad_core(mode: str, device: str, padding: list, fill=0.0):
    img = make_test_tensor(shape=(1, 1, 5, 5))
    if mode == "constant":
        transform = Compose([Pad(padding=padding, fill=fill, padding_mode=mode, device=device)])
        transform_tv = tv.Pad(padding, fill=fill, padding_mode=mode)
    else:
        transform = Compose([Pad(padding=padding, padding_mode=mode, device=device)])
        transform_tv = tv.Pad(padding, padding_mode=mode)
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


@cartesian_params(
    ("constant", "reflect", "symmetric", "edge"),
    ("cpu", "gpu"),
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
def test_pad_symetric_single(mode, device, padding):
    test_pad_core(mode, device, padding)


@cartesian_params(
    ("constant", "reflect", "symmetric", "edge"),
    ("cpu", "gpu"),
    (
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [0, 1, 3, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    ),
)
def test_pad_multi(mode: str, device: str, padding: list):
    test_pad_core(mode, device, padding)


@cartesian_params(
    ("cpu", "gpu"),
    ([2, 2, 2, 2], [5, 5, 5, 5]),
    (2.0,),
)
def test_constant_pad(device: str, padding: list, fill):
    test_pad_core("constant", device, padding, fill)
