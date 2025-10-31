# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torchvision.transforms as tv

from nose2.tools import cartesian_params


def make_test_tensor(shape=(1, 10, 10, 1)):
    total = 1
    for s in shape:
        total *= s
    return torch.arange(total).reshape(shape).to(dtype=torch.uint8)


def test_pad_core(mode: str, device: str, padding: list, fill=0.0):
    img = make_test_tensor(shape=(1, 5, 5, 1))
    if mode == "constant":
        transform = Compose([Pad(padding=padding, fill=fill, padding_mode=mode, device=device)])
        transform_tv = tv.Pad(padding, fill=fill, padding_mode=mode)
    else:
        transform = Compose([Pad(padding=padding, padding_mode=mode, device=device)])
        transform_tv = tv.Pad(padding, padding_mode=mode)
    out = transform(img)
    out_tv = transform_tv(img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    if device == "gpu":
        out = out.cpu()
    assert torch.equal(out, out_tv), f"DALI shape: {out.shape}, TV shape: {out_tv.shape}"


@cartesian_params(
    # TODO: edge will fail, because we do not have a mean to replicate an arrray x times
    ("constant", "reflect", "symmetric", "edge"),
    # TODO: constant on GPU will fail, fn.full is not implemented there
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
    # TODO: edge will fail, because we do not have a mean to replicate an arrray x times
    ("constant", "reflect", "symmetric", "edge"),
    # TODO: constant on GPU will fail, fn.full is not implemented there
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
    # TODO: constant on GPU will fail, fn.full is not implemented there
    ("cpu",),
    ([2, 2, 2, 2], [5, 5, 5, 5]),
    # TODO: no support for fill values > 1D in DALI !!!
    (1.0,),
)
def test_constant_pad(device: str, padding: list, fill):
    test_pad_core("constant", device, padding, fill)
