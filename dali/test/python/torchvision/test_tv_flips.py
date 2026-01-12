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
import torchvision.transforms.v2 as tv

from nose2.tools import params
from nvidia.dali.experimental.torchvision import Compose, RandomHorizontalFlip, RandomVerticalFlip
from nvidia.dali.experimental.torchvision.v2.functional import horizontal_flip, vertical_flip


def make_test_tensor(shape=(1, 10, 10, 3)):
    total = 1
    for s in shape:
        total *= s
    return torch.arange(total).reshape(shape)


@params("gpu", "cpu")
def test_horizontal_random_flip_probability(device):
    img = make_test_tensor()
    transform = Compose([RandomHorizontalFlip(p=1.0, device=device)])  # always flip
    out = transform(img).cpu()
    out_tv = tv.RandomHorizontalFlip(p=1.0)(img)
    out_fn = horizontal_flip(img).cpu()
    assert torch.equal(out, out_tv)
    assert torch.equal(out_fn, out_tv)

    transform = Compose([RandomHorizontalFlip(p=0.0, device=device)])  # never flip
    out = transform(img).cpu()
    assert torch.equal(out, img)


@params("gpu", "cpu")
def test_vertical_random_flip_probability(device):
    img = make_test_tensor()
    transform = Compose([RandomVerticalFlip(p=1.0, device=device)])  # always flip
    out = transform(img).cpu()
    out_tv = tv.RandomVerticalFlip(p=1.0)(img)
    out_fn = vertical_flip(img).cpu()
    assert torch.equal(out, out_tv)

    transform = Compose([RandomVerticalFlip(p=0.0, device=device)])  # never flip
    out = transform(img).cpu()
    assert torch.equal(out, img)


def test_flip_preserves_shape():
    img = make_test_tensor((1, 15, 20, 3))
    hflip_pipeline = Compose([RandomHorizontalFlip(p=1.0)])
    hflip_fn = horizontal_flip(img).cpu()
    hflip = hflip_pipeline(img)
    vflip_pipeline = Compose([RandomVerticalFlip(p=1.0)])
    vflip_fn = vertical_flip(img).cpu()
    vflip = vflip_pipeline(img)
    assert hflip.shape == img.shape
    assert vflip.shape == img.shape
    assert hflip_fn.shape == img.shape
    assert vflip_fn.shape == img.shape
