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
import torchvision.transforms as tv

from nvidia.dali.experimental.torchvision import Compose, RandomHorizontalFlip, RandomVerticalFlip


def make_test_tensor(shape=(1, 10, 10, 3)):
    total = 1
    for s in shape:
        total *= s
    return torch.arange(total).reshape(shape)


def test_horizontal_random_flip_probability():
    img = make_test_tensor()
    transform = Compose([RandomHorizontalFlip(p=1.0)])  # always flip
    out = transform(img)
    tvout = tv.RandomHorizontalFlip(p=1.0)(img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    assert torch.equal(out, tvout)

    transform = Compose([RandomHorizontalFlip(p=0.0)])  # never flip
    out = transform(img)
    assert torch.equal(out, img)


def test_vertical_random_flip_probability():
    img = make_test_tensor()
    transform = Compose([RandomVerticalFlip(p=1.0)])  # always flip
    out = transform(img)
    tvout = tv.RandomVerticalFlip(p=1.0)(img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    assert torch.equal(out, tvout)

    transform = Compose([RandomVerticalFlip(p=0.0)])  # never flip
    out = transform(img)
    assert torch.equal(out, img)


def test_flip_preserves_shape():
    img = make_test_tensor((1, 15, 20, 3))
    hflip = Compose([RandomHorizontalFlip(p=1.0)])(img)
    vflip = Compose([RandomVerticalFlip(p=1.0)])(img)
    assert hflip.shape == img.shape
    assert vflip.shape == img.shape
