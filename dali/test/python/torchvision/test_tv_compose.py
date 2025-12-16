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

from nvidia.dali.experimental.torchvision import Compose, RandomHorizontalFlip

from nose_utils import assert_raises
import numpy as np
from PIL import Image
import torchvision.transforms as tv
import torch
import torchvision.transforms.v2 as transforms


def read_filepath(path):
    return np.frombuffer(path.encode(), dtype=np.int8)


def make_test_tensor(shape=(5, 10, 10, 1)):
    total = 1
    for s in shape:
        total *= s
    return torch.arange(total).reshape(shape).to(dtype=torch.uint8)


dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]
test_input_filenames = [read_filepath(fname) for fname in test_files]


def test_compose_tensor():
    test_tensor = make_test_tensor(shape=(5, 5, 5, 3))
    dali_out = Compose([RandomHorizontalFlip(p=1.0)], batch_size=test_tensor.shape[0])(test_tensor)
    tv_out = tv.RandomHorizontalFlip(p=1.0)(test_tensor)

    assert isinstance(dali_out, torch.Tensor)
    assert torch.equal(dali_out, tv_out)


def test_compose_invalid_batch_tensor():
    test_tensor = make_test_tensor(shape=(5, 5, 5, 1))
    with assert_raises(RuntimeError):
        _ = Compose([RandomHorizontalFlip(p=1.0)], batch_size=1)(test_tensor)


def test_compose_images():
    dali_transform = Compose([RandomHorizontalFlip(p=1.0)])
    tv_transform = tv.RandomHorizontalFlip(p=1.0)

    for fn in test_files:
        img = Image.open(fn)
        out_dali_img = dali_transform(img)

        assert isinstance(out_dali_img, Image.Image)

        tensor_dali_tv = transforms.functional.pil_to_tensor(out_dali_img)
        tensor_tv = transforms.functional.pil_to_tensor(tv_transform(img))

        assert tensor_dali_tv.shape == tensor_tv.shape
        assert torch.equal(tensor_dali_tv, tensor_tv)


def test_compose_invalid_type_images():
    dali_transform = Compose([RandomHorizontalFlip(p=1.0)])

    for fn in test_files:
        img = Image.open(fn)
        with assert_raises(TypeError):
            out_dali_img = dali_transform([img, img, img])
            assert isinstance(out_dali_img, Image.Image)
