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

import numpy as np
from nose2.tools import params
from nose_utils import assert_raises
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms

from nvidia.dali.experimental.torchvision import Compose, Grayscale


def verify_non_one_off(t1: torch.Tensor, t2: torch.Tensor):

    if t1.dtype == torch.uint8:
        t1 = t1.to(torch.int16)
        t2 = t2.to(torch.int16)

    diff = (t1 - t2).abs()
    more_than_one_mask = diff > 1

    return more_than_one_mask.sum().item() == 0


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


def loop_images_test(t, td):
    for fn in test_files:
        img = Image.open(fn)

        # t(img).save(f"{fn}_gs_tv.png")
        # td(img).save(f"{fn}_dali_gs_tv.png")
        out_tv = transforms.functional.pil_to_tensor(t(img))
        out_dali_tv = transforms.functional.pil_to_tensor(td(img))

        assert verify_non_one_off(out_tv, out_dali_tv), f"Images differ {fn}"


@params(
    1,
    3,
)
def test_grayscale(output_channels):
    td = Compose([Grayscale(num_output_channels=output_channels)])
    t = transforms.Compose([transforms.Grayscale(num_output_channels=output_channels)])

    loop_images_test(t, td)


@params(2, 4)
def test_invalid_channel_count_grayscale(output_channels):
    img = Image.open(test_files[0])

    with assert_raises(ValueError):
        # Torchvision raises the exception when executing kernel,
        # we need to raise it from the constructor
        td = Compose([Grayscale(num_output_channels=output_channels)])
        out_dali_tv = transforms.functional.pil_to_tensor(td(img))  # noqa
