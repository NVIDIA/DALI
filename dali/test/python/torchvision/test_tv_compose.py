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

from nvidia.dali.experimental.torchvision import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)

from nose2.tools import params
from nose_utils import assert_raises
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as tv
import torch


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
    test_tensor = make_test_tensor(shape=(5, 3, 5, 5))
    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0)], batch_size=test_tensor.shape[0])
    dali_out = dali_pipeline(test_tensor)
    tv_out = tv.RandomHorizontalFlip(p=1.0)(test_tensor)

    assert isinstance(dali_out, torch.Tensor)
    assert torch.equal(dali_out, tv_out)


def test_compose_multi_tensor():
    test_tensor = make_test_tensor(shape=(5, 3, 5, 5))
    dali_pipeline = Compose(
        [Resize(size=(15, 15)), RandomHorizontalFlip(p=1.0), RandomVerticalFlip(p=1.0)],
        batch_size=test_tensor.shape[0],
    )
    dali_out = dali_pipeline(test_tensor)
    tv_pipeline = tv.Compose(
        [tv.Resize(size=(15, 15)), tv.RandomHorizontalFlip(p=1.0), tv.RandomVerticalFlip(p=1.0)]
    )
    tv_out = tv_pipeline(test_tensor)

    assert isinstance(dali_out, torch.Tensor)
    # All close, because there are pixel differences due to resize
    assert torch.allclose(dali_out, tv_out, rtol=0, atol=1), f"Should be {tv_out} is {dali_out}"


def test_compose_invalid_batch_tensor():
    test_tensor = make_test_tensor(shape=(5, 1, 5, 5))
    with assert_raises(RuntimeError):
        dali_pipeline = Compose([RandomHorizontalFlip(p=1.0)], batch_size=1)
        _ = dali_pipeline(test_tensor)


def test_compose_images():
    dali_transform = Compose([RandomHorizontalFlip(p=1.0)])
    tv_transform = tv.Compose([tv.RandomHorizontalFlip(p=1.0)])

    for fn in test_files:
        img = Image.open(fn)
        out_dali_img = dali_transform(img)

        assert isinstance(out_dali_img, Image.Image)

        tensor_dali_tv = tv.functional.pil_to_tensor(out_dali_img)
        tensor_tv = tv.functional.pil_to_tensor(tv_transform(img))

        assert tensor_dali_tv.shape == tensor_tv.shape
        assert torch.equal(tensor_dali_tv, tensor_tv)


def test_compose_images_multi():
    dali_transform = Compose([RandomVerticalFlip(p=1.0), RandomHorizontalFlip(p=1.0)])
    tv_transform = tv.Compose([tv.RandomVerticalFlip(p=1.0), tv.RandomHorizontalFlip(p=1.0)])

    for fn in test_files:
        img = Image.open(fn)
        out_dali_img = dali_transform(img)

        assert isinstance(out_dali_img, Image.Image)

        tensor_dali_tv = tv.functional.pil_to_tensor(out_dali_img)
        tensor_tv = tv.functional.pil_to_tensor(tv_transform(img))

        assert tensor_dali_tv.shape == tensor_tv.shape
        assert torch.equal(tensor_dali_tv, tensor_tv)


def test_compose_invalid_type_images():
    dali_transform = Compose([RandomHorizontalFlip(p=1.0)])

    for fn in test_files:
        img = Image.open(fn)
        with assert_raises(TypeError):
            _ = dali_transform([img, img, img])


def _make_pil_image(mode, h=50, w=60, seed=42):
    rng = np.random.default_rng(seed)
    if mode == "L":
        data = rng.integers(0, 256, (h, w), dtype=np.uint8)
    elif mode == "RGB":
        data = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif mode == "RGBA":
        data = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return Image.fromarray(data, mode=mode)


@params("RGB", "L", "RGBA")
def test_compose_pil_mode_flip(mode):
    """Horizontal flip must produce a pixel-exact match with torchvision for all PIL modes."""
    img = _make_pil_image(mode)
    dali_transform = Compose([RandomHorizontalFlip(p=1.0)])
    tv_transform = tv.Compose([tv.RandomHorizontalFlip(p=1.0)])

    out_dali = dali_transform(img)
    out_tv = tv_transform(img)

    assert isinstance(out_dali, Image.Image)
    assert out_dali.mode == mode, f"Mode changed: expected {mode}, got {out_dali.mode}"
    assert torch.equal(
        tv.functional.pil_to_tensor(out_dali),
        tv.functional.pil_to_tensor(out_tv),
    ), f"Pixel mismatch for mode {mode}"


@params("RGB", "L", "RGBA")
def test_compose_pil_mode_resize(mode):
    """Resize must produce the correct output shape and preserve PIL mode."""
    img = _make_pil_image(mode)
    target = (30, 40)
    dali_transform = Compose([Resize(size=target)])
    tv_transform = tv.Compose([tv.Resize(size=target)])

    out_dali = dali_transform(img)
    out_tv = tv_transform(img)

    assert isinstance(out_dali, Image.Image)
    assert out_dali.mode == mode, f"Mode changed: expected {mode}, got {out_dali.mode}"
    # PIL size is (w, h); compare as (h, w) to match the target convention
    assert (
        out_dali.size == out_tv.size
    ), f"Size mismatch for mode {mode}: {out_dali.size} != {out_tv.size}"


@params("RGB", "L", "RGBA")
def test_compose_pil_mode_multi_op(mode):
    """Chained flip+resize must preserve mode and match torchvision output shape."""
    img = _make_pil_image(mode)
    dali_transform = Compose([Resize(size=(30, 40)), RandomHorizontalFlip(p=1.0)])
    tv_transform = tv.Compose([tv.Resize(size=(30, 40)), tv.RandomHorizontalFlip(p=1.0)])

    out_dali = dali_transform(img)
    out_tv = tv_transform(img)

    assert isinstance(out_dali, Image.Image)
    assert out_dali.mode == mode, f"Mode changed: expected {mode}, got {out_dali.mode}"
    assert (
        out_dali.size == out_tv.size
    ), f"Size mismatch for mode {mode}: {out_dali.size} != {out_tv.size}"


@params("RGB", "L", "RGBA")
def test_compose_pil_invalid_input_type_raises(mode):
    """Passing a list instead of a PIL Image must raise TypeError regardless of mode."""
    img = _make_pil_image(mode)
    dali_transform = Compose([RandomHorizontalFlip(p=1.0)])
    with assert_raises(TypeError):
        _ = dali_transform([img, img])
