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

<<<<<<< HEAD
=======
import numpy as np
>>>>>>> 2bc5cc11 (Torchvision ColorJitter and Grayscale implementations)
from nose2.tools import params, cartesian_params
from nose_utils import assert_raises
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms

from nvidia.dali.experimental.torchvision import Compose, Grayscale, ColorJitter
from nvidia.dali.experimental.torchvision.v2.functional import to_grayscale, rgb_to_grayscale


def verify_non_one_off(t1: torch.Tensor, t2: torch.Tensor):

    if t1.dtype == torch.uint8:
        t1 = t1.to(torch.int16)
        t2 = t2.to(torch.int16)

    diff = (t1 - t2).abs()
    more_than_one_mask = diff > 1

    return more_than_one_mask.sum().item() == 0


<<<<<<< HEAD
=======
def read_file(path):
    return np.fromfile(path, dtype=np.uint8)


def read_filepath(path):
    return np.frombuffer(path.encode(), dtype=np.int8)


>>>>>>> 2bc5cc11 (Torchvision ColorJitter and Grayscale implementations)
dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]
<<<<<<< HEAD


def loop_images_test(t, td, num_output_channels=1):
=======
test_input_filenames = [read_filepath(fname) for fname in test_files]


def loop_images_test(t, td):
>>>>>>> 2bc5cc11 (Torchvision ColorJitter and Grayscale implementations)
    for fn in test_files:
        img = Image.open(fn)

        out_tv = transforms.functional.pil_to_tensor(t(img))
        out_dali_tv = transforms.functional.pil_to_tensor(td(img))
<<<<<<< HEAD
        out_dali_fn = transforms.functional.pil_to_tensor(
            to_grayscale(img, num_output_channels=num_output_channels)
        )
        out_dali_fn1 = transforms.functional.pil_to_tensor(
            rgb_to_grayscale(img, num_output_channels=num_output_channels)
        )
=======
        out_dali_fn = transforms.functional.pil_to_tensor(to_grayscale(img))
        out_dali_fn1 = transforms.functional.pil_to_tensor(rgb_to_grayscale(img))
>>>>>>> 2bc5cc11 (Torchvision ColorJitter and Grayscale implementations)

        assert verify_non_one_off(out_tv, out_dali_tv), f"Images differ {fn}"
        assert verify_non_one_off(out_tv, out_dali_fn), f"Images differ {fn}"
        assert verify_non_one_off(out_tv, out_dali_fn1), f"Images differ {fn}"


@cartesian_params((1, 3), ("cpu", "gpu"))
def test_grayscale(output_channels, device):
    td = Compose([Grayscale(num_output_channels=output_channels, device=device)])
    t = transforms.Compose([transforms.Grayscale(num_output_channels=output_channels)])

<<<<<<< HEAD
    loop_images_test(t, td, num_output_channels=output_channels)
=======
    loop_images_test(t, td)
>>>>>>> 2bc5cc11 (Torchvision ColorJitter and Grayscale implementations)


@params(2, 4)
def test_invalid_channel_count_grayscale(output_channels):
    img = Image.open(test_files[0])

    with assert_raises(ValueError):
        # Torchvision raises the exception when executing kernel,
        # we need to raise it from the constructor
        td = Compose([Grayscale(num_output_channels=output_channels)])
        _ = transforms.functional.pil_to_tensor(td(img))


def make_tensor_inputs():
    # Single CHW tensor and batched NCHW tensor
    img = torch.rand(3, 64, 64)
    batch = torch.rand(4, 3, 64, 64)
    return (img, batch)


@cartesian_params(
    (
        # brightness, contrast, saturation, hue
        (0.0, 0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0, 0.0),
        (0.0, 0.3, 0.0, 0.0),
        (0.0, 0.0, 0.4, 0.0),
        (0.0, 0.0, 0.0, 0.1),
        (0.5, 0.5, 0.5, 0.1),
        ((0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (-0.1, 0.1)),
    ),
    ("cpu", "gpu"),
)
def test_colorjitter_images(cj_params, device):
    brightness, contrast, saturation, hue = cj_params

    cj = Compose(
        [
            ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                device=device,
            )
        ]
    )

    for fn in test_files:
        img = Image.open(fn)
        _ = cj(img)


"""
<<<<<<< HEAD
TODO (https://github.com/NVIDIA/DALI/issues/DALI-4656):
DALI ColorJitter does not currently work on CHW layout
=======
TODO: DALI ColorJitter does not currently work on CHW layout
>>>>>>> 2bc5cc11 (Torchvision ColorJitter and Grayscale implementations)
@params(
    # brightness, contrast, saturation, hue
    (0.0, 0.0, 0.0, 0.0),
    (0.2, 0.0, 0.0, 0.0),
    (0.0, 0.3, 0.0, 0.0),
    (0.0, 0.0, 0.4, 0.0),
    (0.0, 0.0, 0.0, 0.1),
    (0.5, 0.5, 0.5, 0.1),
    ((0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (-0.1, 0.1)),
)
def test_colorjitter_tensor(brightness, contrast, saturation, hue):
    img, batch = make_tensor_inputs()

    cj = Compose( [ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,)]
    )

    cj_batched = Compose( [ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,)],
        batch_size = 4
    )

    _ = cj(img)
    _ = cj_batched(batch)
"""


@params(
    # invalid brightness / contrast / saturation (negative or bad ranges)
    (-0.1, 0.0, 0.0, 0.0),
    ((1.5, 0.5), 0.0, 0.0, 0.0),
    (0.0, -0.2, 0.0, 0.0),
    (0.0, 0.0, -0.3, 0.0),
    # invalid hue beyond allowed range [-0.5, 0.5]
    (0.0, 0.0, 0.0, 0.6),
    (0.0, 0.0, 0.0, (-1.0, 0.2)),
)
def test_colorjitter_invalid_params_match_errors(brightness, contrast, saturation, hue):
    img, _ = make_tensor_inputs()

    with assert_raises(ValueError):
        cj_my = Compose(
            [
                ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            ]
        )
        _ = cj_my(img)
<<<<<<< HEAD


@params("cpu", "gpu")
def test_colorjitter_no_jitter(device):
    cj = Compose(
        [ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, device=device)]
    )
    for fn in test_files:
        img = Image.open(fn)
        out = cj(img)
        out_tensor = transforms.functional.pil_to_tensor(out)
        ref_tensor = transforms.functional.pil_to_tensor(img)
        assert verify_non_one_off(
            out_tensor, ref_tensor
        ), f"No-jitter output differs from input: {fn}"


def test_colorjitter_default_constructor():
    cj = Compose([ColorJitter()])
    img = Image.open(test_files[0])
    out = cj(img)
    assert isinstance(out, Image.Image)


"""
# TODO: grayscale does not work for CHW layout

@cartesian_params((1, 3), ("cpu", "gpu"))
def test_grayscale_tensor(output_channels, device):
    td = Compose([Grayscale(num_output_channels=output_channels, device=device)])

    img_chw = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    out = td(img_chw)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (output_channels, 64, 64), (
        f"Expected ({output_channels}, 64, 64), got {out.shape}"
    )

    batch = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8)
    out_batch = td(batch)
    assert isinstance(out_batch, torch.Tensor)
    assert out_batch.shape == (4, output_channels, 64, 64), (
        f"Expected (4, {output_channels}, 64, 64), got {out_batch.shape}"
    )
"""


@params("cpu", "gpu")
def test_grayscale_single_channel_input(device):

    img = Image.open(test_files[0]).convert("L")

    # TODO: The following test leads to segfault DALI-4655
    """
    td_1ch = Compose([Grayscale(num_output_channels=1, device=device)])
    out_1ch = td_1ch(img)
    assert isinstance(out_1ch, Image.Image)
    assert out_1ch.mode == "L"
    """

    td_3ch = Compose([Grayscale(num_output_channels=3, device=device)])
    out_3ch = td_3ch(img)
    assert isinstance(out_3ch, Image.Image)
    assert out_3ch.mode == "RGB"


@params("cpu", "gpu")
def test_grayscale_functional_3channels(device):
    for fn in test_files:
        img = Image.open(fn)

        out = to_grayscale(img, num_output_channels=3, device=device)
        out_tensor = transforms.functional.pil_to_tensor(out)
        assert out_tensor.shape[0] == 3, f"Expected 3 channels, got {out_tensor.shape[0]}"

        out2 = rgb_to_grayscale(img, num_output_channels=3, device=device)
        out_tensor2 = transforms.functional.pil_to_tensor(out2)
        assert out_tensor2.shape[0] == 3, f"Expected 3 channels, got {out_tensor2.shape[0]}"

        assert verify_non_one_off(
            out_tensor, out_tensor2
        ), f"to_grayscale and rgb_to_grayscale differ: {fn}"
=======
>>>>>>> 2bc5cc11 (Torchvision ColorJitter and Grayscale implementations)
