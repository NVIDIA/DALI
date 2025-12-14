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

from typing import Sequence, Literal

from nose2.tools import params

# from nose_utils import assert_raises
import torch
import torchvision.transforms.v2 as transforms

from nvidia.dali.experimental.torchvision import Normalize, Compose


def make_test_tensor(shape=(1, 10, 10, 1)):
    total = 1.0
    for s in shape:
        total *= float(s)
    return torch.arange(total).reshape(shape).to(dtype=torch.float)


def test_normalize_core(
    input_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    device: Literal["cpu", "gpu"] = "cpu",
):
    transform = Compose([Normalize(mean=mean, std=std, device=device)])
    transform_tv = transforms.Normalize(mean=mean, std=std)

    out = transform(input_tensor)
    out_tv = transform_tv(input_tensor)

    if device == "gpu":
        out = out.cpu()
    assert torch.allclose(out, out_tv), f"Diff: {out-out_tv}"


@params(
    (
        1,
        [5.0],
        [0.05],
    ),
    (
        2,
        [2.1, 3.2],
        [0.02, 0.05],
    ),
    (
        3,
        [1.0, 0.3, 5.0],
        [0.5, 0.01, 0.023],
    ),
)
def test_normalize_channels(channels: int, mean: Sequence[float], std: Sequence[float]):
    intensor = make_test_tensor(shape=(1, channels, 5, 5))
    test_normalize_core(intensor, mean, std)


"""
def make_sample_tensor():
    # 3x2x2 (C, H, W)
    return torch.tensor(
        [
            [[1.0, 2.0],
             [3.0, 4.0]],
            [[5.0, 6.0],
             [7.0, 8.0]],
            [[9.0, 10.0],
             [11.0, 12.0]],
        ]
    )

@params(
    ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
    ((1.0, 2.0, 3.0), (1.0, 1.0, 1.0)),
    (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 1.0, 1.0])),
)
def test_mean_std_sequence_lengths_match_channels(mean, std):
    x = make_sample_tensor()
    norm = transforms.Normalize(mean=mean, std=std)
    dnorm = Compose([Normalize(mean = mean, std=std)])
    out = norm(x)
    dout = dnorm(x)
    assert out.shape == x.shape
    assert out.shape == dout.shape


@params(
    ([1.0, 2.0], [1.0, 1.0, 1.0]),
    ([1.0, 2.0, 3.0], [1.0, 1.0]),
)
def test_mismatched_mean_or_std_length_raises(mean, std):
    x = make_sample_tensor()
    norm = transforms.Normalize(mean=mean, std=std)
    with assert_raises(RuntimeError):
        _ = norm(x)


@params(
    ([0.0, 0.0, 0.0], [1.0, 0.0, 1.0]),
    ([0.0, 0.0, 0.0], [0.0, 1.0, 1.0]),
)
def test_std_must_be_non_zero(mean, std):
    x = make_sample_tensor()
    norm = transforms.Normalize(mean=mean, std=std)
    with assert_raises(ValueError):
        _ = norm(x)


@params(False, True)
def test_inplace_behavior(inplace):
    x = make_sample_tensor()
    norm = transforms.Normalize(mean=[1.0, 2.0, 3.0],
                                std=[1.0, 1.0, 1.0],
                                inplace=inplace)
    orig = x.clone()
    out = norm(x)

    if inplace:
        # In-place: returned tensor and input should be same storage
        assert out.data_ptr() == x.data_ptr()
        assert not torch.allclose(x, orig)
    else:
        assert torch.allclose(x, orig)
        assert not torch.allclose(out, orig)


def test_non_tensor_input_not_supported():
    norm = transforms.Normalize(mean=[0.5], std=[0.5])
    imarray = np.random.rand(100,100,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
    with assert_raises(TypeError):
        #_ = norm([[1.0, 2.0], [3.0, 4.0]])  # not a tensor
        _ = norm(im)  # not a tensor


@params(
    (torch.ones(1, 2, 2), [0.5], [0.5], 1.0),
    (torch.ones(4, 3, 2, 2), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1.0),
)
def test_single_channel_and_batch_shapes(x, mean, std, expected_value):
    norm = transforms.Normalize(mean=mean, std=std)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.full_like(out, expected_value))


def test_broadcast_over_spatial_dimensions():
    x = make_sample_tensor()
    mean = [1.0, 5.0, 9.0]
    std = [1.0, 1.0, 1.0]
    norm = transforms.Normalize(mean=mean, std=std)
    out = norm(x)

    out_expect = torch.zeros_like(out)
    for i in range(len(mean)):
        out_expect[i] = (x[i]-mean[i])/std[i]
    # Channel 0, value 1 -> (1 - 1) / 1 = 0
    # Channel 1, value 6 -> (6 - 5) / 1 = 1
    # Channel 2, value 12 -> (12 - 9) / 1 = 3
    assert torch.allclose(out, out_expect)


def test_different_std_per_channel():
    x = make_sample_tensor()
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 2.0, 4.0]
    norm = transforms.Normalize(mean=mean, std=std)
    out = norm(x)

    out_expect = torch.zeros_like(out)
    for i in range(len(mean)):
        out_expect[i] = (x[i]-mean[i])/std[i]
    # Channel 0 unchanged, channel 1 halved, channel 2 quartered
    assert torch.allclose(out, out_expect)
"""
