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
import unittest

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as tv_fn

from nose_utils import assert_raises
from nose2.tools import params
from PIL import Image
from nvidia.dali.experimental.torchvision import GaussianBlur, Compose
from nvidia.dali.experimental.torchvision.v2.functional import gaussian_blur

import nvidia.dali.experimental.dynamic as ndd

dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]


def make_input_tensors():
    # 3x32x32 and batched 2x3x32x32
    tens = torch.rand(1, 32, 32)
    batch = torch.rand(2, 3, 32, 32)
    return tens, batch


@params(
    # fixed sigma (float or sequence)
    ((3, 3), 1.0),
    ((5, 5), 2.0),
    # sigma range (tuple -> random)
    ((3, 3), (0.1, 2.0)),
    ((7, 9), (0.5, 1.5)),
)
def test_gaussian_blur_tensor_equivalence(kernel_size, sigma):
    tens, batch = make_input_tensors()

    seed = 1234
    # ensure same global RNG state before both calls for random sigma cases
    torch.manual_seed(seed)
    tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    tv_out_tens = tv_blur(tens)
    tv_out_batch = tv_blur(batch)

    dali_blur = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma, seed=seed)])
    dali_out_tens = dali_blur(tens)
    dali_out_batch = dali_blur(batch)

    ndd.random.set_seed(seed)
    dali_out_fn_tens = gaussian_blur(tens, kernel_size=kernel_size, sigma=sigma)
    dali_out_fn_batch = gaussian_blur(batch, kernel_size=kernel_size, sigma=sigma)

    torch.manual_seed(seed)
    tv_out_fn_tens = T.functional.gaussian_blur(tens, kernel_size=kernel_size, sigma=sigma)
    tv_out_fn_batch = T.functional.gaussian_blur(batch, kernel_size=kernel_size, sigma=sigma)

    assert tv_out_tens.shape == dali_out_tens.shape
    assert tv_out_fn_tens.shape == dali_out_fn_tens.shape
    assert tv_out_batch.shape == dali_out_batch.shape
    assert tv_out_fn_batch.shape == dali_out_fn_batch.shape

    # allow for small numerical differences
    # TODO: results will differ for min, max due to the random seed
    if isinstance(sigma, float):
        assert torch.allclose(tv_out_tens, dali_out_tens, atol=1e-5, rtol=1e-4)
        assert torch.allclose(tv_out_fn_tens, dali_out_fn_tens, atol=1e-5, rtol=1e-4)
        assert torch.allclose(tv_out_batch, dali_out_batch, atol=1e-5, rtol=1e-4)
        assert torch.allclose(tv_out_fn_batch, dali_out_fn_batch, atol=1e-5, rtol=1e-4)


@params(
    (3, 1.0),
    (5, (0.1, 2.0)),
)
def test_gaussian_blur_scalar_kernel_size_expansion(kernel_size, sigma):
    # torchvision allows int kernel_size and expands to (k, k)[web:69][web:73]
    tens, _ = make_input_tensors()

    torch.manual_seed(7)
    tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    tv_out = tv_blur(tens)

    torch.manual_seed(7)
    dali_blur = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma)])
    dali_out = dali_blur(tens)

    torch.manual_seed(7)
    dali_out_fn = gaussian_blur(tens, kernel_size=kernel_size, sigma=sigma)

    torch.manual_seed(7)
    tv_out_fn = T.functional.gaussian_blur(tens, kernel_size=kernel_size, sigma=sigma)

    assert tv_out.shape == dali_out.shape
    assert tv_out.shape == dali_out_fn.shape
    # TODO: results will differ for min, max due to the random seed
    if isinstance(sigma, float):
        assert torch.allclose(tv_out, dali_out, atol=1e-5, rtol=1e-4)
        assert torch.allclose(tv_out_fn, dali_out_fn, atol=1e-5, rtol=1e-4)


"""
@params(
    ((3, 3), (0.1, 2.0)),
)
def test_gaussian_blur_random_sigma_distribution(kernel_size, sigma):
    # Check that both implementations make the same random sigma choices
    tens, _ = make_input_tensors()
    torch.manual_seed(999)
    tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    tv_outputs = [tv_blur(tens) for _ in range(4)]

    torch.manual_seed(999)
    dali_blur = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma)])
    dali_outputs = [dali_blur(tens) for _ in range(4)]

    for tv_out, dali_out in zip(tv_outputs, dali_outputs):
        assert tv_out.shape == dali_out.shape
        assert torch.allclose(tv_out, dali_out, atol=1e-5, rtol=1e-4)
"""


@params(
    ((3, 3), -1.0),  # invalid sigma
    ((0, 3), 1.0),  # kernel size <= 0
    ((3, 3), (2.0, 0.1)),  # min > max
)
def test_gaussian_blur_invalid_params_match_errors(kernel_size, sigma):
    tens, _ = make_input_tensors()

    with assert_raises(ValueError):
        tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        _ = tv_blur(tens)

    with assert_raises(ValueError):
        dali_blur = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma)])
        _ = dali_blur(tens)

    """ Both do not raise ValueError
    with assert_raises(ValueError):
        _ = gaussian_blur(tens, kernel_size=kernel_size, sigma=sigma)

    with assert_raises(ValueError):
        _ = T.functional.gaussian_blur(tens, kernel_size=kernel_size, sigma=sigma)
    """


def _test_pil_images(kernel_size, sigma, device):
    t = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    td = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma, device=device)])

    for path in test_files:
        img = Image.open(path)

        out_tv = tv_fn.pil_to_tensor(t(img))
        out_dali = tv_fn.pil_to_tensor(td(img))
        out_tv_fn = tv_fn.pil_to_tensor(
            T.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        )
        out_dali_fn = tv_fn.pil_to_tensor(
            gaussian_blur(img, kernel_size=kernel_size, sigma=sigma, device=device)
        )

        assert out_tv.shape == out_dali.shape, f"Shape mismatch: {path}"
        assert out_tv_fn.shape == out_dali_fn.shape, f"Functional shape mismatch: {path}"

        if isinstance(sigma, float):
            assert torch.allclose(
                out_tv.float(), out_dali.float(), atol=1.0, rtol=0
            ), f"Values differ by more than 1: {path}"
            assert torch.allclose(
                out_tv_fn.float(), out_dali_fn.float(), atol=1.0, rtol=0
            ), f"Functional values differ by more than 1: {path}"


@params(
    ((3, 3), 1.0),
    ((5, 5), 2.0),
    ((3, 3), (0.1, 2.0)),
)
def test_gaussian_blur_pil_images_cpu(kernel_size, sigma):
    _test_pil_images(kernel_size, sigma, "cpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
@params(
    ((3, 3), 1.0),
    ((5, 5), 2.0),
    ((3, 3), (0.1, 2.0)),
)
def test_gaussian_blur_pil_images_gpu(kernel_size, sigma):
    _test_pil_images(kernel_size, sigma, "gpu")


def _test_tensor_device(kernel_size, sigma, device):
    tens, batch = make_input_tensors()

    if device == "gpu":
        tens = tens.cuda()
        batch = batch.cuda()

    tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    out_tv = tv_blur(tens)
    out_tv_batch = tv_blur(batch)

    td = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma, device=device)])
    out_dali = td(tens)
    out_dali_batch = td(batch)

    out_dali_fn = gaussian_blur(tens, kernel_size=kernel_size, sigma=sigma, device=device)
    out_dali_fn_batch = gaussian_blur(batch, kernel_size=kernel_size, sigma=sigma, device=device)

    assert out_dali.shape == out_tv.shape
    assert out_dali_batch.shape == out_tv_batch.shape
    assert out_dali_fn.shape == out_tv.shape
    assert out_dali_fn_batch.shape == out_tv_batch.shape

    if isinstance(sigma, float):
        assert torch.allclose(out_tv.float().cpu(), out_dali.float().cpu(), atol=1e-5, rtol=1e-4)
        assert torch.allclose(
            out_tv_batch.float().cpu(), out_dali_batch.float().cpu(), atol=1e-5, rtol=1e-4
        )
        assert torch.allclose(out_tv.float().cpu(), out_dali_fn.float().cpu(), atol=1e-5, rtol=1e-4)
        assert torch.allclose(
            out_tv_batch.float().cpu(), out_dali_fn_batch.float().cpu(), atol=1e-5, rtol=1e-4
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
@params(
    ((3, 3), 1.0),
    ((5, 5), 2.0),
    ((3, 3), (0.1, 2.0)),
)
def test_gaussian_blur_tensor_gpu(kernel_size, sigma):
    _test_tensor_device(kernel_size, sigma, "gpu")
