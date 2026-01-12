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
import torchvision.transforms.v2 as T

from nose_utils import assert_raises
from nose2.tools import params
from nvidia.dali.experimental.torchvision import GaussianBlur, Compose
from nvidia.dali.experimental.torchvision.v2.functional import gaussian_blur


def make_input_tensors():
    # 3x32x32 and batched 4x3x32x32
    img = torch.rand(1, 32, 32)
    batch = torch.rand(1, 3, 32, 32)
    return img, batch


@params(
    # fixed sigma (float or sequence)
    ((3, 3), 1.0),
    ((5, 5), 2.0),
    # sigma range (tuple -> random)
    ((3, 3), (0.1, 2.0)),
    ((7, 9), (0.5, 1.5)),
)
def test_gaussian_blur_tensor_equivalence(kernel_size, sigma):
    img, batch = make_input_tensors()

    # ensure same global RNG state before both calls for random sigma cases
    torch.manual_seed(1234)
    tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    tv_out_img = tv_blur(img)
    tv_out_batch = tv_blur(batch)

    torch.manual_seed(1234)
    dali_blur = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma)])
    dali_out_img = dali_blur(img)
    dali_out_batch = dali_blur(batch)

    torch.manual_seed(1234)
    dali_out_fn_img = gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
    dali_out_fn_batch = gaussian_blur(batch, kernel_size=kernel_size, sigma=sigma)

    torch.manual_seed(1234)
    tv_out_fn_img = T.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
    tv_out_fn_batch = T.functional.gaussian_blur(batch, kernel_size=kernel_size, sigma=sigma)

    assert tv_out_img.shape == dali_out_img.shape
    assert tv_out_fn_img.shape == dali_out_fn_img.shape
    assert tv_out_batch.shape == dali_out_batch.shape
    assert tv_out_fn_batch.shape == dali_out_fn_batch.shape

    # allow for small numerical differences
    # TODO: results will differ for min, max due to the random seed
    if isinstance(sigma, float):
        assert torch.allclose(tv_out_img, dali_out_img, atol=1e-5, rtol=1e-4)
        assert torch.allclose(tv_out_fn_img, dali_out_fn_img, atol=1e-5, rtol=1e-4)
        assert torch.allclose(tv_out_batch, dali_out_batch, atol=1e-5, rtol=1e-4)
        assert torch.allclose(tv_out_fn_batch, dali_out_fn_batch, atol=1e-5, rtol=1e-4)


@params(
    (3, 1.0),
    (5, (0.1, 2.0)),
)
def test_gaussian_blur_scalar_kernel_size_expansion(kernel_size, sigma):
    # torchvision allows int kernel_size and expands to (k, k)[web:69][web:73]
    img, _ = make_input_tensors()

    torch.manual_seed(7)
    tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    tv_out = tv_blur(img)

    torch.manual_seed(7)
    dali_blur = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma)])
    dali_out = dali_blur(img)

    torch.manual_seed(7)
    dali_out_fn = gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

    torch.manual_seed(7)
    tv_out_fn = T.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

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
    img, _ = make_input_tensors()
    torch.manual_seed(999)
    tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    tv_outputs = [tv_blur(img) for _ in range(4)]

    torch.manual_seed(999)
    dali_blur = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma)])
    dali_outputs = [dali_blur(img) for _ in range(4)]

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
    img, _ = make_input_tensors()

    with assert_raises(ValueError):
        tv_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        _ = tv_blur(img)

    with assert_raises(ValueError):
        dali_blur = Compose([GaussianBlur(kernel_size=kernel_size, sigma=sigma)])
        _ = dali_blur(img)

    """ Both do not raise ValueError 
    with assert_raises(ValueError):
        _ = gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)

    with assert_raises(ValueError):
        _ = T.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
    """
