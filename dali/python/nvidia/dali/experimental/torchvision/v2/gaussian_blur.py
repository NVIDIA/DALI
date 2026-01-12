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
import nvidia.dali.fn as fn
from .operator import Operator, ArgumentVerificationRule, VerifyIfPositive


class VerifyKernel(ArgumentVerificationRule):
    @classmethod
    def verify(cls, *, kernel_size, **_) -> None:
        VerifyIfPositive.verify(values=kernel_size)


class VerifySigma(ArgumentVerificationRule):
    @classmethod
    def verify(cls, *, sigma, **_) -> None:
        VerifyIfPositive.verify(values=sigma)
        if isinstance(sigma, (list, tuple)) and sigma[1] < sigma[0]:
            raise ValueError("sigma values should be positive and of the form (min, max)")


class GaussianBlur(Operator):
    """
    Blurs image with randomly chosen Gaussian blur kernel.

    The convolution will be using reflection padding corresponding to the kernel size,
    to maintain the input shape.

    If the input is a Tensor, it is expected to have […, C, H, W] shape, where … means
    an arbitrary number of leading dimensions.
    Normalize a tensor image or video with mean and standard deviation.

    Parameters:

    kernel_size (int or sequence) – Size of the Gaussian kernel.
    sigma (float or tuple of python:float (min, max)) – Standard deviation to be used
            for creating kernel to perform blurring. If float, sigma is fixed.
            If it is tuple of float (min, max), sigma is chosen uniformly at random
            to lie in the given range.

    """

    arg_rules = [VerifyKernel, VerifySigma]

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        sigma: int | float | Sequence[float] = (0.1, 2.0),
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(device=device, kernel_size=kernel_size, sigma=sigma)

        self.kernel_size = kernel_size
        self.sigma = sigma

    def _kernel(self, data_input):

        return fn.gaussian_blur(
            data_input,
            window_size=self.kernel_size,
            sigma=self.sigma,
            device=self.device,
        )
