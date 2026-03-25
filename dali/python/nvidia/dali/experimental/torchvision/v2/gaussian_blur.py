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

from typing import Sequence, Literal
from .operator import Operator, ArgumentVerificationRule, VerifyIfPositive, VerifyIfRange

import nvidia.dali.fn as fn


class VerifyKernel(ArgumentVerificationRule):
    """
    Verifies the kernel size argument for the GaussianBlur operator.

    Parameters
    ----------
    kernel_size : sequence or int
        Size of the Gaussian kernel. Should be a positive integer or a sequence of two positive
        integers.
    """

    @classmethod
    def verify(cls, *, kernel_size, **_) -> None:

        VerifyIfPositive.verify(values=kernel_size, name="kernel_size")

        if not isinstance(kernel_size, (int, Sequence)):
            raise ValueError(
                f" Kernel size should be an int or a sequenece of ints, got {kernel_size}"
            )

        if isinstance(kernel_size, Sequence) and any(not isinstance(v, int) for v in kernel_size):
            raise ValueError(
                f" Kernel size should be an int or a sequenece of ints, got {kernel_size}"
            )

        values = [kernel_size] if isinstance(kernel_size, int) else kernel_size

        if any(k % 2 == 0 for k in values):
            raise ValueError(
                f"Kernel size values should be odd positive numbers, got {kernel_size}"
            )


class VerifySigma(ArgumentVerificationRule):
    """
    Verifies the sigma argument for the GaussianBlur operator.

    Parameters
    ----------
    sigma : float or tuple of float
        Standard deviation to be used for creating kernel to perform blurring. If float, sigma
        is fixed.
        If it is tuple of float (min, max), sigma is chosen uniformly at random to lie
        in the given range.
    """

    @classmethod
    def verify(cls, *, sigma, **_) -> None:
        if not isinstance(sigma, (int, float, Sequence)) or isinstance(sigma, str):
            raise ValueError(f"sigma must be a float or a (min, max) sequence, got {type(sigma)}")
        if isinstance(sigma, Sequence) and len(sigma) != 2:
            raise ValueError(
                f"sigma sequence must have exactly 2 elements (min, max), got {len(sigma)}"
            )
        VerifyIfPositive.verify(values=sigma, name="sigma")
        if isinstance(sigma, Sequence):
            VerifyIfRange.verify(values=sigma, name="sigma")


class GaussianBlur(Operator):
    """
    Blurs image with randomly chosen Gaussian blur kernel.

    The convolution will be using reflection padding corresponding to the ``kernel size``,
    to maintain the input shape.

    If the input is a ``Tensor``, it is expected to have […, C, H, W] shape, where … means
    an arbitrary number of leading dimensions.

    Parameters
    ----------
    kernel_size : int or sequence
        Size of the Gaussian kernel. Should be a positive integer or a sequence of two positive
        integers.
    sigma : float or tuple of float
        Standard deviation to be used for creating kernel to perform blurring. If float, sigma
        is fixed.
        If it is tuple of float (min, max), sigma is chosen uniformly at random to lie
        in the given range.
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the GaussianBlur. Can be ``"cpu"`` or ``"gpu"``.
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
        self.sigma = (sigma, sigma) if isinstance(sigma, (int, float)) else sigma

    def _kernel(self, data_input):
        if self.sigma[0] != self.sigma[1]:
            sigma = fn.random.uniform(range=self.sigma)
        else:
            sigma = self.sigma[0]

        return fn.gaussian_blur(
            data_input,
            window_size=self.kernel_size,
            sigma=sigma,
            device=self.device,
        )
