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

from nvidia.dali._typing import TensorLike
import nvidia.dali.experimental.dynamic as ndd

from PIL import Image
import torch


def get_size(inpt: Image.Image | TensorLike) -> list[int]:
    """
    Return the spatial size of an image as ``[height, width]``.

    Mirrors ``torchvision.transforms.v2.functional.get_size``.

    Parameters
    ----------
    inpt : PIL Image or TensorLike
        Input image.  Tensors are expected in ``[…, H, W]`` layout (leading
        channel / batch dimensions are ignored).

    Returns
    -------
    list[int]
        ``[height, width]``
    """
    if isinstance(inpt, Image.Image):
        w, h = inpt.size
        return [h, w]  # PIL .size is (W, H)
    elif isinstance(inpt, (torch.Tensor, ndd.Tensor)):
        if inpt.ndim < 2:
            raise TypeError(
                f"get_image_size requires a tensor with at least 2 dimensions, got {inpt.ndim}."
            )
        return [inpt.shape[-2], inpt.shape[-1]]  # [H, W]
    raise TypeError(f"Unsupported input type: {type(inpt)}.")


def get_image_size(inpt: Image.Image | TensorLike) -> list[int]:
    """
    Return the spatial size of an image as ``[width, height]``.

    Mirrors ``torchvision.transforms.v2.functional.get_image_size``.

    .. note::
        This function is provided for compatibility.  The torchvision successor
        ``get_size`` returns ``[height, width]`` instead.

    Parameters
    ----------
    inpt : PIL Image or TensorLike
        Input image.  Tensors are expected in ``[…, H, W]`` layout (leading
        channel / batch dimensions are ignored).

    Returns
    -------
    list[int]
        ``[width, height]``
    """
    h, w = get_size(inpt)
    return [w, h]


def get_dimensions(inpt: Image.Image | TensorLike) -> list[int]:
    """
    Return the number of channels, height, and width of an image as
    ``[channels, height, width]``.

    Mirrors ``torchvision.transforms.v2.functional.get_dimensions``.

    Parameters
    ----------
    inpt : PIL Image or torch.Tensor
        Input image.  Tensors are expected in ``[H, W]`` or ``[…, C, H, W]`` layout
        (leading batch dimensions are ignored).

    Returns
    -------
    List[int]
        ``[channels, height, width]``
    """
    if isinstance(inpt, Image.Image):
        w, h = inpt.size
        return [len(inpt.getbands()), h, w]
    elif isinstance(inpt, (torch.Tensor, ndd.Tensor)):
        if inpt.ndim < 2:
            raise TypeError(
                f"get_dimensions requires a tensor with at least 2 dimensions, got {inpt.ndim}."
            )
        if inpt.ndim == 2:
            return [1, inpt.shape[-2], inpt.shape[-1]]
        return [inpt.shape[-3], inpt.shape[-2], inpt.shape[-1]]  # [C, H, W]
    raise TypeError(f"Unsupported input type: {type(inpt)}.")
