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

import numpy as np
from PIL import Image
import torch


def pil_to_tensor(inpt: Image.Image | np.ndarray) -> torch.Tensor:
    """
    Convert a ``PIL.Image`` to a uint8 CHW ``torch.Tensor``.

    Values are in [0, 255] and the dtype is ``torch.uint8``.  No scaling is applied.
    Mirrors ``torchvision.transforms.v2.functional.pil_to_tensor``.

    Parameters
    ----------
    inpt : PIL.Image
        Input image.  Modes ``L``, ``RGB``, and ``RGBA`` are supported.

    Returns
    -------
    torch.Tensor
        CHW tensor of dtype ``torch.uint8``.
    """
    if not isinstance(inpt, (Image.Image, np.ndarray)):
        raise TypeError(f"Expected PIL.Image or numpy array, got {type(inpt)}")

    if isinstance(inpt, Image.Image):
        arr = np.array(inpt, copy=True)  # (H, W) for L, (H, W, C) for RGB/RGBA
    else:
        arr = inpt

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)  # (H, W) → (H, W, 1)

    return torch.from_numpy(arr).permute(2, 0, 1)  # (H, W, C) → (C, H, W)


def to_tensor(inpt: Image.Image | np.ndarray) -> torch.Tensor:
    """
    Convert a ``PIL.Image`` to a float32 CHW ``torch.Tensor`` with values in [0, 1].

    Mirrors ``torchvision.transforms.v2.functional.to_tensor`` (deprecated in TV v2,
    but kept here for compatibility).

    Parameters
    ----------
    inpt : PIL.Image
        Input image.  Modes ``L``, ``RGB``, and ``RGBA`` are supported.

    Returns
    -------
    torch.Tensor
        CHW tensor of dtype ``torch.float32`` with values in [0.0, 1.0].
    """
    return pil_to_tensor(inpt).float() / 255.0


def to_pil_image(inpt: torch.Tensor, mode: str | None = None) -> Image.Image:
    """
    Convert a CHW ``torch.Tensor`` to a ``PIL.Image``.

    Mirrors ``torchvision.transforms.v2.functional.to_pil_image``.

    Parameters
    ----------
    inpt : torch.Tensor
        CHW tensor.  Supported channel counts: 1 (``L``), 3 (``RGB``), 4 (``RGBA``).
    mode : str or None, optional
        PIL image mode.  If ``None`` the mode is inferred from the channel count.

    Returns
    -------
    PIL.Image
    """
    if not isinstance(inpt, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(inpt)}")
    if inpt.ndim != 3:
        raise ValueError(f"Expected 3-D CHW tensor, got shape {tuple(inpt.shape)}")

    hwc = inpt.permute(1, 2, 0).cpu()  # (C, H, W) → (H, W, C)
    channels = hwc.shape[-1]

    if mode is None:
        if channels == 1:
            mode = "L"
        elif channels == 3:
            mode = "RGB"
        elif channels == 4:
            mode = "RGBA"
        else:
            raise ValueError(
                f"Cannot infer PIL mode from {channels} channels. " "Pass mode explicitly."
            )

    arr = hwc.numpy()
    if np.issubdtype(arr.dtype, np.floating) and mode != "F":
        arr = (arr * 255).astype(np.uint8)

    if mode == "L":
        arr = arr.squeeze(-1)

    return Image.fromarray(arr, mode=mode)
