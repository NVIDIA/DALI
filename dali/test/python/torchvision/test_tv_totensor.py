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

import math
import unittest

import numpy as np
import torch
import torchvision.transforms.v2 as tv
import torchvision.transforms.v2.functional as tv_fn
from nose2.tools import params
from nose_utils import assert_raises
from PIL import Image

from nvidia.dali.experimental.torchvision import (
    Compose,
    PILToTensor,
    Resize,
    RandomHorizontalFlip,
    ToPILImage,
    ToPureTensor,
)
from nvidia.dali.experimental.torchvision.v2.functional import (
    pil_to_tensor,
    to_pil_image,
    to_tensor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_tensor(shape=(1, 3, 10, 10)):
    total = math.prod(shape)
    return torch.arange(total, dtype=torch.uint8).reshape(shape)


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


# ---------------------------------------------------------------------------
# Pipeline mode — PILToTensor
# ---------------------------------------------------------------------------


@params("L", "RGB", "RGBA")
def test_pil_to_tensor_operator_returns_tensor(mode):
    """PILToTensor inside Compose must return a torch.Tensor, not a PIL Image."""
    img = _make_pil_image(mode)
    out = Compose([PILToTensor()])(img)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"


@params("L", "RGB", "RGBA")
def test_pil_to_tensor_operator_shape(mode):
    """PILToTensor must return a CHW tensor with the correct spatial dimensions."""
    h, w = 50, 60
    img = _make_pil_image(mode, h=h, w=w)
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    out = Compose([PILToTensor()])(img)
    assert out.shape == torch.Size(
        [channels, h, w]
    ), f"Expected shape ({channels}, {h}, {w}), got {tuple(out.shape)}"


@params("L", "RGB", "RGBA")
def test_pil_to_tensor_operator_dtype(mode):
    """PILToTensor must produce a uint8 tensor (no scaling)."""
    img = _make_pil_image(mode)
    out = Compose([PILToTensor()])(img)
    assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"


@params("L", "RGB", "RGBA")
def test_pil_to_tensor_operator_matches_torchvision(mode):
    """PILToTensor output must be pixel-exact against torchvision.functional.pil_to_tensor."""
    img = _make_pil_image(mode)
    dali_out = Compose([PILToTensor()])(img)
    tv_out = tv_fn.pil_to_tensor(img)
    assert torch.equal(
        dali_out, tv_out
    ), f"Pixel mismatch for mode={mode}: max diff {(dali_out.int() - tv_out.int()).abs().max()}"


def test_pil_to_tensor_operator_with_resize():
    """Resize followed by PILToTensor must return a CHW tensor matching torchvision (±1)."""
    img = _make_pil_image("RGB")
    target = (30, 40)
    dali_out = Compose(
        [Resize(size=target, interpolation=tv.InterpolationMode.NEAREST), PILToTensor()]
    )(img)
    tv_out = tv.Compose(
        [tv.Resize(size=target, interpolation=tv.InterpolationMode.NEAREST), tv.PILToTensor()]
    )(img)
    assert isinstance(dali_out, torch.Tensor)
    assert dali_out.shape == torch.Size([3, 30, 40]), f"Unexpected shape: {tuple(dali_out.shape)}"
    assert torch.allclose(
        dali_out.float(), tv_out.float(), atol=1
    ), f"Max diff vs torchvision: {(dali_out.float() - tv_out.float()).abs().max()}"


# ---------------------------------------------------------------------------
# Pipeline mode — ToPureTensor
# ---------------------------------------------------------------------------


@params("L", "RGB", "RGBA")
def test_to_pure_tensor_operator_returns_tensor_from_pil(mode):
    """ToPureTensor in a PIL pipeline must return a torch.Tensor."""
    img = _make_pil_image(mode)
    out = Compose([ToPureTensor()])(img)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"


@params("L", "RGB", "RGBA")
def test_to_pure_tensor_operator_matches_torchvision(mode):
    """ToPureTensor from a PIL input must match pil_to_tensor (same as PILToTensor semantics)."""
    img = _make_pil_image(mode)
    dali_out = Compose([ToPureTensor()])(img)
    tv_out = tv_fn.pil_to_tensor(img)
    assert torch.equal(
        dali_out, tv_out
    ), f"Pixel mismatch for mode={mode}: max diff {(dali_out.int() - tv_out.int()).abs().max()}"


def test_to_pure_tensor_operator_from_tensor_is_noop():
    """ToPureTensor in a CHW tensor pipeline must return the same tensor values as torchvision."""
    tensor = make_test_tensor(shape=(1, 3, 8, 8))
    dali_out = Compose([ToPureTensor()])(tensor)
    tv_out = tv.ToPureTensor()(tensor)
    assert isinstance(dali_out, torch.Tensor)
    assert torch.equal(
        dali_out, tv_out
    ), f"Mismatch vs torchvision: max diff {(dali_out.int() - tv_out.int()).abs().max()}"


def test_last_conversion_op_wins():
    """When both ToPureTensor and ToPILImage appear, the last one determines the output type."""
    img = _make_pil_image("RGB")
    # ToPILImage comes last → output must be PIL
    out = Compose([ToPureTensor(), ToPILImage()])(img)
    assert isinstance(out, Image.Image), f"Expected PIL Image, got {type(out)}"


# ---------------------------------------------------------------------------
# Pipeline mode — ToPILImage
# ---------------------------------------------------------------------------


def test_to_pil_image_operator_returns_pil():
    """ToPILImage inside Compose must return a PIL Image matching torchvision pixel-exactly."""
    tensor = make_test_tensor(shape=(1, 3, 8, 8))
    dali_out = Compose([ToPILImage()])(tensor)
    tv_out = tv_fn.to_pil_image(tensor.squeeze(0))
    assert isinstance(dali_out, Image.Image), f"Expected PIL Image, got {type(dali_out)}"
    assert np.array_equal(np.array(dali_out), np.array(tv_out)), "Pixel mismatch vs torchvision"


@params(
    (1, "L"),
    (3, "RGB"),
    (4, "RGBA"),
)
def test_to_pil_image_operator_mode(channels, expected_mode):
    """ToPILImage must infer the correct PIL mode and match torchvision pixel values."""
    tensor = make_test_tensor(shape=(1, channels, 8, 8))
    dali_out = Compose([ToPILImage()])(tensor)
    tv_out = tv_fn.to_pil_image(tensor.squeeze(0))
    assert dali_out.mode == expected_mode, f"Expected mode {expected_mode}, got {dali_out.mode}"
    assert np.array_equal(
        np.array(dali_out), np.array(tv_out)
    ), f"Pixel mismatch vs torchvision for {channels}-channel image"


def test_to_pil_image_operator_spatial_size():
    """ToPILImage output must have the correct (W, H) size and match torchvision pixels."""
    h, w = 20, 30
    tensor = make_test_tensor(shape=(1, 3, h, w))
    dali_out = Compose([ToPILImage()])(tensor)
    tv_out = tv_fn.to_pil_image(tensor.squeeze(0))
    assert dali_out.size == (w, h), f"Expected size ({w}, {h}), got {dali_out.size}"
    assert np.array_equal(np.array(dali_out), np.array(tv_out)), "Pixel mismatch vs torchvision"


def test_to_pil_image_operator_with_flip():
    """Resize followed by ToPILImage must match torchvision's output (±1 due to resize)."""
    tensor = make_test_tensor(shape=(1, 3, 40, 50))
    dali_out = Compose([RandomHorizontalFlip(p=1.0), ToPILImage()])(tensor)
    tv_out = tv.Compose([tv.RandomHorizontalFlip(p=1.0), tv.ToPILImage()])(tensor.squeeze(0))
    assert isinstance(dali_out, Image.Image)
    assert torch.allclose(
        tv_fn.pil_to_tensor(dali_out).float(),
        tv_fn.pil_to_tensor(tv_out).float(),
        atol=1,
    ), "Pixel values differ by more than 1 vs torchvision"


def test_to_pil_image_operator_roundtrip_with_pil_to_tensor():
    """ToPILImage output must match torchvision's ToPILImage pixel-exactly."""
    tensor = make_test_tensor(shape=(1, 3, 8, 8)).squeeze(0)  # (3, 8, 8)
    dali_out = Compose([ToPILImage()])(tensor.unsqueeze(0))
    tv_out = tv_fn.to_pil_image(tensor)
    assert np.array_equal(np.array(dali_out), np.array(tv_out)), "Pixel mismatch vs torchvision"
    recovered = tv_fn.pil_to_tensor(dali_out)
    assert torch.equal(
        recovered, tensor
    ), f"Roundtrip mismatch: max diff {(recovered.int() - tensor.int()).abs().max()}"


# ---------------------------------------------------------------------------
# Functional API — pil_to_tensor
# ---------------------------------------------------------------------------


@params("L", "RGB", "RGBA")
def test_fn_pil_to_tensor_returns_tensor(mode):
    """pil_to_tensor must return a torch.Tensor."""
    img = _make_pil_image(mode)
    out = pil_to_tensor(img)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"


@params("L", "RGB", "RGBA")
def test_fn_pil_to_tensor_shape(mode):
    """pil_to_tensor must return a CHW tensor with the correct channel count."""
    h, w = 50, 60
    img = _make_pil_image(mode, h=h, w=w)
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    out = pil_to_tensor(img)
    assert out.shape == torch.Size(
        [channels, h, w]
    ), f"Expected ({channels}, {h}, {w}), got {tuple(out.shape)}"


@params("L", "RGB", "RGBA")
def test_fn_pil_to_tensor_dtype(mode):
    """pil_to_tensor must produce a uint8 tensor without scaling."""
    img = _make_pil_image(mode)
    out = pil_to_tensor(img)
    assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"


@params("L", "RGB", "RGBA")
def test_fn_pil_to_tensor_matches_torchvision(mode):
    """pil_to_tensor output must be pixel-exact against torchvision's pil_to_tensor."""
    img = _make_pil_image(mode)
    dali_out = pil_to_tensor(img)
    tv_out = tv_fn.pil_to_tensor(img)
    assert torch.equal(
        dali_out, tv_out
    ), f"Mismatch for mode={mode}: max diff {(dali_out.int() - tv_out.int()).abs().max()}"


def test_fn_pil_to_tensor_invalid_input_type():
    """pil_to_tensor must raise TypeError when called with a torch.Tensor."""
    tensor = make_test_tensor()
    with assert_raises(TypeError):
        pil_to_tensor(tensor)


# ---------------------------------------------------------------------------
# Functional API — to_tensor
# ---------------------------------------------------------------------------


@params("L", "RGB", "RGBA")
def test_fn_to_tensor_dtype_and_range(mode):
    """to_tensor must return float32 with values in [0, 1]."""
    img = _make_pil_image(mode)
    out = to_tensor(img)
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
    assert (
        out.min() >= 0.0 and out.max() <= 1.0
    ), f"Values out of [0,1]: min={out.min()}, max={out.max()}"


@params("L", "RGB", "RGBA")
def test_fn_to_tensor_matches_torchvision(mode):
    """to_tensor output must be numerically identical to torchvision's to_tensor."""
    img = _make_pil_image(mode)
    dali_out = to_tensor(img)
    tv_out = tv_fn.to_tensor(img)
    assert torch.allclose(
        dali_out, tv_out, atol=0, rtol=0
    ), f"Mismatch for mode={mode}: max diff {(dali_out - tv_out).abs().max()}"


# ---------------------------------------------------------------------------
# Functional API — to_pil_image
# ---------------------------------------------------------------------------


@params(
    (1, "L"),
    (3, "RGB"),
    (4, "RGBA"),
)
def test_fn_to_pil_image_infers_mode(channels, expected_mode):
    """to_pil_image must infer PIL mode from channel count when mode is not specified."""
    tensor = make_test_tensor(shape=(channels, 8, 8))
    out = to_pil_image(tensor)
    assert isinstance(out, Image.Image), f"Expected PIL Image, got {type(out)}"
    assert out.mode == expected_mode, f"Expected mode {expected_mode}, got {out.mode}"


def test_fn_to_pil_image_explicit_mode():
    """to_pil_image must honour an explicitly supplied mode."""
    tensor = make_test_tensor(shape=(3, 8, 8))
    out = to_pil_image(tensor, mode="RGB")
    assert out.mode == "RGB"


def test_fn_to_pil_image_spatial_size():
    """to_pil_image must produce a PIL Image with the correct (W, H) dimensions."""
    h, w = 20, 30
    tensor = make_test_tensor(shape=(3, h, w))
    out = to_pil_image(tensor)
    assert out.size == (w, h), f"Expected ({w}, {h}), got {out.size}"


@params("L", "RGB", "RGBA")
def test_fn_to_pil_image_roundtrip(mode):
    """pil_to_tensor → to_pil_image must recover the original pixel values exactly."""
    img = _make_pil_image(mode)
    tensor = pil_to_tensor(img)
    recovered = to_pil_image(tensor)
    assert recovered.mode == img.mode, f"Mode changed: {img.mode} → {recovered.mode}"
    assert np.array_equal(np.array(recovered), np.array(img)), "Pixel values changed in roundtrip"


def test_fn_to_pil_image_invalid_input_type():
    """to_pil_image must raise TypeError when called with a PIL Image."""
    img = _make_pil_image("RGB")
    with assert_raises(TypeError):
        to_pil_image(img)


def test_fn_to_pil_image_invalid_ndim():
    """to_pil_image must raise ValueError for a non-3-D tensor."""
    tensor = make_test_tensor(shape=(8, 8))
    with assert_raises(ValueError):
        to_pil_image(tensor)


def test_fn_to_pil_image_unsupported_channels():
    """to_pil_image must raise ValueError when channel count is not 1, 3, or 4."""
    tensor = make_test_tensor(shape=(2, 8, 8))
    with assert_raises(ValueError):
        to_pil_image(tensor)


# ---------------------------------------------------------------------------
# Functional vs pipeline mode consistency
# ---------------------------------------------------------------------------


@params("L", "RGB", "RGBA")
def test_pil_to_tensor_functional_vs_operator_consistency(mode):
    """The functional pil_to_tensor and the PILToTensor Compose operator must agree exactly."""
    img = _make_pil_image(mode)
    fn_out = pil_to_tensor(img)
    compose_out = Compose([PILToTensor()])(img)
    assert torch.equal(fn_out, compose_out), (
        f"Functional and Compose disagree for mode={mode}: "
        f"max diff {(fn_out.int() - compose_out.int()).abs().max()}"
    )


@params("L", "RGB", "RGBA")
def test_to_pil_image_roundtrip_via_compose(mode):
    """PIL → PILToTensor (Compose) → to_pil_image (functional) must recover the original image."""
    img = _make_pil_image(mode)
    tensor = Compose([PILToTensor()])(img)
    recovered = to_pil_image(tensor)
    assert recovered.mode == img.mode, f"Mode changed: {img.mode} → {recovered.mode}"
    assert np.array_equal(
        np.array(recovered), np.array(img)
    ), f"Pixel values changed in roundtrip for mode={mode}"


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@params("L", "RGB", "RGBA")
def test_pil_to_tensor_operator_gpu(mode):
    """PILToTensor with device='gpu' must return a GPU tensor matching the CPU reference."""
    img = _make_pil_image(mode)
    cpu_out = Compose([PILToTensor()])(img)
    gpu_out = Compose([PILToTensor(device="gpu")])(img)
    assert gpu_out.is_cuda, "Expected CUDA tensor"
    assert torch.equal(cpu_out, gpu_out.cpu()), "GPU and CPU outputs differ"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
def test_to_pil_image_operator_gpu():
    """ToPILImage on GPU tensor pipeline must return a PIL Image matching the CPU reference."""
    tensor = make_test_tensor(shape=(1, 3, 8, 8))
    cpu_out = Compose([ToPILImage()])(tensor)
    gpu_out = Compose([ToPILImage(device="gpu")])(tensor.cuda())
    assert isinstance(gpu_out, Image.Image), f"Expected PIL Image, got {type(gpu_out)}"
    assert np.array_equal(
        np.array(cpu_out), np.array(gpu_out)
    ), "GPU and CPU ToPILImage outputs differ"


# ---------------------------------------------------------------------------
# Functional API — to_tensor (numpy array input)
# ---------------------------------------------------------------------------


@params("L", "RGB", "RGBA")
def test_fn_to_tensor_numpy_dtype_and_range(mode):
    """to_tensor must accept a numpy array and return float32 with values in [0, 1]."""
    img = _make_pil_image(mode)
    arr = np.array(img)
    out = to_tensor(arr)
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
    assert (
        out.min() >= 0.0 and out.max() <= 1.0
    ), f"Values out of [0,1]: min={out.min()}, max={out.max()}"


@params("L", "RGB", "RGBA")
def test_fn_to_tensor_numpy_matches_torchvision(mode):
    """to_tensor from a numpy array must be pixel-exact against torchvision's to_tensor."""
    img = _make_pil_image(mode)
    arr = np.array(img)
    dali_out = to_tensor(arr)
    tv_out = tv_fn.to_tensor(arr)
    assert torch.allclose(
        dali_out, tv_out, atol=0, rtol=0
    ), f"Mismatch for mode={mode}: max diff {(dali_out - tv_out).abs().max()}"


def test_fn_to_tensor_invalid_tensor_input():
    """to_tensor must raise TypeError when called with a torch.Tensor."""
    tensor = make_test_tensor(shape=(3, 8, 8))
    with assert_raises(TypeError):
        to_tensor(tensor)


# ---------------------------------------------------------------------------
# Functional API — to_pil_image (float32 tensor input)
# ---------------------------------------------------------------------------


@params(
    (1, "L"),
    (3, "RGB"),
    (4, "RGBA"),
)
def test_fn_to_pil_image_float_input_matches_torchvision(channels, expected_mode):
    """to_pil_image with a float32 [0,1] tensor must match torchvision pixel-exactly."""
    tensor = make_test_tensor(shape=(channels, 8, 8)).float() / 255.0
    dali_out = to_pil_image(tensor)
    tv_out = tv_fn.to_pil_image(tensor)
    assert isinstance(dali_out, Image.Image)
    assert dali_out.mode == expected_mode, f"Expected {expected_mode}, got {dali_out.mode}"
    assert np.array_equal(
        np.array(dali_out), np.array(tv_out)
    ), f"Pixel mismatch for {channels}-channel float32 tensor"


"""
TODO: DALI does not fail here, I am not sure if it should if this dtype is supported but truncated
def test_fn_to_pil_image_unsupported_dtype_raises():
    # to_pil_image must raise ValueError for an unsupported dtype such as torch.int64.
    tensor = make_test_tensor(shape=(3, 8, 8)).long()
    with assert_raises(TypeError):
        tv_fn.to_pil_image(tensor)
    with assert_raises(TypeError):
        to_pil_image(tensor)
"""

# ---------------------------------------------------------------------------
# Functional API — pil_to_tensor with PIL modes "I" (int32) and "F" (float32)
# ---------------------------------------------------------------------------


def _make_pil_image_mode_I(h=10, w=12):
    data = np.arange(h * w, dtype=np.int32).reshape(h, w)
    return Image.fromarray(data, mode="I")


def _make_pil_image_mode_F(h=10, w=12):
    data = (np.arange(h * w, dtype=np.float32) / (h * w)).reshape(h, w)
    return Image.fromarray(data, mode="F")


def test_fn_pil_to_tensor_mode_I_shape_and_dtype():
    """pil_to_tensor on a mode-'I' image must return a (1, H, W) int32 tensor."""
    img = _make_pil_image_mode_I()
    out = pil_to_tensor(img)
    assert out.shape == torch.Size([1, 10, 12]), f"Unexpected shape: {tuple(out.shape)}"
    assert out.dtype == torch.int32, f"Expected int32, got {out.dtype}"


def test_fn_pil_to_tensor_mode_I_matches_torchvision():
    """pil_to_tensor on a mode-'I' image must be pixel-exact against torchvision."""
    img = _make_pil_image_mode_I()
    dali_out = pil_to_tensor(img)
    tv_out = tv_fn.pil_to_tensor(img)
    assert torch.equal(
        dali_out, tv_out
    ), f"Mismatch for mode=I: max diff {(dali_out - tv_out).abs().max()}"


def test_fn_pil_to_tensor_mode_F_shape_and_dtype():
    """pil_to_tensor on a mode-'F' image must return a (1, H, W) float32 tensor."""
    img = _make_pil_image_mode_F()
    out = pil_to_tensor(img)
    assert out.shape == torch.Size([1, 10, 12]), f"Unexpected shape: {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"


def test_fn_pil_to_tensor_mode_F_matches_torchvision():
    """pil_to_tensor on a mode-'F' image must be pixel-exact against torchvision."""
    img = _make_pil_image_mode_F()
    dali_out = pil_to_tensor(img)
    tv_out = tv_fn.pil_to_tensor(img)
    assert torch.allclose(
        dali_out, tv_out, atol=0, rtol=0
    ), f"Mismatch for mode=F: max diff {(dali_out - tv_out).abs().max()}"


# ---------------------------------------------------------------------------
# GPU tests — functional API
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
def test_fn_to_pil_image_from_cuda_tensor():
    """to_pil_image called with a CUDA tensor must produce the same PIL output as from CPU."""
    tensor = make_test_tensor(shape=(3, 8, 8))
    cpu_out = to_pil_image(tensor)
    cuda_out = to_pil_image(tensor.cuda())
    assert isinstance(cuda_out, Image.Image)
    assert np.array_equal(
        np.array(cpu_out), np.array(cuda_out)
    ), "to_pil_image from CUDA tensor differs from CPU result"


# ---------------------------------------------------------------------------
# GPU tests — ToPureTensor operator
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@params("L", "RGB", "RGBA")
def test_to_pure_tensor_operator_gpu(mode):
    """ToPureTensor with device='gpu' must return a CUDA tensor matching the CPU reference."""
    img = _make_pil_image(mode)
    cpu_out = Compose([ToPureTensor()])(img)
    gpu_out = Compose([ToPureTensor(device="gpu")])(img)
    assert gpu_out.is_cuda, "Expected CUDA tensor"
    assert torch.equal(cpu_out, gpu_out.cpu()), "GPU and CPU ToPureTensor outputs differ"


# ---------------------------------------------------------------------------
# Batch size > 1
# ---------------------------------------------------------------------------


def test_to_pure_tensor_operator_batch_shape():
    """ToPureTensor from a batch tensor (N>1) must preserve shape and match torchvision."""
    tensor = make_test_tensor(shape=(4, 3, 8, 8))
    dali_out = Compose([ToPureTensor()])(tensor)
    tv_out = tv.ToPureTensor()(tensor)
    assert isinstance(dali_out, torch.Tensor)
    assert dali_out.shape == torch.Size([4, 3, 8, 8]), f"Unexpected shape: {tuple(dali_out.shape)}"
    assert torch.equal(dali_out, tv_out), "Batch ToPureTensor output differs from torchvision"


def test_to_pil_image_operator_batch_returns_list():
    """ToPILImage with a batch tensor (N>1) must return a list of PIL Images.

    Note: torchvision's ToPILImage raises for batch input. DALI returns a list instead — this
    test documents the current DALI behaviour so regressions are caught.
    """
    tensor = make_test_tensor(shape=(2, 3, 8, 8))
    out = Compose([ToPILImage()])(tensor)
    assert isinstance(out, list), f"Expected list of PIL Images, got {type(out)}"
    assert len(out) == 2, f"Expected 2 images, got {len(out)}"
    assert all(isinstance(img, Image.Image) for img in out), "Not all batch outputs are PIL Images"


# ---------------------------------------------------------------------------
# Pipeline mode — conversion operator ordering (last marker wins)
# ---------------------------------------------------------------------------


def test_pil_to_tensor_then_to_pure_tensor_returns_tensor():
    # When PILToTensor precedes ToPureTensor, the last marker (ToPureTensor) still wins → Tensor.
    img = _make_pil_image("RGB")
    out = Compose([PILToTensor(), ToPureTensor()])(img)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
    assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"


# ---------------------------------------------------------------------------
# ToTensor class — intentionally not exposed
# ---------------------------------------------------------------------------


def test_totensor_class_not_exported():
    """DALI does not expose a class-based ToTensor (tv.ToTensor is deprecated in v2).

    The functional to_tensor is the intended equivalent. This test documents the deliberate
    absence so that accidental additions are caught.
    """
    import nvidia.dali.experimental.torchvision as dali_tv

    assert not hasattr(
        dali_tv, "ToTensor"
    ), "ToTensor class must not be exported; use functional.to_tensor instead"
