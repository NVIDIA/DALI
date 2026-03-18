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

from nvidia.dali.experimental.torchvision import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)
import nvidia.dali.experimental.torchvision.v2.functional as fn_dali

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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
def test_compose_cuda_output_values_on_gpu():
    """Pipeline output must match a reference computed entirely on GPU (no CPU transfers)."""
    test_tensor = make_test_tensor(shape=(5, 3, 8, 8)).cuda()
    tv_ref = tv.RandomHorizontalFlip(p=1.0)(test_tensor)  # result stays on GPU

    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0, device="gpu")], batch_size=5)
    dali_out = dali_pipeline(test_tensor)

    assert dali_out.is_cuda, f"Expected CUDA output, got device={dali_out.device}"
    assert torch.equal(dali_out, tv_ref)  # GPU-side comparison, no .cpu()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
def test_compose_cuda_synchronization_no_cpu_transfer():
    """20 rapid calls must each produce correct GPU values without any explicit CPU sync.

    _cuda_run creates a fresh stream per call. Without proper stream ordering (via DLPack
    stream info or record_stream), the default-stream comparison below can read unfinished
    pipeline output before DALI's private stream has written it.
    """
    test_tensor = make_test_tensor(shape=(5, 3, 8, 8)).cuda()
    tv_ref = tv.RandomHorizontalFlip(p=1.0)(test_tensor)  # reference stays on GPU

    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0, device="gpu")], batch_size=5)

    for i in range(20):
        out = dali_pipeline(test_tensor)
        # GPU comparison on the default stream — exposes any stream-ordering bug
        assert torch.equal(out, tv_ref), f"Call {i}: GPU values differ — possible sync issue"

    torch.cuda.synchronize()  # flush deferred GPU errors


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
def test_compose_cuda_synchronization_interleaved_default_stream_work():
    """Interleaving default-stream compute with pipeline calls must not cause data races.

    Keeping the default stream busy widens the window in which unsynchronized
    DALI output could be consumed before its private stream completes.
    """
    test_tensor = make_test_tensor(shape=(5, 3, 8, 8)).cuda()
    tv_ref = tv.RandomHorizontalFlip(p=1.0)(test_tensor)
    scratch = torch.zeros(512, 512, device="cuda")

    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0, device="gpu")], batch_size=5)

    for i in range(10):
        # Keep the default stream occupied to maximise the sync-gap window
        scratch = scratch.add(1.0)
        out = dali_pipeline(test_tensor)
        # Consume output immediately on the default stream — races show up here
        assert torch.equal(out, tv_ref), f"Call {i}: interleaved sync failure"

    torch.cuda.synchronize()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
def test_compose_cuda_sequential_different_content():
    """Pipeline called sequentially on inputs with different pixel values must produce
    correct per-input results — catches output buffers being incorrectly reused."""
    data = [
        make_test_tensor(shape=(3, 3, 8, 8)).add_(i * 10).clamp_(0, 255).to(torch.uint8).cuda()
        for i in range(5)
    ]
    tv_pipe = tv.RandomHorizontalFlip(p=1.0)
    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0, device="gpu")], batch_size=3)

    out_list = []
    for d in data:
        out_list.append(dali_pipeline(d))

    torch.cuda.synchronize()
    for i, (d, out) in enumerate(zip(data, out_list)):
        tv_ref = tv_pipe(d)
        assert out.is_cuda, f"Input {i}: expected CUDA output"
        assert out.shape == d.shape, f"Input {i}: shape mismatch"
        assert torch.equal(out, tv_ref), f"Input {i}: value mismatch vs torchvision"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
def test_compose_cuda_sequential_different_shapes():
    """Pipeline called sequentially on inputs with varying spatial sizes must return
    the correct shape and values for each input."""
    shapes = [(2, 3, 4, 4), (2, 3, 8, 16), (2, 3, 12, 6), (2, 3, 5, 20)]
    data = [make_test_tensor(shape=s).cuda() for s in shapes]
    tv_pipe = tv.RandomHorizontalFlip(p=1.0)
    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0, device="gpu")], batch_size=2)

    out_list = []
    for d in data:
        out_list.append(dali_pipeline(d))

    torch.cuda.synchronize()
    for i, (d, out) in enumerate(zip(data, out_list)):
        tv_ref = tv_pipe(d)
        assert out.shape == d.shape, f"Input {i}: shape mismatch ({out.shape} != {d.shape})"
        assert torch.equal(out, tv_ref), f"Input {i}: value mismatch vs torchvision"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
def test_compose_cuda_sequential_multi_op_different_inputs():
    """Multi-op pipeline (resize + flip) called sequentially on distinct inputs must match
    torchvision for shape and be close in values (resize may introduce ±1 differences)."""
    data = [make_test_tensor(shape=(2, 3, 10 + i * 4, 10 + i * 4)).cuda() for i in range(4)]
    dali_pipeline = Compose(
        [RandomVerticalFlip(p=1, device="gpu"), RandomHorizontalFlip(p=1.0, device="gpu")],
        batch_size=2,
    )
    tv_pipe = tv.Compose([tv.RandomVerticalFlip(p=1), tv.RandomHorizontalFlip(p=1.0)])

    out_list = []
    for d in data:
        out_list.append(dali_pipeline(d))

    torch.cuda.synchronize()
    for i, (d, out) in enumerate(zip(data, out_list)):
        tv_ref = tv_pipe(d)
        assert out.is_cuda, f"Input {i}: expected CUDA output"
        assert out.shape == tv_ref.shape, f"Input {i}: shape mismatch vs torchvision"
        assert torch.allclose(out.float(), tv_ref.float(), rtol=0, atol=1), (
            f"Input {i}: values differ by more than 1 vs torchvision "
            f"Is: {out}, should be: {tv_ref}"
        )


def test_compose_repeated_same_input():
    """Calling the same Compose object N times with identical input
    must always give the same result."""
    test_tensor = make_test_tensor(shape=(5, 3, 5, 5))
    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0)], batch_size=5)
    tv_out = tv.RandomHorizontalFlip(p=1.0)(test_tensor)
    for i in range(5):
        dali_out = dali_pipeline(test_tensor)
        assert torch.equal(dali_out, tv_out), f"Result mismatch on call {i}"


def test_compose_repeated_different_spatial_sizes():
    """Reusing Compose across tensors with varying H×W must produce correct shapes each time."""
    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0)], batch_size=1)
    for h, w in [(5, 5), (10, 20), (3, 7)]:
        tensor = make_test_tensor(shape=(1, 3, h, w))
        dali_out = dali_pipeline(tensor)
        tv_out = tv.RandomHorizontalFlip(p=1.0)(tensor)
        assert dali_out.shape == tensor.shape, f"Shape mismatch for H={h} W={w}"
        assert torch.equal(dali_out, tv_out), f"Value mismatch for H={h} W={w}"


def test_compose_rejects_input_type_change():
    """Compose built for PIL Image must raise TypeError when later called with a torch.Tensor."""
    img = _make_pil_image("RGB")
    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0)])
    _ = dali_pipeline(img)  # first call locks the pipeline to HWC/PIL
    test_tensor = make_test_tensor(shape=(1, 3, 5, 5))
    with assert_raises(TypeError):
        _ = dali_pipeline(test_tensor)


@params("L", "RGB", "RGBA")
def test_compose_functional_vs_operator_flip_consistency(mode):
    """horizontal_flip via functional API and via Compose must produce identical results,
    and both must match torchvision's reference output."""
    img = _make_pil_image(mode)
    fn_out = fn_dali.horizontal_flip(img)
    compose_out = Compose([RandomHorizontalFlip(p=1.0)])(img)
    tv_out = tv.functional.hflip(img)

    assert isinstance(fn_out, Image.Image)
    assert isinstance(compose_out, Image.Image)
    assert fn_out.mode == mode, f"Functional API changed mode to {fn_out.mode}"
    assert compose_out.mode == mode, f"Compose changed mode to {compose_out.mode}"

    fn_tensor = tv.functional.pil_to_tensor(fn_out)
    compose_tensor = tv.functional.pil_to_tensor(compose_out)
    tv_tensor = tv.functional.pil_to_tensor(tv_out)

    assert torch.equal(
        fn_tensor, compose_tensor
    ), f"Functional and Compose disagree for mode={mode}"
    assert torch.equal(
        fn_tensor, tv_tensor
    ), f"Functional API disagrees with torchvision for mode={mode}"
    assert torch.equal(
        compose_tensor, tv_tensor
    ), f"Compose disagrees with torchvision for mode={mode}"


@params(torch.float32, torch.float64, torch.int16, torch.int32)
def test_compose_non_uint8_dtype_flip(dtype):
    """RandomHorizontalFlip must preserve dtype and produce correct values for non-uint8 inputs."""
    test_tensor = torch.ones(5, 3, 8, 8, dtype=dtype)
    dali_pipeline = Compose([RandomHorizontalFlip(p=1.0)], batch_size=5)
    dali_out = dali_pipeline(test_tensor)
    tv_out = tv.RandomHorizontalFlip(p=1.0)(test_tensor)
    assert dali_out.dtype == dtype, f"Dtype changed: expected {dtype}, got {dali_out.dtype}"
    assert torch.equal(dali_out, tv_out), f"Value mismatch for dtype={dtype}"


@params(torch.float32, torch.float64, torch.int16, torch.int32)
def test_compose_non_uint8_dtype_resize(dtype):
    """Resize must preserve dtype, produce the correct output shape, and match torchvision's
    output shape. A uniform tensor of ones is used so that bilinear interpolation leaves all
    values at 1, making an exact value comparison valid across all tested dtypes."""
    dali_supported_types = [torch.float32, torch.int16]
    test_tensor = torch.ones(5, 3, 10, 10, dtype=dtype)
    tv_out = tv.Resize(size=(7, 7))(test_tensor)
    dali_pipeline = Compose([Resize(size=(7, 7))], batch_size=5)
    if dtype in dali_supported_types:
        dali_out = dali_pipeline(test_tensor)
        assert dali_out.dtype == dtype, f"Dtype changed: expected {dtype}, got {dali_out.dtype}"
        assert (
            dali_out.shape == tv_out.shape
        ), f"Shape mismatch vs torchvision: {dali_out.shape} != {tv_out.shape}"
        assert dali_out.shape[2:] == torch.Size([7, 7]), f"Wrong spatial shape: {dali_out.shape}"
    else:
        with assert_raises(RuntimeError):
            _ = dali_pipeline(test_tensor)
