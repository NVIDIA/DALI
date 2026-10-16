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

"""Regression tests for the dynamic (ndd) API's "uniform inputs and arguments" merge
(`nvidia.dali.ops._names.MERGED_INPUT_ARGS` et al.): positional inputs that duplicate a named
argument are hidden from the ndd signature, but are still accepted positionally.
"""

import os

import numpy as np
import nvidia.dali.backend as _backend
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
from ndd_utils import eval_modes
from nose_utils import SkipTest
from test_utils import get_dali_extra_path


def _skip_if_no_gpu():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU needed for the test")


def _to_numpy(x):
    if isinstance(x, ndd.Batch):
        x = ndd.as_tensor(x)
    return np.array(x.cpu())


# --- Reshape / Reinterpret: the hidden `shape_input` is still accepted positionally ---


@eval_modes()
@params(ndd.reshape, ndd.reinterpret)
def test_reshape_like_positional_shape_still_works(op):
    data = ndd.tensor(np.arange(24, dtype=np.float32))
    shape = ndd.tensor(np.array([2, 12], dtype=np.int32))
    positional = op(data, shape)
    named = op(data, shape=shape)
    assert tuple(positional.shape) == (2, 12)
    np.testing.assert_array_equal(_to_numpy(positional), _to_numpy(named))


# --- Slice: positional inputs are interpreted by dtype, named arguments by name ---


def _slice_image(device="cpu"):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
    return ndd.tensor(data, device=device, layout="HWC"), data


def _t(values, dtype, device="cpu"):
    return ndd.tensor(np.array(values, dtype=dtype), device=device)


@eval_modes()
@params("cpu", "gpu")
def test_slice_positional_integers_are_absolute(device):
    if device == "gpu":
        _skip_if_no_gpu()
    image, data = _slice_image(device)
    for arg_device in ("cpu", "gpu") if device == "gpu" else ("cpu",):
        out = ndd.slice(
            image,
            _t([5, 10], np.int32, arg_device),
            _t([20, 30], np.int32, arg_device),
            axes=[0, 1],
        )
        np.testing.assert_array_equal(_to_numpy(out), data[5:25, 10:40])


@eval_modes()
def test_slice_positional_floats_follow_normalized_flags():
    image, data = _slice_image()
    # default: floats are relative (the operator's default is normalized_anchor/shape=True)
    rel = ndd.slice(image, _t([0.25, 0.5], np.float32), _t([0.5, 0.25], np.float32), axes=[0, 1])
    np.testing.assert_array_equal(_to_numpy(rel), data[10:30, 30:45])
    # the same dtype is absolute once the flags say so
    absolute = ndd.slice(
        image,
        _t([5, 10], np.float32),
        _t([20, 30], np.float32),
        axes=[0, 1],
        normalized_anchor=False,
        normalized_shape=False,
    )
    np.testing.assert_array_equal(_to_numpy(absolute), data[5:25, 10:40])


@eval_modes()
def test_slice_named_arguments_ignore_the_dtype():
    """The named arguments say absolute or relative by their name, whatever their dtype."""
    image, data = _slice_image()
    for dtype in (np.int32, np.float32):
        out = ndd.slice(image, start=_t([5, 10], dtype), shape=_t([20, 30], dtype), axes=[0, 1])
        np.testing.assert_array_equal(_to_numpy(out), data[5:25, 10:40])
    out = ndd.slice(
        image,
        rel_start=_t([0.25, 0.5], np.float32),
        rel_shape=_t([0.5, 0.25], np.float32),
        axes=[0, 1],
    )
    np.testing.assert_array_equal(_to_numpy(out), data[10:30, 30:45])


@eval_modes()
@params(np.int32, np.int64, np.float32, np.float64)
def test_slice_gpu_named_arguments_are_copied_to_the_host(dtype):
    """GPU values given as named arguments are accepted: they are converted to the argument's
    dtype and copied to the host, and the operator stays on the image's backend."""
    _skip_if_no_gpu()
    image, data = _slice_image("gpu")
    out = ndd.slice(
        image,
        start=_t([5, 10], dtype, "gpu"),
        shape=_t([20, 30], dtype, "gpu"),
        axes=[0, 1],
    )
    assert out.device.device_type == "gpu"
    np.testing.assert_array_equal(_to_numpy(out), data[5:25, 10:40])


# --- decoders.image_slice: same inputs and arguments as Slice ---


def _encoded_image():
    path = os.path.join(
        get_dali_extra_path(), "db", "single", "jpeg", "100", "swan-3584559_640.jpg"
    )
    return ndd.tensor(np.fromfile(path, dtype=np.uint8))


@eval_modes()
@params(
    *(
        (op, device)
        for op in (ndd.decoders.image_slice, ndd.experimental.decoders.image_slice)
        for device in ("cpu", "gpu")
    )
)
def test_decoder_image_slice_positional_and_named_forms_match(op, device):
    if device == "gpu":
        _skip_if_no_gpu()
    encoded = _encoded_image()
    full = _to_numpy(ndd.decoders.image(encoded, device=device))
    expected = full[20:120, 30:230]
    anchor = _t([20, 30], np.int32)
    shape = _t([100, 200], np.int32)
    positional = op(encoded, anchor, shape, axes=[0, 1], device=device)
    named = op(encoded, start=anchor, shape=shape, axes=[0, 1], device=device)
    np.testing.assert_array_equal(_to_numpy(positional), expected)
    np.testing.assert_array_equal(_to_numpy(named), expected)
