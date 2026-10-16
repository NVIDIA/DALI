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
(`nvidia.dali.ops._names.MERGED_INPUT_ARGS` et al.): WarpAffine's `mtx` positional input is
hidden and merged into its `matrix` argument. Arguments stay CPU-only, so the merge must not turn a
GPU value given as `matrix=` into a GPU input (DALI-4638, "Remove redundant GPU inputs that could
be arguments", was cancelled): a GPU matrix has to be passed positionally, as before the merge.
"""

import contextlib

import numpy as np
import nvidia.dali.backend as _backend
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
from ndd_utils import eval_modes
from nose_utils import SkipTest
from nvidia.dali.experimental.dynamic._ops import Operator


def _skip_if_no_gpu():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU needed for the test")


_MATRIX = np.array([[0.9, 0.1, 1.5], [-0.1, 0.9, -2.0]], dtype=np.float32)


def _image(device="cpu"):
    rng = np.random.default_rng(1234)
    data = rng.integers(0, 256, size=(16, 20, 3), dtype=np.uint8)
    return ndd.tensor(data, device=device, layout="HWC")


def _matrix(device):
    return ndd.tensor(_MATRIX, device=device)


def _to_numpy(x):
    if isinstance(x, ndd.Batch):
        x = ndd.as_tensor(x)
    return np.array(x.cpu())


@contextlib.contextmanager
def _spy_process_params():
    """Record the (cls_name, raw_args, raw_kwargs) each `Operator._process_params` call
    receives. The runtime can insert its own internal `Copy` calls around the op being tested,
    so callers must pick the call by class name (see `_warp_affine_call`) instead of assuming
    the op's own call is the last one recorded."""
    calls = []
    orig = Operator.__dict__["_process_params"]

    def spy(cls, backend, op_device, batch_size, *raw_args, **raw_kwargs):
        calls.append((cls.__name__, raw_args, dict(raw_kwargs)))
        return orig.__func__(cls, backend, op_device, batch_size, *raw_args, **raw_kwargs)

    Operator._process_params = classmethod(spy)
    try:
        yield calls
    finally:
        Operator._process_params = orig


def _warp_affine_call(calls):
    """Pick the recorded `_process_params` call that belongs to WarpAffine itself, out of the
    calls captured by `_spy_process_params` (which also sees internal `Copy` calls the runtime
    may insert)."""
    matches = [
        (raw_args, raw_kwargs)
        for cls_name, raw_args, raw_kwargs in calls
        if cls_name == "WarpAffine"
    ]
    assert len(matches) == 1, calls
    return matches[0]


# --- fn-style (`ndd.warp_affine`) ---


@eval_modes()
def test_warp_affine_fn_positional_matrix_still_works():
    """Backward-compatible positional call for the merged `mtx` input must keep working."""
    out = ndd.warp_affine(_image(), _matrix("cpu"))
    assert out.device.device_type == "cpu"
    np.testing.assert_array_equal(
        _to_numpy(out), _to_numpy(ndd.warp_affine(_image(), matrix=_matrix("cpu")))
    )


@eval_modes()
def test_warp_affine_fn_cpu_matrix_stays_an_argument():
    """A CPU-placed `matrix=` is not routed: it keeps going through the argument path, and the
    backend is still inferred from the (CPU) image alone."""
    matrix = _matrix("cpu")
    with _spy_process_params() as calls:
        out = ndd.warp_affine(_image(), matrix=matrix)
    assert out.device.device_type == "cpu"
    raw_args, raw_kwargs = _warp_affine_call(calls)
    assert raw_kwargs.get("matrix") is matrix
    assert not any(a is matrix for a in raw_args)


# --- class-based (`ndd._ops.WarpAffine`) ---


@eval_modes()
def test_warp_affine_ops_class_cpu_matrix_stays_an_argument():
    matrix = _matrix("cpu")
    op = ndd._ops.WarpAffine()
    with _spy_process_params() as calls:
        out = op(_image(), matrix=matrix)
    assert out.device.device_type == "cpu"
    raw_args, raw_kwargs = _warp_affine_call(calls)
    assert raw_kwargs.get("matrix") is matrix
    assert not any(a is matrix for a in raw_args)


@eval_modes()
def test_warp_affine_gpu_matrix_kwarg_is_not_routed_to_an_input():
    """DALI-4638 (GPU inputs turned into arguments) is not implemented: a GPU `matrix=` is not
    moved into the positional inputs, so it does not move the op to the GPU. It stays an
    argument, i.e. it is copied to the host and the CPU image keeps the op on the CPU."""
    _skip_if_no_gpu()
    matrix = _matrix("gpu")
    with _spy_process_params() as calls:
        out = ndd.warp_affine(_image("cpu"), matrix=matrix)
    assert out.device.device_type == "cpu"
    raw_args, raw_kwargs = _warp_affine_call(calls)
    assert raw_kwargs.get("matrix") is matrix
    assert not any(a is matrix for a in raw_args)
    np.testing.assert_array_equal(
        _to_numpy(out), _to_numpy(ndd.warp_affine(_image("cpu"), matrix=_matrix("cpu")))
    )


@eval_modes()
def test_warp_affine_ops_class_gpu_matrix_kwarg_is_not_routed_to_an_input():
    _skip_if_no_gpu()
    matrix = _matrix("gpu")
    op = ndd._ops.WarpAffine()
    with _spy_process_params() as calls:
        out = op(_image("cpu"), matrix=matrix)
    assert out.device.device_type == "cpu"
    raw_args, raw_kwargs = _warp_affine_call(calls)
    assert raw_kwargs.get("matrix") is matrix
    assert not any(a is matrix for a in raw_args)


@eval_modes()
def test_warp_affine_gpu_matrix_positional_input_still_works():
    """The way to place the matrix on the GPU is the positional input, as before the merge."""
    _skip_if_no_gpu()
    image = _image("gpu")
    out = ndd.warp_affine(image, _matrix("gpu"))
    assert out.device.device_type == "gpu"
    np.testing.assert_array_equal(
        _to_numpy(out), _to_numpy(ndd.warp_affine(image, matrix=_matrix("cpu")))
    )


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
