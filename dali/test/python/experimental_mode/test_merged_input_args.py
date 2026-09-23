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
hidden and merged into its `matrix` argument, but a GPU-placed value given for `matrix` must
still be routed like a GPU positional input would be, since arguments are always CPU-only.
"""

import contextlib
import os

import numpy as np
import nvidia.dali.backend as _backend
import nvidia.dali.experimental.dynamic as ndd
from ndd_utils import eval_modes
from nose_utils import SkipTest
from nvidia.dali.experimental.dynamic._ops import Operator
from test_utils import get_dali_extra_path


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
    """Record the (raw_args, raw_kwargs) each `Operator._process_params` call receives."""
    calls = []
    orig = Operator.__dict__["_process_params"]

    def spy(cls, backend, op_device, batch_size, *raw_args, **raw_kwargs):
        calls.append((raw_args, dict(raw_kwargs)))
        return orig.__func__(cls, backend, op_device, batch_size, *raw_args, **raw_kwargs)

    Operator._process_params = classmethod(spy)
    try:
        yield calls
    finally:
        Operator._process_params = orig


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
    raw_args, raw_kwargs = calls[-1]
    assert raw_kwargs.get("matrix") is matrix
    assert not any(a is matrix for a in raw_args)


@eval_modes()
def test_warp_affine_fn_gpu_matrix_routes_backend_to_gpu():
    """A GPU-placed `matrix=` kwarg must be visible to backend resolution the same way a GPU
    `mtx` positional input would be - otherwise the op silently runs on CPU instead of the
    documented GPU placement. Regression test for the routing running too late relative to
    `_capture_intercept`'s backend resolution."""
    _skip_if_no_gpu()
    matrix = _matrix("gpu")
    with _spy_process_params() as calls:
        out = ndd.warp_affine(_image("cpu"), matrix=matrix)
    assert out.device.device_type == "gpu"
    raw_args, raw_kwargs = calls[-1]
    assert "matrix" not in raw_kwargs
    assert raw_args[-1] is matrix


@eval_modes()
def test_warp_affine_fn_gpu_matrix_matches_other_forms():
    """Routing must not change the result: a GPU `matrix=` gives exactly what the same GPU
    kernel gives with the matrix as a CPU argument or as a GPU positional input."""
    _skip_if_no_gpu()
    image = _image("gpu")
    routed = ndd.warp_affine(image, matrix=_matrix("gpu"))
    as_argument = ndd.warp_affine(image, matrix=_matrix("cpu"))
    as_positional = ndd.warp_affine(image, _matrix("gpu"))
    for out in (routed, as_argument, as_positional):
        assert out.device.device_type == "gpu"
    np.testing.assert_array_equal(_to_numpy(routed), _to_numpy(as_argument))
    np.testing.assert_array_equal(_to_numpy(routed), _to_numpy(as_positional))


@eval_modes()
def test_warp_affine_fn_gpu_matrix_batch():
    """Same as above, with `Batch` image and `Batch` matrix (per-sample matrices)."""
    _skip_if_no_gpu()
    matrices = [_MATRIX, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)]
    images = ndd.batch([_image("cpu"), _image("cpu")], layout="HWC")
    images_gpu = ndd.batch([_image("gpu"), _image("gpu")], layout="HWC")
    routed = ndd.warp_affine(images, matrix=ndd.batch(matrices, device="gpu"))
    as_argument = ndd.warp_affine(images_gpu, matrix=ndd.batch(matrices))
    assert isinstance(routed, ndd.Batch)
    assert routed.device.device_type == "gpu"
    np.testing.assert_array_equal(_to_numpy(routed), _to_numpy(as_argument))


@eval_modes()
def test_warp_affine_fn_gpu_matrix_per_frame():
    """A GPU per-frame matrix given via `matrix=` must be routed too, and match the CPU-argument
    form frame by frame."""
    _skip_if_no_gpu()
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, size=(2, 16, 20, 3), dtype=np.uint8)
    sequence = ndd.tensor(frames, layout="FHWC")
    sequence_gpu = ndd.tensor(frames, layout="FHWC", device="gpu")
    per_frame = np.stack([_MATRIX, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)])
    routed = ndd.warp_affine(sequence, matrix=ndd.per_frame(ndd.tensor(per_frame, device="gpu")))
    as_argument = ndd.warp_affine(sequence_gpu, matrix=ndd.per_frame(ndd.tensor(per_frame)))
    assert routed.device.device_type == "gpu"
    np.testing.assert_array_equal(_to_numpy(routed), _to_numpy(as_argument))


# --- class-based (`ndd._ops.WarpAffine`) ---


@eval_modes()
def test_warp_affine_ops_class_gpu_matrix_routes_to_input():
    """The class-based `__call__` must route a GPU-placed `matrix=` the same way the fn-style
    wrapper does, instead of letting `_process_params` force it through the CPU-only argument
    path. Regression test for `build_call_function`'s `call()` missing the GPU-routing step."""
    _skip_if_no_gpu()
    image = _image("gpu")
    matrix = _matrix("gpu")
    op = ndd._ops.WarpAffine(device="gpu")
    with _spy_process_params() as calls:
        out = op(image, matrix=matrix)
    assert out.device.device_type == "gpu"
    raw_args, raw_kwargs = calls[-1]
    assert "matrix" not in raw_kwargs
    assert raw_args[-1] is matrix
    np.testing.assert_array_equal(_to_numpy(out), _to_numpy(op(image, matrix=_matrix("cpu"))))


@eval_modes()
def test_warp_affine_ops_class_cpu_matrix_stays_an_argument():
    matrix = _matrix("cpu")
    op = ndd._ops.WarpAffine()
    with _spy_process_params() as calls:
        out = op(_image(), matrix=matrix)
    assert out.device.device_type == "cpu"
    raw_args, raw_kwargs = calls[-1]
    assert raw_kwargs.get("matrix") is matrix
    assert not any(a is matrix for a in raw_args)


# --- capture mode ---


def test_warp_affine_gpu_matrix_capture_parity():
    """Capture mode must give the same (GPU) results as eager mode for a GPU `matrix=`. The
    backend-resolution fix must not disturb capture classification, which maps the call's
    inputs/kwargs back to its source by position/name."""
    _skip_if_no_gpu()
    images_root = os.path.join(get_dali_extra_path(), "db", "single", "jpeg")
    matrix = _matrix("gpu")

    def run(capture):
        reader = ndd.readers.File(file_root=images_root, pad_last_batch=True)
        results = []
        for _ in range(2):  # the first capture-mode epoch traces, the second one replays
            for jpegs, _ in reader.next_epoch(batch_size=4, capture=capture):
                images = ndd.decoders.image(jpegs)
                images = ndd.resize(images, size=[32, 32])
                out = ndd.warp_affine(images, matrix=matrix)
                assert out.device.device_type == "gpu"
                results.append(_to_numpy(out))
        return results

    eager = run(capture=False)
    captured = run(capture=True)
    assert len(eager) == len(captured)
    for e, c in zip(eager, captured):
        np.testing.assert_array_equal(e, c)
