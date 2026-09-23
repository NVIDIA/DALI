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

import numpy as np
import nvidia.dali.backend as _backend
import nvidia.dali.experimental.dynamic as ndd
from ndd_utils import eval_modes
from nose_utils import SkipTest


def _skip_if_no_gpu():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 GPU needed for the test")


def _cpu_image():
    return ndd.tensor(np.zeros((4, 4, 3), dtype=np.uint8), device="cpu", layout="HWC")


def _identity_matrix(device):
    return ndd.tensor(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), device=device)


@eval_modes()
def test_warp_affine_fn_positional_matrix_still_works():
    """Backward-compatible positional call for the merged `mtx` input must keep working."""
    image = _cpu_image()
    matrix = _identity_matrix("cpu")
    out = ndd.warp_affine(image, matrix)
    assert out.device.device_type == "cpu"


@eval_modes()
def test_warp_affine_fn_gpu_matrix_routes_backend_to_gpu():
    """A GPU-placed `matrix=` kwarg must be visible to backend resolution the same way a GPU
    `mtx` positional input would be - otherwise the op silently runs on CPU instead of the
    documented GPU placement. Regression test for the routing running too late relative to
    `_capture_intercept`'s backend resolution."""
    _skip_if_no_gpu()
    image = _cpu_image()
    matrix = _identity_matrix("gpu")
    out = ndd.warp_affine(image, matrix=matrix)
    assert out.device.device_type == "gpu"


@eval_modes()
def test_warp_affine_ops_class_gpu_matrix_routes_to_input():
    """The class-based (`ndd._ops.WarpAffine`) `__call__` must route a GPU-placed `matrix=` the
    same way the `fn`-style wrapper does, instead of letting `_process_params` silently force it
    through the CPU-only argument-processing path. Regression test for `build_call_function`'s
    `call()` missing the GPU-routing step that `build_fn_wrapper`'s `fn_call` has."""
    _skip_if_no_gpu()
    from nvidia.dali.experimental.dynamic._ops import Operator

    captured = {}
    orig_process_params = Operator._process_params.__func__

    def spy(cls, backend, op_device, batch_size, *raw_args, **raw_kwargs):
        captured["raw_args"] = raw_args
        captured["raw_kwargs"] = raw_kwargs
        return orig_process_params(cls, backend, op_device, batch_size, *raw_args, **raw_kwargs)

    Operator._process_params = classmethod(spy)
    try:
        image = ndd.tensor(np.zeros((4, 4, 3), dtype=np.uint8), device="gpu", layout="HWC")
        matrix = _identity_matrix("gpu")
        op = ndd._ops.WarpAffine(device="gpu")
        out = op(image, matrix=matrix)
    finally:
        Operator._process_params = orig_process_params

    assert out.device.device_type == "gpu"
    assert "matrix" not in captured["raw_kwargs"], (
        "GPU-placed `matrix` should have been routed into a positional input instead of being "
        "left as a kwarg, which forces an unnecessary CPU copy in `_process_params`"
    )
    assert any(a is matrix for a in captured["raw_args"])
