# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nose_utils import attr

# it is enough to just import all functions from test_internals_operator_external_source
# nose will query for the methods available and will run them
# the test_internals_operator_external_source is 99% the same for cupy and numpy tests
# so it is better to store everything in one file and just call `use_cupy` to
# switch between the default numpy and cupy
from test_external_source_impl import *  # noqa:F403, F401
from test_external_source_impl import use_cupy
from test_utils import check_output, check_output_pattern
import nvidia.dali
from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.tensors import TensorGPU
import numpy as np


use_cupy()


# extra tests, GPU-specific
import cupy as cp  # noqa:E402  - we need to call this after use_cupy()


assert nvidia.dali.types._is_cupy_array(cp.array([1, 2, 3])), "CuPy array not recognized"


def test_external_source_with_iter_cupy_stream():
    with cp.cuda.Stream(non_blocking=True):
        for attempt in range(10):
            pipe = Pipeline(1, 3, 0)

            def get_data(i):
                return [cp.array([attempt * 100 + i * 10 + 1.5], dtype=cp.float32)]

            pipe.set_outputs(fn.external_source(get_data))

            for i in range(10):
                check_output(
                    pipe.run(), [np.array([attempt * 100 + i * 10 + 1.5], dtype=np.float32)]
                )


def test_external_source_mixed_contiguous():
    batch_size = 2
    iterations = 4

    def generator(i):
        if i % 2:
            return cp.array([[100 + i * 10 + 1.5]] * batch_size, dtype=cp.float32)
        else:
            return batch_size * [cp.array([100 + i * 10 + 1.5], dtype=cp.float32)]

    pipe = Pipeline(batch_size, 3, 0)

    pipe.set_outputs(fn.external_source(device="gpu", source=generator, no_copy=True))

    pattern = (
        "ExternalSource operator should not mix contiguous and noncontiguous inputs. "
        "In such a case the internal memory used to gather data in a contiguous chunk of "
        "memory would be trashed."
    )
    with check_output_pattern(pattern):
        for _ in range(iterations):
            pipe.run()


def _test_cross_device(src, dst, use_dali_tensor=False):
    # The use_dali_tensor converts (via the Dlpack) to the DALI native Tensor before feeding the
    # data, to additionally check if the constructor works correctly wrt to device_id.
    # TODO(klecki): [device_id] currently the device_id is not exposed in Python Tensors, so there
    # is no other way we may verify it.
    import nvidia.dali.fn as fn
    import numpy as np

    pipe = Pipeline(1, 3, dst)

    iter = 0

    with cp.cuda.Device(src):
        with cp.cuda.Stream(src):

            def get_data():
                nonlocal iter
                data = cp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=cp.float32) + iter
                iter += 1
                if use_dali_tensor:
                    return TensorGPU(data.toDlpack())
                return data

            with pipe:
                pipe.set_outputs(fn.external_source(get_data, batch=False, device="gpu"))

            for i in range(10):
                (out,) = pipe.run()
                assert np.array_equal(
                    np.array(out[0].as_cpu()), np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) + i
                )


@attr("multigpu")
def test_cross_device():
    if cp.cuda.runtime.getDeviceCount() > 1:
        for src in [0, 1]:
            for dst in [0, 1]:
                for use_dali_tensor in [True, False]:
                    yield _test_cross_device, src, dst, use_dali_tensor


def _test_memory_consumption(device, test_case):
    batch_size = 32
    num_iters = 128

    if device == "cpu":
        import numpy as np

        fw = np
    else:
        fw = cp

    def no_copy_sample():
        batch = [fw.full((1024, 1024, 4), i, dtype=fw.int32) for i in range(batch_size)]

        def cb(sample_info):
            return batch[sample_info.idx_in_batch]

        return cb

    def copy_sample():
        def cb(sample_info):
            return fw.full((1024, 1024, 4), sample_info.idx_in_batch, dtype=fw.int32)

        return cb

    def copy_batch():
        def cb():
            return fw.full((batch_size, 1024, 1024, 4), 42, dtype=fw.int32)

        return cb

    cases = {
        "no_copy_sample": (no_copy_sample, True, False),
        "copy_sample": (copy_sample, False, False),
        "copy_batch": (copy_batch, False, True),
    }

    cb, no_copy, batch_mode = cases[test_case]

    @pipeline_def
    def pipeline():
        return fn.external_source(source=cb(), device=device, batch=batch_mode, no_copy=no_copy)

    pipe = pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    for _ in range(num_iters):
        pipe.run()


def test_memory_consumption():
    for device in ["cpu", "gpu"]:
        for test_case in ["no_copy_sample", "copy_sample", "copy_batch"]:
            yield _test_memory_consumption, device, test_case
