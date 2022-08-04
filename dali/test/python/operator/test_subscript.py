# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import numpy as np
from nvidia.dali.pipeline import Pipeline
from nose_utils import assert_raises


@dali.pipeline_def(batch_size=2, num_threads=3, device_id=0)
def index_pipe(data_source, indexing_func):
    src = data_source

    cpu = indexing_func(src)
    gpu = indexing_func(src.gpu())

    return src, cpu, gpu


def test_plain_indexing():
    data = [np.float32([[0, 1, 2], [3, 4, 5]]), np.float32([[0, 1], [2, 3], [4, 5]])]
    src = fn.external_source(lambda: data, layout="AB")
    pipe = index_pipe(src, lambda x: x[1, 1])
    pipe.build()
    inp, cpu, gpu = pipe.run()
    for i in range(len(inp)):
        x = inp.at(i)
        assert np.array_equal(x[1, 1], cpu.at(i))
        assert np.array_equal(x[1, 1], gpu.as_cpu().at(i))


def _test_indexing(data_gen, input_layout, output_layout, dali_index_func, ref_index_func=None):
    src = fn.external_source(data_gen, layout=input_layout)
    pipe = index_pipe(src, dali_index_func)
    pipe.build()
    inp, cpu, gpu = pipe.run()
    for i in range(len(inp)):
        x = inp.at(i)
        ref = (ref_index_func or dali_index_func)(x)
        assert np.array_equal(ref, cpu.at(i))
        assert np.array_equal(ref, gpu.as_cpu().at(i))
        assert cpu.layout() == output_layout
        assert gpu.layout() == output_layout


def test_constant_ranges():
    def data_gen():
        return [np.float32([[0, 1, 2], [3, 4, 5]]), np.float32([[0, 1], [2, 3], [4, 5]])]
    yield _test_indexing, data_gen, "AB", "AB", lambda x: x[1:, :2], None
    yield _test_indexing, data_gen, "AB", "AB", lambda x: x[-1:, :-2], None
    yield _test_indexing, data_gen, "AB", "AB", lambda x: x[:-1, :-1], None
    yield _test_indexing, data_gen, "AB", "B", lambda x: x[1, :2], None
    yield _test_indexing, data_gen, "AB", "B", lambda x: x[1, :-2], None
    yield _test_indexing, data_gen, "AB", "A", lambda x: x[:-1, -1], None
    yield _test_indexing, data_gen, "AB", "A", lambda x: x[:-1, 0], None


def test_swapped_ends():
    data = [np.uint8([1, 2, 3]), np.uint8([1, 2])]
    src = fn.external_source(lambda: data)
    pipe = index_pipe(src, lambda x: x[2:1])
    pipe.build()
    inp, cpu, gpu = pipe.run()
    for i in range(len(inp)):
        x = inp.at(i)
        assert np.array_equal(x[2:1], cpu.at(i))
        assert np.array_equal(x[2:1], gpu.as_cpu().at(i))


def test_noop():
    node = dali.types.Constant(np.float32([1, 2, 2]))
    indexed = node[:]
    assert "SubscriptDimCheck" in indexed.name


def test_runtime_indexing():
    def data_gen():
        return [np.float32([[0, 1, 2], [3, 4, 5]]), np.float32([[0, 1], [2, 3], [4, 5]])]
    src = fn.external_source(data_gen)
    lo_idxs = [np.array(x, dtype=np.int64) for x in [1, -5, 0, 2, -2, 1]]
    hi_idxs = [np.array(x, dtype=np.int16) for x in [5, -1, 1, 2, 4]]
    lo0 = fn.external_source(source=lo_idxs, batch=False, cycle=True)
    hi1 = fn.external_source(source=hi_idxs, batch=False, cycle=True)
    pipe = index_pipe(src, lambda x: x[lo0:, :hi1])
    pipe.build()
    j = 0
    k = 0
    for _ in range(4):
        inp, cpu, gpu = pipe.run()
        for i in range(len(inp)):
            x = inp.at(i)
            ref = x[lo_idxs[j]:, :hi_idxs[k]]
            j = (j + 1) % len(lo_idxs)
            k = (k + 1) % len(hi_idxs)
            assert np.array_equal(ref, cpu.at(i))
            assert np.array_equal(ref, gpu.as_cpu().at(i))


def test_new_axis():
    def data_gen():
        return [np.float32([[0, 1, 2], [3, 4, 5]]), np.float32([[0, 1], [2, 3], [4, 5]])]

    yield (_test_indexing, data_gen, "AB", "",
           lambda x: x[1:, dali.newaxis, :2],
           lambda x: x[1:, np.newaxis, :2])
    yield (_test_indexing, data_gen, "AB", "CAB",
           lambda x: x[dali.newaxis("C"), -1:, :-2],
           lambda x: x[np.newaxis, -1:, :-2])
    yield (_test_indexing, data_gen, "AB", "ACB",
           lambda x: x[:, dali.newaxis("C"), :],
           lambda x: x[:, np.newaxis, :])
    yield (_test_indexing, data_gen, "AB", "C",
           lambda x: x[1, dali.newaxis("C"), 1],
           lambda x: x[1, np.newaxis, 1])


def _test_invalid_args(device, args, message, run):
    data = [np.uint8([[1, 2, 3]]), np.uint8([[1, 2]])]
    pipe = Pipeline(2, 1, 0)
    src = fn.external_source(lambda: data, device=device)
    pipe.set_outputs(fn.tensor_subscript(src, **args))
    with assert_raises(RuntimeError, glob=message):
        pipe.build()
        if run:
            pipe.run()


def test_inconsistent_args():
    for device in ["cpu", "gpu"]:
        for args, message in [
                    ({"lo_0": 0, "at_0": 0}, "both as an index"),
                    ({"at_0": 0, "step_0": 1}, "cannot have a step")
                ]:
            yield _test_invalid_args, device, args, message, False


def test_unsupported_step():
    for device in ["cpu", "gpu"]:
        for args in [
                    {"step_0": 2},
                    {"step_1": -1},
                ]:
            yield _test_invalid_args, device, args, "not implemented", True


def _test_out_of_range(device, idx):
    data = [np.uint8([1, 2, 3]), np.uint8([1, 2])]
    src = fn.external_source(lambda: data, device=device)
    pipe = index_pipe(src, lambda x: x[idx])
    pipe.build()
    with assert_raises(RuntimeError, glob="out of range"):
        _ = pipe.run()


def test_out_of_range():
    for device in ["cpu", "gpu"]:
        for idx in [-3, 2]:
            yield _test_out_of_range, device, idx


def _test_too_many_indices(device):
    data = [np.uint8([1, 2, 3]), np.uint8([1, 2])]
    src = fn.external_source(lambda: data, device=device)
    pipe = index_pipe(src, lambda x: x[1, :])

    # Verified by tensor_subscript
    with assert_raises(RuntimeError, glob="Too many indices"):
        pipe.build()
        _ = pipe.run()

    # Verified by subscript_dim_check
    pipe = index_pipe(src, lambda x: x[:, :])
    with assert_raises(RuntimeError, glob="Too many indices"):
        pipe.build()
        _ = pipe.run()

    # Verified by expand_dims
    pipe = index_pipe(src, lambda x: x[:, :, dali.newaxis])
    with assert_raises(RuntimeError, glob="not enough dimensions"):
        pipe.build()
        _ = pipe.run()

    # Verified by subscript_dim_check
    pipe = index_pipe(src, lambda x: x[dali.newaxis, :, dali.newaxis, :])
    with assert_raises(RuntimeError, glob="Too many indices"):
        pipe.build()
        _ = pipe.run()


def test_too_many_indices():
    for device in ["cpu", "gpu"]:
        yield _test_too_many_indices, device


def test_stride_not_implemented():
    data = [np.uint8([1, 2, 3]), np.uint8([1, 2])]
    src = fn.external_source(lambda: data)
    src[::1]
    with assert_raises(NotImplementedError):
        src[::2]


def test_ellipsis_not_implemented():
    data = [np.uint8([1, 2, 3]), np.uint8([1, 2])]
    src = fn.external_source(lambda: data)
    with assert_raises(NotImplementedError):
        src[..., :1]
