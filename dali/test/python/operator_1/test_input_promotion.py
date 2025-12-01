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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import numpy as np


def test_cat_numpy_array():
    pipe = dali.pipeline.Pipeline(1, 1, None)
    src = fn.external_source([[np.array([[10, 11], [12, 13]], dtype=np.float32)]])
    pipe.set_outputs(fn.cat(src, np.array([[20], [21]], dtype=np.float32), axis=1))
    o = pipe.run()
    assert np.array_equal(o[0].at(0), np.array([[10, 11, 20], [12, 13, 21]]))


def test_stack_numpy_scalar():
    pipe = dali.pipeline.Pipeline(1, 1, None)
    src = fn.external_source([[np.array([[10, 11], [12, 13]], dtype=np.float32)]])
    pipe.set_outputs(fn.cat(src, np.array([[20], [21]], dtype=np.float32), axis=1))
    o = pipe.run()
    assert np.array_equal(o[0].at(0), np.array([[10, 11, 20], [12, 13, 21]]))


def test_slice_fn():
    pipe = dali.pipeline.Pipeline(1, 1, 0)
    src = fn.external_source(
        [[np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32)]]
    )
    out_cpu = fn.slice(src, np.array([1, 1]), np.array([2, 1]), axes=[0, 1])
    out_gpu = fn.slice(src.gpu(), np.array([1, 1]), np.array([2, 1]), axes=[0, 1])
    pipe.set_outputs(out_cpu, out_gpu)
    out0, out1 = tuple(out.as_cpu() for out in pipe.run())
    assert np.array_equal(out0.at(0), np.array([[14], [17]]))
    assert np.array_equal(np.array(out1.at(0)), np.array([[14], [17]]))


def test_slice_ops():
    pipe = dali.pipeline.Pipeline(1, 1, 0)
    src = fn.external_source(
        [[np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32)]]
    )
    slice_cpu = dali.ops.Slice(axes=[0, 1], device="cpu")
    slice_gpu = dali.ops.Slice(axes=[0, 1], device="gpu")
    out_cpu = slice_cpu(src, np.array([1, 1]), np.array([2, 1]))
    out_gpu = slice_gpu(src.gpu(), np.array([1, 1]), np.array([2, 1]))
    pipe.set_outputs(out_cpu, out_gpu)
    out0, out1 = tuple(out.as_cpu() for out in pipe.run())
    assert np.array_equal(out0.at(0), np.array([[14], [17]]))
    assert np.array_equal(out1.at(0), np.array([[14], [17]]))


def test_python_function():
    pipe = dali.pipeline.Pipeline(3, 1, 0, exec_async=False, exec_pipelined=False)
    with pipe:

        def func(inp):
            ret = [x * x for x in inp]
            return ret

        out_cpu = fn.python_function(
            np.array([[1, 2], [3, 4]]), function=func, batch_processing=True
        )
        pipe.set_outputs(out_cpu)
    o = pipe.run()
    assert np.array_equal(o[0].at(0), np.array([[1, 4], [9, 16]]))


def test_arithm_ops():
    pipe = dali.pipeline.Pipeline(1, 1, None)
    with pipe:
        in1 = fn.external_source([[np.uint8([[1, 2], [3, 4]])]])
        pipe.set_outputs(
            in1 + np.array([[10, 20], [30, 40]]), in1 + np.array(5), in1 + np.uint8(100)
        )
    o = pipe.run()
    assert np.array_equal(o[0].at(0), np.array([[11, 22], [33, 44]]))
    assert np.array_equal(o[1].at(0), np.array([[6, 7], [8, 9]]))
    assert np.array_equal(o[2].at(0), np.array([[101, 102], [103, 104]]))


def test_arg_input():
    pipe = dali.pipeline.Pipeline(1, 1, None)
    with pipe:
        in1 = fn.external_source([[np.float32([[1, 2, 3], [4, 5, 6]])]])
        pipe.set_outputs(fn.transforms.translation(in1, offset=np.float32([10, 20])))
    o = pipe.run()
    assert np.array_equal(o[0].at(0), np.array([[1, 2, 13], [4, 5, 26]]))
