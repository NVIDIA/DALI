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
import math
from test_utils import check_batch

np.random.seed(1234)


def input_generator(num_inputs, batch_size, ndim, variable_axis=None):
    if ndim <= 0:
        max_extent = 1
    else:
        max_extent = int(math.ceil(math.pow(1e6 / (batch_size * num_inputs), 1 / ndim)))

    def gen():
        if ndim == 0:
            inputs = []
            for i in range(num_inputs):
                inputs.append([np.float32(np.random.random()) for _ in range(batch_size)])
            return inputs

        inputs = [[] for _ in range(num_inputs)]
        for i in range(batch_size):
            base_shape = np.random.randint(1, max_extent, [ndim])
            for j in range(num_inputs):
                shape = list(base_shape)
                if variable_axis is not None:
                    shape[variable_axis] = np.random.randint(1, 10)
                inputs[j].append(np.random.random(shape).astype(np.float32))
        return inputs

    return gen


def test_cat_different_length():
    pipe = dali.pipeline.Pipeline(batch_size=1, num_threads=3, device_id=0)
    with pipe:
        src1 = dali.types.Constant(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
        src2 = dali.types.Constant(np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]]))
        out_cpu = fn.cat(src1, src2, axis=1)
        out_gpu = fn.cat(src1.gpu(), src2.gpu(), axis=1)
        pipe.set_outputs(out_cpu, out_gpu)

    o = pipe.run()

    o = list(o)
    o[1] = o[1].as_cpu()

    ref = np.array(
        [[1, 2, 3, 4, 13, 14, 15], [5, 6, 7, 8, 16, 17, 18], [9, 10, 11, 12, 19, 20, 21]]
    )
    assert np.array_equal(o[0].at(0), ref)
    assert np.array_equal(o[1].at(0), ref)


def test_cat_empty_input():
    pipe = dali.pipeline.Pipeline(batch_size=1, num_threads=3, device_id=0)
    with pipe:
        src1 = dali.types.Constant(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
        src2 = dali.types.Constant(np.array([[], [], []], dtype=np.int32))
        src3 = dali.types.Constant(np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]]))
        out_cpu = fn.cat(src1, src2, src3, axis=1)
        out_gpu = fn.cat(src1.gpu(), src2.gpu(), src3.gpu(), axis=1)
        pipe.set_outputs(out_cpu, out_gpu)

    o = pipe.run()

    o = list(o)
    o[1] = o[1].as_cpu()

    ref = np.array(
        [[1, 2, 3, 4, 13, 14, 15], [5, 6, 7, 8, 16, 17, 18], [9, 10, 11, 12, 19, 20, 21]]
    )
    assert np.array_equal(o[0].at(0), ref)
    assert np.array_equal(o[1].at(0), ref)


def test_cat_all_empty():
    pipe = dali.pipeline.Pipeline(batch_size=1, num_threads=3, device_id=0)
    with pipe:
        src1 = dali.types.Constant(np.array([[], [], []], dtype=np.int32))
        out_cpu = fn.cat(src1, src1, src1, axis=1)
        out_gpu = fn.cat(src1.gpu(), src1.gpu(), src1.gpu(), axis=1)
        pipe.set_outputs(out_cpu, out_gpu)

    o = pipe.run()

    o = list(o)
    o[1] = o[1].as_cpu()

    ref = np.array([[], [], []], dtype=np.int32)
    assert np.array_equal(o[0].at(0), ref)
    assert np.array_equal(o[1].at(0), ref)


def ref_cat(input_batches, axis):
    N = len(input_batches[0])
    out = []
    for i in range(N):
        inputs = [x[i] for x in input_batches]
        out.append(np.concatenate(inputs, axis=axis))
    return out


def ref_stack(input_batches, axis):
    N = len(input_batches[0])
    out = []
    for i in range(N):
        inputs = [x[i] for x in input_batches]
        out.append(np.stack(inputs, axis=axis))
    return out


def _run_test_cat(num_inputs, layout, ndim, axis, axis_name):
    num_iter = 3
    batch_size = 4
    if ndim is None:
        ndim = len(layout)

    ref_axis = layout.find(axis_name) if axis_name is not None else axis if axis is not None else 0
    assert ref_axis >= -ndim and ref_axis < ndim

    axis_arg = None if axis_name else axis

    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=3, device_id=0)
    with pipe:
        inputs = fn.external_source(
            input_generator(num_inputs, batch_size, ndim, ref_axis),
            num_outputs=num_inputs,
            layout=layout,
        )
        out_cpu = fn.cat(*inputs, axis=axis_arg, axis_name=axis_name)
        out_gpu = fn.cat(*(x.gpu() for x in inputs), axis=axis_arg, axis_name=axis_name)
        pipe.set_outputs(out_cpu, out_gpu, *inputs)

    for iter in range(num_iter):
        o_cpu, o_gpu, *inputs = pipe.run()
        ref = ref_cat(inputs, ref_axis)
        check_batch(o_cpu, ref, batch_size, eps=0, expected_layout=layout)
        check_batch(o_gpu, ref, batch_size, eps=0, expected_layout=layout)


def _run_test_stack(num_inputs, layout, ndim, axis, axis_name):
    num_iter = 3
    batch_size = 4
    if ndim is None:
        ndim = len(layout)

    ref_axis = axis if axis is not None else 0
    assert ref_axis >= -ndim and ref_axis <= ndim

    if axis_name:
        axis_pos = axis + ndim + 1 if axis < 0 else axis
        ref_layout = layout[:axis_pos] + axis_name + layout[axis_pos:] if layout else axis_name
    else:
        ref_layout = ""

    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=3, device_id=0)
    with pipe:
        inputs = fn.external_source(
            input_generator(num_inputs, batch_size, ndim), num_outputs=num_inputs, layout=layout
        )
        out_cpu = fn.stack(*inputs, axis=axis, axis_name=axis_name)
        out_gpu = fn.stack(*(x.gpu() for x in inputs), axis=axis, axis_name=axis_name)
        pipe.set_outputs(out_cpu, out_gpu, *inputs)

    for _ in range(num_iter):
        o_cpu, o_gpu, *inputs = pipe.run()
        ref = ref_stack(inputs, ref_axis)
        check_batch(o_cpu, ref, batch_size, eps=0, expected_layout=ref_layout)
        check_batch(o_gpu, ref, batch_size, eps=0, expected_layout=ref_layout)


def test_cat():
    for num_inputs in [1, 2, 3, 100]:
        for layout, ndim in [
            (None, 0),
            (None, 1),
            ("A", 1),
            (None, 2),
            ("HW", 2),
            (None, 3),
            ("DHW", 3),
        ]:
            for axis in range(-ndim, ndim):
                axis_name = layout[axis] if layout else None
                yield _run_test_cat, num_inputs, layout, ndim, axis, axis_name


def test_stack():
    for num_inputs in [1, 2, 3, 100]:
        for layout, ndim in [
            (None, 0),
            (None, 1),
            ("A", 1),
            (None, 2),
            ("HW", 2),
            (None, 3),
            ("DHW", 3),
        ]:
            for axis in range(-ndim, ndim + 1):
                axis_names = [None] if layout is None and ndim > 0 else [None, "C"]
                for axis_name in axis_names:
                    yield _run_test_stack, num_inputs, layout, ndim, axis, axis_name
