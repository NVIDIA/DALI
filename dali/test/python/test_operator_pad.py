# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
import os
from nose.tools import assert_raises

from test_utils import RandomlyShapedDataIterator

class PadSynthDataPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator,
                 layout="HWC", num_threads=1, device_id=0, num_gpus=1, axes=(), axis_names="", align=(), shape_arg=()):
        super(PadSynthDataPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=1234)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.pad = ops.Pad(device = self.device, axes=axes, axis_names=axis_names, align=align, shape=shape_arg)

    def define_graph(self):
        self.data = self.inputs()
        input_data = self.data
        data = input_data.gpu() if self.device == 'gpu' else input_data
        out = self.pad(data)
        return input_data, out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def check_pad(device, batch_size, input_max_shape, axes, axis_names, align, shape_arg):
    eii = RandomlyShapedDataIterator(batch_size, max_shape=input_max_shape)
    layout = "HWC"
    pipe = PadSynthDataPipeline(device, batch_size, iter(eii), axes=axes, axis_names=axis_names,
                                align=align, shape_arg=shape_arg, layout=layout)
    pipe.build()

    if axis_names:
        axes = []
        for axis_name in axis_names:
            axis_idx = layout.find(axis_name)
            assert(axis_idx >= 0)
            axes.append(axis_idx)

    actual_axes = axes if (axes and len(axes) > 0) else range(len(input_max_shape))
    assert(len(actual_axes)>0)

    if not shape_arg or len(shape_arg) == 0:
        shape_arg = [-1] * len(actual_axes)
    assert(len(shape_arg) == len(actual_axes))

    if not align or len(align) == 0:
        align = [1] * len(actual_axes)
    elif len(align) == 1 and len(actual_axes) > 1:
        align = [align[0] for _ in actual_axes]
    assert(len(align) == len(actual_axes))

    for k in range(5):
        out1, out2 = pipe.run()

        out1_data = out1.as_cpu() if isinstance(out1[0], dali.backend_impl.TensorGPU) else out1
        max_shape = [-1] * len(input_max_shape)

        for i in range(len(actual_axes)):
            dim = actual_axes[i]
            align_val = align[i]
            shape_arg_val = shape_arg[i]
            for i in range(batch_size):
                input_shape = out1_data.at(i).shape
                if input_shape[dim] > max_shape[dim]:
                    max_shape[dim] = input_shape[dim]

        out2_data = out2.as_cpu() if isinstance(out2[0], dali.backend_impl.TensorGPU) else out2
        for i in range(batch_size):
            input_shape = out1_data.at(i).shape
            output_shape = out2_data.at(i).shape

            for j in range(len(actual_axes)):
                dim = actual_axes[j]
                align_val = align[j]
                shape_arg_val = shape_arg[j]
                if shape_arg_val >= 0:
                    in_extent = input_shape[dim]
                    expected_extent = in_extent if in_extent > shape_arg_val else shape_arg_val
                else:
                    expected_extent = max_shape[dim]
                remainder = expected_extent % align_val
                if remainder > 0:
                    expected_extent = expected_extent - remainder + align_val
                assert(output_shape[dim] == expected_extent)

def test_pad():
    for device in ["cpu", "gpu"]:
        for batch_size in {1, 8}:
            for input_max_shape, axes, axis_names, align, shape_arg in \
                [((200, 400, 3), (0,), None, None, None),
                 ((200, 400, 3), None, "H", None, None),
                 ((200, 400, 3), (1,), None, None, None),
                 ((200, 400, 3), None, "W", None, None),
                 ((200, 400, 3), (0, 1), None, None, None),
                 ((200, 400, 3), None, "HW", None, None),
                 ((200, 400, 3), (), None, None, None),
                 ((200, 400, 3), [], None, None, None),
                 ((200, 400, 3), None, "", None, None),
                 ((200, 400, 3), (2,), None, (4,), None),
                 ((200, 400, 3), None, "C", (4,), None),
                 ((200, 400, 3), (0, 1), None, (256, 256), None),
                 ((200, 400, 3), None, "HW", (256, 256), None),
                 ((200, 400, 3), (0, 1), None, (16, 64), None),
                 ((200, 400, 3), None, "HW", (16, 64), None),
                 ((200, 400, 3), (0, 1), None, (256,), None),
                 ((200, 400, 3), None, "HW", (256,), None),
                 ((200, 400, 3), None, None, None, (-1, -1, 4)),
                 ((25, 100, 3), (0,), None, None, (25,)),
                 ((200, 400, 3), (0, 1), None, (4, 16), (1, 200))]:
                yield check_pad, device, batch_size, input_max_shape, axes, axis_names, align, shape_arg

def test_pad_error():
    batch_size = 2
    input_max_shape = (5, 5, 3)
    device = "cpu"
    axes = None
    layout = "HWC"
    axis_names = "H"
    align = 0
    shape_arg = None

    eii = RandomlyShapedDataIterator(batch_size, max_shape=input_max_shape)
    pipe = PadSynthDataPipeline(device, batch_size, iter(eii), axes=axes, axis_names=axis_names,
                                align=align, shape_arg=shape_arg, layout=layout)

    pipe.build()
    assert_raises(RuntimeError, pipe.run)

def is_aligned(sh, align, axes):
    assert len(sh) == len(align)
    for i, axis in enumerate(axes):
        if sh[axis] % align[i] > 0:
            return False
    return True

def check_pad_per_sample_shapes_and_alignment(device='cpu', batch_size=3, ndim=2, num_iter=3):
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=0, seed=1234)
    axes = (0, 1)
    with pipe:
        in_shape = fn.cast(fn.random.uniform(range=(10, 20), shape=(ndim,)), dtype=types.INT32)
        in_data = fn.random.uniform(range=(0., 1.), shape=in_shape)
        if device == 'gpu':
            in_data = in_data.gpu()
        req_shape = fn.cast(fn.random.uniform(range=(21, 30), shape=(ndim,)), dtype=types.INT32)
        req_align = fn.cast(fn.random.uniform(range=(3, 5), shape=(ndim,)), dtype=types.INT32)
        out_pad_shape = fn.pad(in_data, axes=axes, align=None, shape=req_shape)
        out_pad_align = fn.pad(in_data, axes=axes, align=req_align, shape=None)
        out_pad_both = fn.pad(in_data, axes=axes, align=req_align, shape=req_shape)
        pipe.set_outputs(in_shape, in_data, req_shape, req_align, out_pad_shape, out_pad_align, out_pad_both)
    pipe.build()
    for _ in range(num_iter):
        outs = [out.as_cpu() if isinstance(out, TensorListGPU) else out for out in pipe.run()]
        for i in range(batch_size):
            in_shape, in_data, req_shape, req_align, out_pad_shape, out_pad_align, out_pad_both = \
                [outs[out_idx].at(i) for out_idx in range(len(outs))]
            assert (in_shape == in_data.shape).all()
            # Pad to explicit shape
            assert (out_pad_shape.shape >= in_shape).all()
            assert (req_shape == out_pad_shape.shape).all()

            # Alignment only
            assert (out_pad_align.shape >= in_shape).all()
            assert is_aligned(out_pad_align.shape, req_align, axes)

            # Explicit shape + alignment
            assert (out_pad_both.shape >= in_shape).all()
            assert (req_shape <= out_pad_both.shape).all()
            assert is_aligned(out_pad_both.shape, req_align, axes)

def test_pad_per_sample_shapes_and_alignment():
    yield check_pad_per_sample_shapes_and_alignment, 'cpu'
    yield check_pad_per_sample_shapes_and_alignment, 'gpu'

def check_pad_to_square(device='cpu', batch_size=3, ndim=2, num_iter=3):
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=0, seed=1234)
    axes = (0, 1)
    with pipe:
        in_shape = fn.cast(fn.random.uniform(range=(10, 20), shape=(ndim,)), dtype=types.INT32)
        in_data = fn.reshape(fn.random.uniform(range=(0., 1.), shape=in_shape), layout="HW")
        shape = fn.shapes(in_data, dtype=types.INT32)
        h = fn.slice(shape, 0, 1, axes = [0])
        w = fn.slice(shape, 1, 1, axes = [0])
        side = math.max(h, w)
        if device == 'gpu':
            in_data = in_data.gpu()
        out_data = fn.pad(in_data, axis_names="HW", shape=fn.cat(side, side, axis=0))
        pipe.set_outputs(in_data, out_data)
    pipe.build()
    for _ in range(num_iter):
        outs = [out.as_cpu() if isinstance(out, TensorListGPU) else out for out in pipe.run()]
        for i in range(batch_size):
            in_data, out_data = \
                [outs[out_idx].at(i) for out_idx in range(len(outs))]
            in_shape = in_data.shape
            max_side = max(in_shape)
            for s in out_data.shape:
                assert s == max_side
            np.testing.assert_equal(out_data[:in_shape[0], :in_shape[1]], in_data)
            np.testing.assert_equal(out_data[in_shape[0]:, :], 0)
            np.testing.assert_equal(out_data[:, in_shape[1]:], 0)

def test_pad_to_square():
    yield check_pad_to_square, 'cpu'
    yield check_pad_to_square, 'gpu'
