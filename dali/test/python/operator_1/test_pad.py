# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali import Pipeline, pipeline_def

from nose_utils import assert_raises
from test_utils import (
    RandomlyShapedDataIterator,
    generator_random_axes_for_3d_input,
    generator_random_data,
    as_array,
)


class PadSynthDataPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        layout="HWC",
        num_threads=1,
        device_id=0,
        num_gpus=1,
        axes=(),
        axis_names="",
        align=(),
        shape_arg=(),
    ):
        super().__init__(batch_size, num_threads, device_id, seed=1234)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.pad = ops.Pad(
            device=self.device, axes=axes, axis_names=axis_names, align=align, shape=shape_arg
        )

    def define_graph(self):
        self.data = self.inputs()
        input_data = self.data
        data = input_data.gpu() if self.device == "gpu" else input_data
        out = self.pad(data)
        return input_data, out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def check_pad(device, batch_size, input_max_shape, axes, axis_names, align, shape_arg):
    eii = RandomlyShapedDataIterator(batch_size, max_shape=input_max_shape)
    layout = "HWC"
    pipe = PadSynthDataPipeline(
        device,
        batch_size,
        iter(eii),
        axes=axes,
        axis_names=axis_names,
        align=align,
        shape_arg=shape_arg,
        layout=layout,
    )

    if axis_names:
        axes = []
        for axis_name in axis_names:
            axis_idx = layout.find(axis_name)
            assert axis_idx >= 0
            axes.append(axis_idx)

    actual_axes = axes if (axes and len(axes) > 0) else range(len(input_max_shape))
    assert len(actual_axes) > 0

    if not shape_arg or len(shape_arg) == 0:
        shape_arg = [-1] * len(actual_axes)
    assert len(shape_arg) == len(actual_axes)

    if not align or len(align) == 0:
        align = [1] * len(actual_axes)
    elif len(align) == 1 and len(actual_axes) > 1:
        align = [align[0] for _ in actual_axes]
    assert len(align) == len(actual_axes)

    for _ in range(5):
        out0, out1 = tuple(out.as_cpu() for out in pipe.run())
        max_shape = [-1] * len(input_max_shape)

        for i in range(len(actual_axes)):
            dim = actual_axes[i]
            align_val = align[i]
            shape_arg_val = shape_arg[i]
            for i in range(batch_size):
                input_shape = out0.at(i).shape
                if input_shape[dim] > max_shape[dim]:
                    max_shape[dim] = input_shape[dim]

        for i in range(batch_size):
            input_shape = out0.at(i).shape
            output_shape = out1.at(i).shape

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
                assert output_shape[dim] == expected_extent


def test_pad():
    for device in ["cpu", "gpu"]:
        for batch_size in {1, 8}:
            for input_max_shape, axes, axis_names, align, shape_arg in [
                ((200, 400, 3), (0,), None, None, None),
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
                ((200, 400, 3), (0, 1), None, (4, 16), (1, 200)),
            ]:
                yield (
                    check_pad,
                    device,
                    batch_size,
                    input_max_shape,
                    axes,
                    axis_names,
                    align,
                    shape_arg,
                )


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
    pipe = PadSynthDataPipeline(
        device,
        batch_size,
        iter(eii),
        axes=axes,
        axis_names=axis_names,
        align=align,
        shape_arg=shape_arg,
        layout=layout,
    )

    with assert_raises(RuntimeError, glob="Values of `align` argument must be positive."):
        pipe.run()


def is_aligned(sh, align, axes):
    assert len(sh) == len(align)
    for i, axis in enumerate(axes):
        if sh[axis] % align[i] > 0:
            return False
    return True


def check_pad_per_sample_shapes_and_alignment(device="cpu", batch_size=3, ndim=2, num_iter=3):
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=0, seed=1234)
    axes = (0, 1)
    with pipe:
        in_shape = fn.cast(fn.random.uniform(range=(10, 20), shape=(ndim,)), dtype=types.INT32)
        in_data = fn.random.uniform(range=(0.0, 1.0), shape=in_shape)
        if device == "gpu":
            in_data = in_data.gpu()
        req_shape = fn.cast(fn.random.uniform(range=(21, 30), shape=(ndim,)), dtype=types.INT32)
        req_align = fn.cast(fn.random.uniform(range=(3, 5), shape=(ndim,)), dtype=types.INT32)
        out_pad_shape = fn.pad(in_data, axes=axes, align=None, shape=req_shape)
        out_pad_align = fn.pad(in_data, axes=axes, align=req_align, shape=None)
        out_pad_both = fn.pad(in_data, axes=axes, align=req_align, shape=req_shape)
        pipe.set_outputs(
            in_shape, in_data, req_shape, req_align, out_pad_shape, out_pad_align, out_pad_both
        )
    for _ in range(num_iter):
        outs = tuple(out.as_cpu() for out in pipe.run())
        for i in range(batch_size):
            in_shape, in_data, req_shape, req_align, out_pad_shape, out_pad_align, out_pad_both = [
                outs[out_idx].at(i) for out_idx in range(len(outs))
            ]
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
    yield check_pad_per_sample_shapes_and_alignment, "cpu"
    yield check_pad_per_sample_shapes_and_alignment, "gpu"


def check_pad_to_square(device="cpu", batch_size=3, ndim=2, num_iter=3):
    pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=0, seed=1234)
    with pipe:
        in_shape = fn.cast(fn.random.uniform(range=(10, 20), shape=(ndim,)), dtype=types.INT32)
        in_data = fn.reshape(fn.random.uniform(range=(0.0, 1.0), shape=in_shape), layout="HW")
        shape = in_data.shape(dtype=types.INT32)
        h = fn.slice(shape, 0, 1, axes=[0])
        w = fn.slice(shape, 1, 1, axes=[0])
        side = math.max(h, w)
        if device == "gpu":
            in_data = in_data.gpu()
        out_data = fn.pad(in_data, axis_names="HW", shape=fn.cat(side, side, axis=0))
        pipe.set_outputs(in_data, out_data)
    for _ in range(num_iter):
        outs = tuple(out.as_cpu() for out in pipe.run())
        for i in range(batch_size):
            in_data, out_data = [outs[out_idx].at(i) for out_idx in range(len(outs))]
            in_shape = in_data.shape
            max_side = max(in_shape)
            for s in out_data.shape:
                assert s == max_side
            np.testing.assert_equal(out_data[: in_shape[0], : in_shape[1]], in_data)
            np.testing.assert_equal(out_data[in_shape[0] :, :], 0)
            np.testing.assert_equal(out_data[:, in_shape[1] :], 0)


def test_pad_to_square():
    yield check_pad_to_square, "cpu"
    yield check_pad_to_square, "gpu"


def check_pad_dynamic_axes(device, batch_size, num_threads, use_negative, use_empty):
    shape_arg_desc = (100, 120, np.int32)
    get_dynamic_axes = generator_random_axes_for_3d_input(
        batch_size, use_negative=use_negative, use_empty=use_empty, extra_out_desc=[shape_arg_desc]
    )

    image_gen = generator_random_data(
        batch_size, min_sh=(10, 10, 3), max_sh=(100, 100, 3), dtype=np.float32, val_range=[0.0, 1.0]
    )

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    def make_pipe():
        image = fn.external_source(source=image_gen)
        if device == "gpu":
            image = image.gpu()
        axes, shape = fn.external_source(source=get_dynamic_axes, num_outputs=2)
        fill_value = fn.random.uniform(device="cpu", range=[0.0, 255.0])
        pad1 = fn.pad(image, axes=axes, fill_value=fill_value)
        pad2 = fn.pad(image, axes=axes, shape=shape, fill_value=fill_value)
        return image, axes, shape, pad1, pad2, fill_value

    pipe = make_pipe()
    ndim = 3
    for _ in range(3):
        outs = pipe.run()
        max_shape = ndim * [-1]
        for sample_idx in range(batch_size):
            in_img_sh = as_array(outs[0][sample_idx]).shape
            for dim in range(ndim):
                if in_img_sh[dim] > max_shape[dim]:
                    max_shape[dim] = in_img_sh[dim]

        for sample_idx in range(batch_size):
            in_img = as_array(outs[0][sample_idx])
            axes = as_array(outs[1][sample_idx])
            naxes = axes.shape[0]
            if naxes == 0:  # Empty axes mean "all axes"
                axes = np.array(range(ndim), dtype=np.int32)
            shape = as_array(outs[2][sample_idx])
            pad1 = as_array(outs[3][sample_idx])
            pad2 = as_array(outs[4][sample_idx])
            fill_value = as_array(outs[5][sample_idx])
            in_sh = in_img.shape
            expected_pad1_sh = np.copy(pad1.shape)
            for d in axes:
                expected_pad1_sh[d] = max_shape[d]
            np.testing.assert_allclose(expected_pad1_sh, pad1.shape)
            np.testing.assert_allclose(pad1[: in_sh[0], : in_sh[1], : in_sh[2]], in_img)
            np.testing.assert_allclose(pad1[in_sh[0] :, : in_sh[1] :, : in_sh[2] :], fill_value)
            expected_pad2_sh = np.copy(pad2.shape)
            for d, req_extent in zip(axes, shape):
                expected_pad2_sh[d] = req_extent if req_extent > 0 else max_shape[d]
            np.testing.assert_allclose(expected_pad2_sh, pad2.shape)
            np.testing.assert_allclose(pad2[: in_sh[0], : in_sh[1], : in_sh[2]], in_img)
            np.testing.assert_allclose(pad2[in_sh[0] :, : in_sh[1] :, : in_sh[2] :], fill_value)


def test_dynamic_axes():
    batch_size = 10
    num_threads = 3
    for device in ["cpu", "gpu"]:
        yield check_pad_dynamic_axes, device, batch_size, num_threads, False, False


def test_negative_axes():
    batch_size = 10
    num_threads = 3
    for device in ["cpu", "gpu"]:
        yield check_pad_dynamic_axes, device, batch_size, num_threads, True, False


def test_empty_axes():
    batch_size = 10
    num_threads = 3
    for device in ["cpu", "gpu"]:
        yield check_pad_dynamic_axes, device, batch_size, num_threads, False, True


def check_pad_wrong_axes(device, wrong_axes_range=None):
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def make_pipe():
        fake_data = types.Constant(0, shape=[10, 10, 3], dtype=types.FLOAT, device=device)
        axes = fn.random.uniform(range=wrong_axes_range, shape=(2,), dtype=types.INT32)
        padded = fn.pad(fake_data, axes=axes)
        return padded

    p = make_pipe()
    # Note: [[] and []] are '[' and ']' characters.
    assert_raises(
        RuntimeError,
        p.run,
        glob="Axis * out of range. Expected range is [[]-3, 2[]] for a 3D input",
    )


def test_wrong_axes():
    for device in ["cpu", "gpu"]:
        for wrong_axes_range in [(-10, -4), (3, 10)]:
            yield check_pad_wrong_axes, device, wrong_axes_range
