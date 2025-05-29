# Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import Pipeline, pipeline_def, ops, fn, types
import numpy as np
import os
from functools import partial
from math import floor
from test_utils import (
    compare_pipelines,
    get_dali_extra_path,
    RandomDataIterator,
    generator_random_axes_for_3d_input,
    generator_random_data,
    as_array,
)
from nose_utils import assert_raises
from nose2.tools import params

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, "db", "lmdb")
test_data_video = os.path.join(test_data_root, "db", "optical_flow", "sintel_trailer")


def roundint(num):
    # std::round has different behavior than np.round so manually add 0.5 and truncate to int
    return int(num + (0.5 if num >= 0 else -0.5))


def abs_slice_start_and_end(
    in_shape, slice_anchor, slice_shape, normalized_anchor, normalized_shape
):
    ndim = len(in_shape)
    if normalized_anchor and normalized_shape:
        start = [roundint(in_shape[i] * np.float64(slice_anchor[i])) for i in range(ndim)]
        end = [
            roundint(in_shape[i] * np.float64(slice_anchor[i] + slice_shape[i]))
            for i in range(ndim)
        ]
    else:
        if normalized_anchor:
            start = [roundint(in_shape[i] * np.float64(slice_anchor[i])) for i in range(ndim)]
        else:
            start = [roundint(slice_anchor[i]) for i in range(ndim)]

        if normalized_shape:
            end = [
                start[i] + roundint(in_shape[i] * np.float64(slice_shape[i])) for i in range(ndim)
            ]
        else:
            end = [start[i] + roundint(slice_shape[i]) for i in range(ndim)]
    out_shape = [end[i] - start[i] for i in range(ndim)]
    return start, end, out_shape


class SliceSynthDataPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        layout,
        iterator,
        pos_size_iter,
        num_threads=1,
        device_id=0,
        num_gpus=1,
        axes=None,
        axis_names=None,
        normalized_anchor=True,
        normalized_shape=True,
        extra_outputs=False,
        out_of_bounds_policy=None,
        fill_values=None,
        input_type=types.FLOAT,
        output_type=None,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=1234)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.pos_size_iter = pos_size_iter
        self.inputs = ops.ExternalSource()
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()
        self.extra_outputs = extra_outputs
        self.cast_in = ops.Cast(dtype=input_type)
        self.slice = ops.Slice(
            device=self.device,
            dtype=output_type,
            normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape,
            axes=axes,
            axis_names=axis_names,
            out_of_bounds_policy=out_of_bounds_policy,
            fill_values=fill_values,
        )

    def define_graph(self):
        self.data = self.inputs()
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()
        data = self.cast_in(self.data)
        data = data.gpu() if self.device == "gpu" else data
        out = self.slice(data, self.crop_pos, self.crop_size)
        if self.extra_outputs:
            return out, self.data, self.crop_pos, self.crop_size
        else:
            return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)


class SlicePipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        pos_size_iter,
        num_threads=1,
        device_id=0,
        is_fused_decoder=False,
        axes=None,
        axis_names=None,
        normalized_anchor=True,
        normalized_shape=True,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=1234)
        self.is_fused_decoder = is_fused_decoder
        self.pos_size_iter = pos_size_iter
        self.device = device
        self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=False)
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()

        if self.is_fused_decoder:
            self.decode = ops.decoders.ImageSlice(
                device="cpu",
                output_type=types.RGB,
                normalized_anchor=normalized_anchor,
                normalized_shape=normalized_shape,
                axis_names=axis_names,
                axes=axes,
            )
        else:
            self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
            self.slice = ops.Slice(
                device=self.device,
                normalized_anchor=normalized_anchor,
                normalized_shape=normalized_shape,
                axis_names=axis_names,
                axes=axes,
            )

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()

        if self.is_fused_decoder:
            images = self.decode(inputs, self.crop_pos, self.crop_size)
        else:
            images = self.decode(inputs)
            if self.device == "gpu":
                images = images.gpu()
            images = self.slice(images, self.crop_pos, self.crop_size)
        return images

    def iter_setup(self):
        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)


class SliceArgsIterator(object):
    def __init__(
        self,
        batch_size,
        num_dims=3,
        image_shape=None,  # Needed if normalized_anchor and normalized_shape are False
        image_layout=None,  # Needed if axis_names is used to specify the slice
        normalized_anchor=True,
        normalized_shape=True,
        axes=None,
        axis_names=None,
        min_norm_anchor=0.0,
        max_norm_anchor=0.2,
        min_norm_shape=0.4,
        max_norm_shape=0.75,
        seed=54643613,
    ):
        self.batch_size = batch_size
        self.num_dims = num_dims
        self.image_shape = image_shape
        self.image_layout = image_layout
        self.normalized_anchor = normalized_anchor
        self.normalized_shape = normalized_shape
        self.axes = axes
        self.axis_names = axis_names
        self.min_norm_anchor = min_norm_anchor
        self.max_norm_anchor = max_norm_anchor
        self.min_norm_shape = min_norm_shape
        self.max_norm_shape = max_norm_shape
        self.seed = seed

        if not self.axis_names and not self.axes:
            self.axis_names = "WH"

        if self.axis_names:
            self.axes = []
            for axis_name in self.axis_names:
                assert axis_name in self.image_layout
                self.axes.append(self.image_layout.index(axis_name))
        assert len(self.axes) > 0

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        pos = []
        size = []
        anchor_amplitude = self.max_norm_anchor - self.min_norm_anchor
        anchor_offset = self.min_norm_anchor
        shape_amplitude = self.max_norm_shape - self.min_norm_shape
        shape_offset = self.min_norm_shape
        np.random.seed(self.seed)
        for k in range(self.batch_size):
            norm_anchor = anchor_amplitude * np.random.rand(len(self.axes)) + anchor_offset
            norm_shape = shape_amplitude * np.random.rand(len(self.axes)) + shape_offset

            if self.normalized_anchor:
                anchor = norm_anchor
            else:
                anchor = [
                    floor(norm_anchor[i] * self.image_shape[self.axes[i]])
                    for i in range(len(self.axes))
                ]

            if self.normalized_shape:
                shape = norm_shape
            else:
                shape = [
                    floor(norm_shape[i] * self.image_shape[self.axes[i]])
                    for i in range(len(self.axes))
                ]

            pos.append(np.asarray(anchor, dtype=np.float32))
            size.append(np.asarray(shape, dtype=np.float32))
            self.i = (self.i + 1) % self.n
        return (pos, size)

    next = __next__


def slice_func_helper(
    axes, axis_names, layout, normalized_anchor, normalized_shape, image, slice_anchor, slice_shape
):
    # TODO(janton): remove this
    if not axes and not axis_names:
        axis_names = "WH"

    if axis_names:
        axes = []
        for axis_name in axis_names:
            assert axis_name in layout
            axis_pos = layout.find(axis_name)
            axes.append(axis_pos)

    shape = image.shape
    full_slice_anchor = [0] * len(shape)
    full_slice_shape = list(shape)
    for axis in axes:
        idx = axes.index(axis)
        full_slice_anchor[axis] = slice_anchor[idx]
        full_slice_shape[axis] = slice_shape[idx]

    start, end, _ = abs_slice_start_and_end(
        shape, full_slice_anchor, full_slice_shape, normalized_anchor, normalized_shape
    )

    if len(full_slice_anchor) == 1:
        return image[start[0] : end[0]]
    elif len(full_slice_anchor) == 2:
        return image[start[0] : end[0], start[1] : end[1]]
    elif len(full_slice_anchor) == 3:
        return image[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
    elif len(full_slice_anchor) == 4:
        return image[start[0] : end[0], start[1] : end[1], start[2] : end[2], start[3] : end[3]]
    else:
        assert False


class SliceSynthDataPipelinePythonOp(Pipeline):
    def __init__(
        self,
        batch_size,
        layout,
        iterator,
        pos_size_iter,
        num_threads=1,
        device_id=0,
        num_gpus=1,
        axes=None,
        axis_names=None,
        normalized_anchor=True,
        normalized_shape=True,
        input_type=types.FLOAT,
        output_type=None,
    ):
        super().__init__(
            batch_size, num_threads, device_id, seed=12345, exec_async=False, exec_pipelined=False
        )
        self.device = "cpu"
        self.layout = layout
        self.iterator = iterator
        self.pos_size_iter = pos_size_iter
        self.inputs = ops.ExternalSource()
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()
        self.cast_in = ops.Cast(dtype=input_type)
        function = partial(
            slice_func_helper, axes, axis_names, self.layout, normalized_anchor, normalized_shape
        )
        self.slice = ops.PythonFunction(function=function, output_layouts=layout)
        self.output_type = output_type
        if self.output_type is not None:
            self.cast_out = ops.Cast(dtype=output_type)

    def define_graph(self):
        self.data = self.inputs()
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()
        out = self.cast_in(self.data)
        out = self.slice(out, self.crop_pos, self.crop_size)
        if self.output_type is not None:
            out = self.cast_out(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)


class SlicePythonOp(Pipeline):
    def __init__(
        self,
        batch_size,
        pos_size_iter,
        num_threads=1,
        device_id=0,
        num_gpus=1,
        axes=None,
        axis_names=None,
        normalized_anchor=True,
        normalized_shape=True,
    ):
        super().__init__(
            batch_size, num_threads, device_id, seed=12345, exec_async=False, exec_pipelined=False
        )
        self.device = "cpu"
        self.layout = "HWC"
        self.pos_size_iter = pos_size_iter

        self.input = ops.readers.Caffe(path=caffe_db_folder, random_shuffle=False)
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)

        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()

        function = partial(
            slice_func_helper, axes, axis_names, self.layout, normalized_anchor, normalized_shape
        )
        self.slice = ops.PythonFunction(function=function, output_layouts="HWC")

    def define_graph(self):
        imgs, _ = self.input()
        imgs = self.decode(imgs)
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()
        out = self.slice(imgs, self.crop_pos, self.crop_size)
        return out

    def iter_setup(self):
        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)


def check_slice_synth_data_vs_numpy(
    device,
    batch_size,
    input_shape,
    layout,
    axes,
    axis_names,
    normalized_anchor,
    normalized_shape,
    input_type,
    output_type,
):
    eiis = [RandomDataIterator(batch_size, shape=input_shape) for k in range(2)]
    eii_args = [
        SliceArgsIterator(
            batch_size,
            len(input_shape),
            image_shape=input_shape,
            image_layout=layout,
            axes=axes,
            axis_names=axis_names,
            normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape,
        )
        for k in range(2)
    ]

    compare_pipelines(
        SliceSynthDataPipeline(
            device,
            batch_size,
            layout,
            iter(eiis[0]),
            iter(eii_args[0]),
            axes=axes,
            axis_names=axis_names,
            normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape,
            input_type=input_type,
            output_type=output_type,
        ),
        SliceSynthDataPipelinePythonOp(
            batch_size,
            layout,
            iter(eiis[0]),
            iter(eii_args[1]),
            axes=axes,
            axis_names=axis_names,
            normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape,
            input_type=input_type,
            output_type=output_type,
        ),
        batch_size=batch_size,
        N_iterations=3,
    )


def test_slice_synth_data_vs_numpy():
    for device in ["cpu", "gpu"]:
        for batch_size in {1, 8}:
            for input_shape, layout, axes, axis_names, input_type, output_type in [
                ((200, 400, 3), "HWC", None, "WH", types.FLOAT, None),
                ((200, 400, 3), "HWC", None, "HW", types.FLOAT, None),
                ((200, 400, 3), "HWC", None, "HW", types.INT32, types.FLOAT),
                ((200, 400, 3), "HWC", None, "HW", types.INT64, types.UINT8),
                ((200, 400, 3), "HWC", None, "C", types.FLOAT, None),
                ((200, 400, 3), "HWC", (1, 0), None, types.FLOAT, types.FLOAT16),
                ((200, 400, 3), "HWC", (0, 1), None, types.FLOAT16, types.FLOAT16),
                ((200, 400, 3), "HWC", (2,), None, types.FLOAT, None),
                ((200,), "H", (0,), None, types.FLOAT, None),
                ((200,), "H", None, "H", types.FLOAT, None),
                ((200, 400), "HW", (1,), None, types.FLOAT, None),
                ((200, 400), "HW", None, "W", types.FLOAT, None),
                ((80, 30, 20, 3), "DHWC", (2, 1, 0), None, types.FLOAT, None),
                ((80, 30, 20, 3), "DHWC", (0, 1, 2), None, types.FLOAT, None),
                ((80, 30, 20, 3), "DHWC", (2, 1), None, types.FLOAT, None),
                ((80, 30, 20, 3), "DHWC", None, "WHD", types.FLOAT, None),
                ((80, 30, 20, 3), "DHWC", None, "DHW", types.FLOAT, None),
                ((80, 30, 20, 3), "DHWC", None, "WH", types.FLOAT, None),
                ((80, 30, 20, 3), "DHWC", None, "C", types.FLOAT, None),
            ]:
                normalized_anchor = np.random.choice([True, False])
                normalized_shape = np.random.choice([True, False])
                yield (
                    check_slice_synth_data_vs_numpy,
                    device,
                    batch_size,
                    input_shape,
                    layout,
                    axes,
                    axis_names,
                    normalized_anchor,
                    normalized_shape,
                    input_type,
                    output_type,
                )


def check_slice_vs_fused_decoder(device, batch_size, axes, axis_names):
    eii_args = [
        SliceArgsIterator(batch_size, image_layout="HWC", axes=axes, axis_names=axis_names)
        for k in range(2)
    ]
    compare_pipelines(
        SlicePipeline(
            device,
            batch_size,
            iter(eii_args[0]),
            axes=axes,
            axis_names=axis_names,
            is_fused_decoder=False,
        ),
        SlicePipeline(
            device,
            batch_size,
            iter(eii_args[1]),
            axes=axes,
            axis_names=axis_names,
            is_fused_decoder=True,
        ),
        batch_size=batch_size,
        N_iterations=3,
    )


def test_slice_vs_fused_decoder():
    for device in ["cpu", "gpu"]:
        for batch_size in {1}:
            for axes, axis_names in [(None, "WH"), (None, "HW"), ((1, 0), None), ((0, 1), None)]:
                yield check_slice_vs_fused_decoder, device, batch_size, axes, axis_names


def check_slice_vs_numpy(device, batch_size, axes, axis_names):
    eii_args = [
        SliceArgsIterator(batch_size, image_layout="HWC", axes=axes, axis_names=axis_names)
        for k in range(2)
    ]
    compare_pipelines(
        SlicePipeline(device, batch_size, iter(eii_args[0]), axes=axes, axis_names=axis_names),
        SlicePythonOp(batch_size, iter(eii_args[1]), axes=axes, axis_names=axis_names),
        batch_size=batch_size,
        N_iterations=3,
    )


def test_slice_vs_numpy():
    for device in ["cpu", "gpu"]:
        for batch_size in {1}:
            for axes, axis_names in [(None, "WH"), (None, "HW"), ((1, 0), None), ((0, 1), None)]:
                yield check_slice_vs_numpy, device, batch_size, axes, axis_names


def check_slice_output(
    sample_in,
    sample_out,
    anchor,
    abs_slice_shape,
    abs_start,
    abs_end,
    out_of_bounds_policy,
    fill_values,
    naxes=2,
    mean=None,
    std=None,
    flip=None,
    permute=None,
):
    in_shape = sample_in.shape
    out_shape = sample_out.shape
    ndim = len(out_shape)
    orig_nchannels = in_shape[2]
    out_ch_dim = permute.index(2) if permute is not None else 2
    out_nchannels = out_shape[out_ch_dim]

    if out_of_bounds_policy == "pad":
        if permute is not None:
            assert all(
                [
                    abs_slice_shape[permute[i]] == out_shape[i]
                    for i in range(ndim)
                    if permute[i] < naxes
                ]
            )
        else:
            assert all([abs_slice_shape[i] == out_shape[i] for i in range(naxes)])
    elif out_of_bounds_policy == "trim_to_shape":
        assert all([out_shape[i] <= in_shape[i] for i in range(naxes)])
        for i in range(naxes):
            if abs_start[i] < 0:
                abs_start[i] = 0
            if abs_end[i] > in_shape[i]:
                abs_end[i] = in_shape[i]
            abs_slice_shape[i] = abs_end[i] - abs_start[i]
        if permute is not None:
            assert all(
                [
                    abs_slice_shape[permute[i]] == out_shape[i]
                    for i in range(ndim)
                    if permute[i] < naxes
                ]
            )
        else:
            assert all([abs_slice_shape[i] == out_shape[i] for i in range(naxes)])
    else:
        raise ValueError(f"Wrong out_of_bounds_policy: {out_of_bounds_policy}")

    pad_before = [-abs_start[i] if abs_start[i] < 0 else 0 for i in range(naxes)]
    pad_after = [abs_end[i] - in_shape[i] if in_shape[i] < abs_end[i] else 0 for i in range(naxes)]
    sliced = [abs_slice_shape[i] - pad_before[i] - pad_after[i] for i in range(naxes)]

    if out_of_bounds_policy == "trim_to_shape":
        assert all([pad_before[i] == 0 for i in range(naxes)])
        assert all([pad_after[i] == 0 for i in range(naxes)])

        if permute is not None:
            assert all(
                [sliced[permute[i]] == out_shape[i] for i in range(ndim) if permute[i] < naxes]
            )
        else:
            assert all([sliced[i] == out_shape[i] for i in range(naxes)])

    pos_start = [abs_start[i] if abs_start[i] >= 0 else 0 for i in range(naxes)]

    in_sliced = sample_in[
        pos_start[0] : pos_start[0] + sliced[0],  # noqa:E203
        pos_start[1] : pos_start[1] + sliced[1],
        :,
    ]  # noqa:E203

    slice_shape = (abs_slice_shape[0], abs_slice_shape[1], out_nchannels)
    expected = np.zeros(slice_shape, dtype=np.float32)
    expected[:, :, :orig_nchannels] = np.full(
        (slice_shape[0], slice_shape[1], orig_nchannels), fill_values
    )
    should_normalize = mean is not None and std is not None

    expected[
        pad_before[0] : pad_before[0] + sliced[0],  # noqa:E203
        pad_before[1] : pad_before[1] + sliced[1],  # noqa:E203
        :orig_nchannels,
    ] = (
        (in_sliced - mean) / std if should_normalize else in_sliced
    )

    if flip is not None:
        for d in range(len(flip)):
            if flip[d]:
                expected = np.flip(expected, d)

    if permute is not None:
        expected = np.transpose(expected, permute)

    np.testing.assert_allclose(sample_out, expected, atol=1e-07)


def check_slice_with_out_of_bounds_policy_support(
    device,
    batch_size,
    input_shape=(100, 200, 3),
    out_of_bounds_policy=None,
    fill_values=(0x76, 0xB9, 0x00),
    normalized_anchor=False,
    normalized_shape=False,
):
    # This test case is written with HWC layout in mind and "HW" axes in slice arguments
    axis_names = "HW"
    axes = None
    layout = "HWC"
    assert len(input_shape) == 3
    if fill_values is not None and len(fill_values) > 1:
        assert input_shape[2] == len(fill_values)

    eii = RandomDataIterator(batch_size, shape=input_shape)
    eii_arg = SliceArgsIterator(
        batch_size,
        len(input_shape),
        image_shape=input_shape,
        image_layout=layout,
        axes=axes,
        axis_names=axis_names,
        normalized_anchor=normalized_anchor,
        normalized_shape=normalized_shape,
        min_norm_anchor=-0.5,
        max_norm_anchor=-0.1,
        min_norm_shape=1.1,
        max_norm_shape=3.6,
    )
    pipe = SliceSynthDataPipeline(
        device,
        batch_size,
        layout,
        iter(eii),
        iter(eii_arg),
        axes=axes,
        axis_names=axis_names,
        normalized_anchor=normalized_anchor,
        normalized_shape=normalized_shape,
        out_of_bounds_policy=out_of_bounds_policy,
        fill_values=fill_values,
        extra_outputs=True,
    )
    if fill_values is None:
        fill_values = 0
    for _ in range(3):
        outs = pipe.run()
        out, in_data, anchor_data, shape_data = outs
        assert batch_size == len(out)
        for idx in range(batch_size):
            sample_in = as_array(in_data[idx])
            sample_out = as_array(out[idx])
            anchor = as_array(anchor_data[idx])
            shape = as_array(shape_data[idx])
            in_shape = sample_in.shape
            abs_start, abs_end, abs_slice_shape = abs_slice_start_and_end(
                in_shape[:2], anchor, shape, normalized_anchor, normalized_shape
            )
            check_slice_output(
                sample_in,
                sample_out,
                anchor,
                abs_slice_shape,
                abs_start,
                abs_end,
                out_of_bounds_policy,
                fill_values,
            )


def test_slice_with_out_of_bounds_policy_support():
    in_shape = (40, 80, 3)
    for out_of_bounds_policy in ["pad", "trim_to_shape"]:
        for device in ["gpu", "cpu"]:
            for batch_size in [1, 3]:
                for normalized_anchor, normalized_shape in [(False, False), (True, True)]:
                    for fill_values in [None, (0x76, 0xB0, 0x00)]:
                        yield (
                            check_slice_with_out_of_bounds_policy_support,
                            device,
                            batch_size,
                            in_shape,
                            out_of_bounds_policy,
                            fill_values,
                            normalized_anchor,
                            normalized_shape,
                        )


def check_slice_with_out_of_bounds_error(
    device, batch_size, input_shape=(100, 200, 3), normalized_anchor=False, normalized_shape=False
):
    # This test case is written with HWC layout in mind and "HW" axes in slice arguments
    axis_names = "HW"
    axes = None
    layout = "HWC"
    assert len(input_shape) == 3

    eii = RandomDataIterator(batch_size, shape=input_shape)
    eii_arg = SliceArgsIterator(
        batch_size,
        len(input_shape),
        image_shape=input_shape,
        image_layout=layout,
        axes=axes,
        axis_names=axis_names,
        normalized_anchor=normalized_anchor,
        normalized_shape=normalized_shape,
        min_norm_anchor=-0.5,
        max_norm_anchor=-0.1,
        min_norm_shape=1.1,
        max_norm_shape=3.6,
    )
    pipe = SliceSynthDataPipeline(
        device,
        batch_size,
        layout,
        iter(eii),
        iter(eii_arg),
        axes=axes,
        axis_names=axis_names,
        normalized_anchor=normalized_anchor,
        normalized_shape=normalized_shape,
        out_of_bounds_policy="error",
    )

    with assert_raises(
        RuntimeError, glob="Slice can't be placed out of bounds with current policy. Got:"
    ):
        _ = pipe.run()


def test_slice_with_out_of_bounds_error():
    in_shape = (40, 80, 3)
    for device in ["gpu", "cpu"]:
        for batch_size in [1, 3]:
            for normalized_anchor, normalized_shape in [(False, False), (True, True)]:
                yield (
                    check_slice_with_out_of_bounds_error,
                    device,
                    batch_size,
                    in_shape,
                    normalized_anchor,
                    normalized_shape,
                )


def check_slice_named_args(device, batch_size):
    test_data_shape = [5, 4, 3]

    def get_data():
        out = [
            np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
            for _ in range(batch_size)
        ]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    with pipe:
        data = fn.external_source(source=get_data, layout="HWC")
        in_shape_list = [5, 4]
        start_list = [1, 2]
        shape_list = [3, 1]
        in_shape = np.array(in_shape_list)
        start = np.array(start_list)
        shape = np.array(shape_list)
        end_list = [start_list[i] + shape_list[i] for i in range(2)]
        end = start + shape
        rel_start_list = [start_list[i] / in_shape_list[i] for i in range(2)]
        rel_start = start / in_shape
        rel_shape_list = [shape_list[i] / in_shape_list[i] for i in range(2)]
        rel_shape = shape / in_shape
        rel_end_list = [end_list[i] / in_shape_list[i] for i in range(2)]
        rel_end = end / in_shape

        outs = [
            fn.slice(data, start, shape, axes=(0, 1)),
            fn.slice(data, rel_start, rel_shape, axes=(0, 1)),
        ]

        for start_arg in [start, start_list]:
            for shape_arg in [shape, shape_list]:
                outs += [fn.slice(data, start=start_arg, shape=shape_arg, axes=(0, 1))]
            for end_arg in [end, end_list]:
                outs += [fn.slice(data, start=start_arg, end=end_arg, axes=(0, 1))]
        for rel_start_arg in [rel_start, rel_start_list]:
            for rel_shape_arg in [rel_shape, rel_shape_list]:
                outs += [
                    fn.slice(data, rel_start=rel_start_arg, rel_shape=rel_shape_arg, axes=(0, 1))
                ]
            for rel_end_arg in [rel_end, rel_end_list]:
                outs += [fn.slice(data, rel_start=rel_start_arg, rel_end=rel_end_arg, axes=(0, 1))]
            for shape_arg in [shape, shape_list]:
                outs += [fn.slice(data, rel_start=rel_start_arg, shape=shape_arg, axes=(0, 1))]

        pipe.set_outputs(*outs)
    for _ in range(3):
        outs = pipe.run()
        for out_idx in range(1, len(outs)):
            for sample in range(batch_size):
                np.testing.assert_equal(np.array(outs[0][sample]), np.array(outs[out_idx][sample]))


def test_slice_named_args():
    yield check_slice_named_args, "cpu", 3
    yield check_slice_named_args, "gpu", 3


def check_slice_named_args_default_start_or_end(device, batch_size):
    test_data_shape = [5, 4, 3]

    def get_data():
        out = [
            np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
            for _ in range(batch_size)
        ]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    with pipe:
        data = fn.external_source(source=get_data, layout="HWC")
        in_shape = np.array([5, 4])
        start = np.array([1, 2])
        shape = np.array([3, 1])
        end = start + shape
        outs = [
            fn.slice(data, start=start, end=in_shape, axes=(0, 1)),
            fn.slice(data, start=[0, 0], end=end, axes=(0, 1)),
            fn.slice(data, start=start, axes=(0, 1)),
            fn.slice(data, end=end, axes=(0, 1)),
        ]
        pipe.set_outputs(*outs)
    for _ in range(3):
        outs = pipe.run()
        for sample in range(batch_size):
            np.testing.assert_equal(np.array(outs[0][sample]), np.array(outs[2][sample]))
            np.testing.assert_equal(np.array(outs[1][sample]), np.array(outs[3][sample]))


def test_slice_named_default_start_or_end_args():
    yield check_slice_named_args_default_start_or_end, "cpu", 3
    yield check_slice_named_args_default_start_or_end, "gpu", 3


def check_slice_named_args_errors(device, batch_size):
    test_data_shape = [5, 4, 3]

    def get_data():
        out = [
            np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
            for _ in range(batch_size)
        ]
        return out

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)

    with pipe:
        data = fn.external_source(source=get_data, layout="HWC")
        start = np.array([1, 2])
        shape = np.array([3, 1])
        outs = [
            fn.slice(data, start, shape, start=start, end=start + shape, shape=shape, axes=(0, 1)),
        ]
        pipe.set_outputs(*outs)

    with assert_raises(
        RuntimeError,
        glob='"end", "rel_end", "shape", and "rel_shape" arguments are mutually exclusive',
    ):
        for _ in range(1):
            outs = pipe.run()


def test_slice_named_args_errors():
    yield check_slice_named_args_errors, "cpu", 1
    yield check_slice_named_args_errors, "gpu", 1


def check_no_slice(device, dtype, batch_size, num_threads):
    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    def make_pipe():
        encoded, _ = fn.readers.caffe(path=caffe_db_folder, random_shuffle=False)
        image = fn.decoders.image(encoded, device="cpu", output_type=types.RGB)
        if device == "gpu":
            image = image.gpu()
        image = fn.cast(image, dtype=dtype)
        sliced1 = fn.slice(image, 0, 3, axes=(2,))
        sliced2 = fn.slice(image, rel_start=(0, 0, 0), rel_end=(1, 1, 1), axis_names="HWC")
        return image, sliced1, sliced2

    pipe = make_pipe()
    for _ in range(3):
        outs = pipe.run()
        nouts = len(outs)
        in_img = as_array(outs[0][0])
        for out_idx in range(1, nouts):
            out_img = as_array(outs[out_idx][0])
            np.testing.assert_array_equal(in_img, out_img)


def test_no_slice():
    batch_size = 4
    num_threads = 3
    for device in ["cpu", "gpu"]:
        for dtype in [types.UINT8, types.UINT16, types.FLOAT]:
            yield check_no_slice, device, dtype, batch_size, num_threads


def check_rel_start_rel_shape(
    device, batch_size, num_threads, get_dynamic_axes=None, args_device="cpu"
):
    image_gen = generator_random_data(
        batch_size, min_sh=(10, 10, 3), max_sh=(100, 100, 3), dtype=np.float32, val_range=[0.0, 1.0]
    )

    # Args GPU only possible with GPU backend
    assert args_device == device or device == "gpu"

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    def make_pipe():
        image = fn.external_source(source=image_gen)
        if device == "gpu":
            image = image.gpu()
        if get_dynamic_axes:
            axes, rel_start, rel_shape = fn.external_source(source=get_dynamic_axes, num_outputs=3)
        else:
            axes = types.Constant(np.array([0, 1], dtype=np.int32), device="cpu")
            rel_start = fn.random.uniform(
                range=(0.1, 0.2), shape=(2,), dtype=types.FLOAT, device=args_device
            )
            rel_shape = fn.random.uniform(
                range=(0.4, 0.6), shape=(2,), dtype=types.FLOAT, device=args_device
            )
        if args_device == "gpu":
            sliced = fn.slice(image, rel_start, rel_shape, axes=axes)
            return image, axes, rel_start, rel_shape, sliced
        else:
            sliced1 = fn.slice(image, rel_start=rel_start, rel_shape=rel_shape, axes=axes)
            sliced2 = fn.slice(image, rel_start, rel_shape, axes=axes)
            return image, axes, rel_start, rel_shape, sliced1, sliced2

    pipe = make_pipe()
    ndim = 3
    for _ in range(3):
        outs = pipe.run()
        for sample_idx in range(batch_size):
            in_img = as_array(outs[0][sample_idx])
            axes = as_array(outs[1][sample_idx])
            naxes = axes.shape[0]
            if naxes == 0:  # Empty axes mean "all axes"
                axes = np.array(range(ndim), dtype=np.int32)
            rel_start = as_array(outs[2][sample_idx])
            rel_shape = as_array(outs[3][sample_idx])
            start = np.zeros([ndim], dtype=np.int32)
            end = np.array([in_img.shape[i] for i in range(ndim)], dtype=np.int32)
            for i in range(len(axes)):
                a = axes[i]
                assert a >= -ndim and a <= (ndim - 1)
                start[a] = roundint(rel_start[i] * in_img.shape[a])
                end[a] = roundint((rel_start[i] + rel_shape[i]) * in_img.shape[a])
            ref_sliced = in_img[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
            # With GPU arguments we don't test named arguments
            for out_idx in range(2 if args_device == "cpu" else 1):
                sliced = as_array(outs[4 + out_idx][sample_idx])
                np.testing.assert_allclose(ref_sliced, sliced)


def check_dynamic_axes(device, batch_size, num_threads, use_negative, use_empty):
    get_dynamic_axes = generator_random_axes_for_3d_input(
        batch_size,
        use_negative=use_negative,
        use_empty=use_empty,
        extra_out_desc=[(0.0, 0.2, np.float32), (0.4, 0.6, np.float32)],  # rel_start  # rel_shape
    )
    check_rel_start_rel_shape(
        device, batch_size, num_threads, get_dynamic_axes=get_dynamic_axes, args_device="cpu"
    )


def test_dynamic_axes():
    batch_size = 10
    num_threads = 3
    for device in ["cpu", "gpu"]:
        yield check_dynamic_axes, device, batch_size, num_threads, False, False


def test_negative_axes():
    batch_size = 10
    num_threads = 3
    for device in ["cpu", "gpu"]:
        yield check_dynamic_axes, device, batch_size, num_threads, True, False


def test_empty_axes():
    batch_size = 10
    num_threads = 3
    for device in ["cpu", "gpu"]:
        yield check_dynamic_axes, device, batch_size, num_threads, False, True


def check_wrong_axes(device, wrong_axes_range=None, named_args=False):
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def make_pipe():
        fake_data = types.Constant(0, shape=[10, 10, 3], dtype=types.FLOAT, device=device)
        axes = fn.random.uniform(range=wrong_axes_range, shape=(2,), dtype=types.INT32)
        rel_start = fn.random.uniform(range=[0.0, 0.3], shape=(2,), dtype=types.FLOAT)
        rel_shape = fn.random.uniform(range=[0.4, 0.6], shape=(2,), dtype=types.FLOAT)
        if named_args:
            sliced = fn.slice(fake_data, rel_start=rel_start, rel_shape=rel_shape, axes=axes)
        else:
            sliced = fn.slice(fake_data, rel_start, rel_shape, axes=axes)
        return sliced

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
            for named_args in [False, True]:
                yield check_wrong_axes, device, wrong_axes_range, named_args


def check_scalar(device):
    batch_size = 5

    def get_data():
        out = [np.random.ranf(size=[1000]).astype(dtype=np.single) for _ in range(batch_size)]
        return out

    @pipeline_def(batch_size=batch_size, num_threads=1, device_id=0)
    def test_pipe():
        data = fn.external_source(source=get_data)
        shape = types.ScalarConstant(10)
        anchor = types.ScalarConstant(5)
        if device != "cpu":
            data = data.gpu()
        sliced = fn.slice(data, start=anchor, shape=shape, axes=[0], device=device)
        return data, sliced, shape, anchor

    pipe = test_pipe()
    ref, data, shape, anchor = pipe.run()
    for sample_idx in range(batch_size):
        d = as_array(data[sample_idx])
        r = as_array(ref[sample_idx])
        s = as_array(shape[sample_idx])
        a = as_array(anchor[sample_idx])
        np.testing.assert_allclose(d, r[a : a + s])


def test_scalar():
    for device in ["cpu", "gpu"]:
        yield check_scalar, device


def test_gpu_args():
    batch_size = 10
    num_threads = 3
    check_rel_start_rel_shape("gpu", batch_size, num_threads, args_device="gpu")


@params(
    ("cpu", False),
    ("gpu", False),
    ("cpu", True),
    ("gpu", True),
)
def test_empty_input(device, use_empty_input):
    @pipeline_def(batch_size=1, num_threads=1, device_id=0 if device == "gpu" else None)
    def make_pipe():
        inp = [] if use_empty_input else [42]
        inp = np.array(inp, dtype="int")
        x = types.Constant(inp, device=device)
        anchor = 0 if use_empty_input else 1
        return fn.slice(x, anchor, 0, axes=[0])

    p = make_pipe()
    (o,) = p.run()
    if device == "gpu":
        o = o.as_cpu()
    assert np.array_equal(o[0], [])


def test_wrong_arg_backend():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def make_pipe():
        fake_data = types.Constant(0, shape=[10, 10, 3], dtype=types.FLOAT, device="cpu")
        rel_start = fn.random.uniform(range=[0.0, 0.3], shape=(2,), dtype=types.FLOAT, device="gpu")
        rel_shape = fn.random.uniform(range=[0.4, 0.6], shape=(2,), dtype=types.FLOAT, device="gpu")
        sliced = fn.slice(fake_data, rel_start, rel_shape, device="cpu")
        return sliced

    with assert_raises(RuntimeError, glob='is stored on incompatible device "gpu"'):
        p = make_pipe()
        p.run()


def test_wrong_backend_named_args():
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def make_pipe():
        fake_data = types.Constant(0, shape=[10, 10, 3], dtype=types.FLOAT, device="cpu")
        rel_start = fn.random.uniform(range=[0.0, 0.3], shape=(2,), dtype=types.FLOAT, device="gpu")
        rel_shape = fn.random.uniform(range=[0.4, 0.6], shape=(2,), dtype=types.FLOAT, device="gpu")
        sliced = fn.slice(fake_data, rel_start=rel_start, rel_shape=rel_shape, device="cpu")
        return sliced

    with assert_raises(ValueError, glob="Invalid device \"gpu\" for argument 'rel*'"):
        p = make_pipe()
        p.run()
