# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
from functools import partial

from test_utils import compare_pipelines
from test_utils import RandomDataIterator


class ErasePipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        layout,
        iterator,
        anchor,
        shape,
        axis_names,
        axes,
        fill_value,
        normalized_anchor=False,
        normalized_shape=False,
        num_threads=1,
        device_id=0,
        num_gpus=1,
    ):
        super(ErasePipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()

        if isinstance(fill_value, RandomDataIterator):
            self.fill_value_iterator = fill_value
            self.fill_value_inputs = ops.ExternalSource()
            fill_value = None
        else:
            self.fill_value_iterator = None

        self.erase = ops.Erase(
            device=self.device,
            anchor=anchor,
            shape=shape,
            axis_names=axis_names,
            axes=axes,
            fill_value=fill_value,
            normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape,
        )

    def define_graph(self):
        self.data = self.inputs()
        random_data = self.data.gpu() if self.device == "gpu" else self.data
        if self.fill_value_iterator is not None:
            self.fill_value_data = self.fill_value_inputs()
            out = self.erase(random_data, fill_value=self.fill_value_data)
        else:
            out = self.erase(random_data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)
        if self.fill_value_iterator is not None:
            fill_value_data = self.fill_value_iterator.next()
            self.feed_input(self.fill_value_data, fill_value_data)


def get_axes(layout, axis_names):
    axes = []
    for axis_name in axis_names:
        axis_idx = layout.find(axis_name)
        assert axis_idx >= 0
        axes.append(axis_idx)
    return axes


def get_regions(in_shape, axes, arg_anchor, arg_shape):
    assert len(arg_shape) % len(axes) == 0
    nregions = int(len(arg_shape) / len(axes))
    region_length = int(len(arg_shape) / nregions)
    starts = []
    ends = []
    for region_idx in range(nregions):
        start_i = [0] * len(in_shape)
        end_i = list(in_shape)
        for k in range(region_length):
            axis = axes[k]
            anchor_val = arg_anchor[region_idx * region_length + k]
            shape_val = arg_shape[region_idx * region_length + k]
            end_val = anchor_val + shape_val
            start_i[axis] = anchor_val
            end_i[axis] = end_val
        starts.append(start_i)
        ends.append(end_i)
    return (starts, ends)


def erase_func(anchor, shape, axis_names, axes, layout, fill_value, image):
    assert len(anchor) == len(shape)

    if not axes:
        axes = get_axes(layout, axis_names)

    if fill_value is None:
        fill_value = 0

    roi_starts, roi_ends = get_regions(image.shape, axes, anchor, shape)
    assert len(roi_starts) == len(roi_ends)
    for region_idx in range(len(roi_starts)):
        start = roi_starts[region_idx]
        end = roi_ends[region_idx]
        assert len(start) == len(end)
        if len(start) == 3:
            image[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = fill_value
        elif len(start) == 4:
            image[start[0] : end[0], start[1] : end[1], start[2] : end[2], start[3] : end[3]] = (
                fill_value
            )
        else:
            assert False
    return image


class ErasePythonPipeline(Pipeline):
    def __init__(
        self,
        function,
        batch_size,
        data_layout,
        iterator,
        anchor,
        shape,
        axis_names,
        axes,
        fill_value,
        erase_func=erase_func,
        num_threads=1,
        device_id=0,
    ):
        super(ErasePythonPipeline, self).__init__(
            batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False
        )
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_layout = data_layout

        if isinstance(fill_value, RandomDataIterator):
            self.fill_value_iterator = fill_value
            self.fill_value_inputs = ops.ExternalSource()
            fill_value = None
            function = partial(erase_func, anchor, shape, axis_names, axes, data_layout)
        else:
            self.fill_value_iterator = None
            function = partial(erase_func, anchor, shape, axis_names, axes, data_layout, fill_value)

        self.erase = ops.PythonFunction(function=function, output_layouts=data_layout)

    def define_graph(self):
        self.data = self.inputs()
        if self.fill_value_iterator is not None:
            self.fill_value_data = self.fill_value_inputs()
            out = self.erase(self.fill_value_data, self.data)
        else:
            out = self.erase(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)
        if self.fill_value_iterator is not None:
            fill_value_data = self.fill_value_iterator.next()
            self.feed_input(self.fill_value_data, fill_value_data)


def check_operator_erase_vs_python(
    device, batch_size, input_shape, anchor, shape, axis_names, axes, input_layout, fill_value
):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)

    fill_value_arg1 = fill_value
    fill_value_arg2 = fill_value
    if fill_value == "random":
        fill_eii1 = RandomDataIterator(batch_size, shape=input_shape[-1:], dtype=np.float32)
        fill_eii2 = RandomDataIterator(batch_size, shape=input_shape[-1:], dtype=np.float32)
        fill_value_arg1 = iter(fill_eii1)
        fill_value_arg2 = iter(fill_eii2)

    compare_pipelines(
        ErasePipeline(
            device,
            batch_size,
            input_layout,
            iter(eii1),
            anchor=anchor,
            shape=shape,
            axis_names=axis_names,
            axes=axes,
            fill_value=fill_value_arg1,
        ),
        ErasePythonPipeline(
            device,
            batch_size,
            input_layout,
            iter(eii2),
            anchor=anchor,
            shape=shape,
            axis_names=axis_names,
            axes=axes,
            fill_value=fill_value_arg2,
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1e-04,
        expected_layout=input_layout,
    )


def test_operator_erase_vs_python():
    # layout, shape, axis_names, axes, anchor, shape, fill_value
    rois = [
        ("HWC", (60, 80, 3), "HW", None, (4, 10), (40, 50), 0),
        ("HWC", (60, 80, 3), "HW", None, (4, 10), (40, 50), None),
        ("HWC", (60, 80, 3), "HW", None, (4, 2, 3, 4), (50, 10, 10, 50), -1),
        ("HWC", (60, 80, 3), "HW", None, (4, 2, 3, 4), (50, 10, 10, 50), (118, 185, 0)),
        ("HWC", (60, 80, 3), "HW", None, (4, 2, 3, 4), (50, 10, 10, 50), "random"),
        ("HWC", (60, 80, 3), "H", None, (4,), (7,), 0),
        ("HWC", (60, 80, 3), "H", None, (4, 15), (7, 8), 0),
        ("HWC", (60, 80, 3), "W", None, (4,), (7,), 0),
        ("HWC", (60, 80, 3), "W", None, (4, 15), (7, 8), 0),
        ("HWC", (60, 80, 3), "W", None, (4, 15), (7, 8), "random"),
        ("HWC", (60, 80, 3), None, (0, 1), (4, 10), (40, 50), 0),
        ("HWC", (60, 80, 3), None, (0, 1), (4, 2, 3, 4), (50, 10, 10, 50), 0),
        ("HWC", (60, 80, 3), None, (0,), (4,), (7,), 0),
        ("HWC", (60, 80, 3), None, (0,), (4, 15), (7, 8), 0),
        ("HWC", (60, 80, 3), None, (1,), (4,), (7,), 0),
        ("HWC", (60, 80, 3), None, (1,), (4, 15), (7, 8), 0),
        ("HWC", (60, 80, 3), None, (1,), (4, 15), (7, 8), "random"),
        ("DHWC", (10, 60, 80, 3), "DHW", None, (2, 4, 15), (3, 7, 8), 0),
        ("HWC", (60, 80, 1), "HW", None, (4, 15), (7, 8), 0),
        ("XYZ", (60, 80, 3), "XY", None, (4, 10), (40, 50), -1),
    ]

    for device in ["cpu"]:
        for batch_size in [1, 8]:
            for input_layout, input_shape, axis_names, axes, anchor, shape, fill_value in rois:
                assert len(input_layout) == len(input_shape)
                assert len(anchor) == len(shape)
                if axis_names:
                    assert axes is None
                    assert len(anchor) % len(axis_names) == 0
                else:
                    assert len(axes) > 0
                    assert len(anchor) % len(axes) == 0

                yield (
                    check_operator_erase_vs_python,
                    device,
                    batch_size,
                    input_shape,
                    anchor,
                    shape,
                    axis_names,
                    axes,
                    input_layout,
                    fill_value,
                )


def check_operator_erase_with_normalized_coords(
    device,
    batch_size,
    input_shape,
    anchor,
    shape,
    axis_names,
    input_layout,
    anchor_norm,
    shape_norm,
    normalized_anchor,
    normalized_shape,
):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        ErasePipeline(
            device,
            batch_size,
            input_layout,
            iter(eii1),
            anchor=anchor_norm,
            shape=shape_norm,
            normalized_anchor=normalized_anchor,
            normalized_shape=normalized_shape,
            axis_names=axis_names,
            axes=None,
            fill_value=0,
        ),
        ErasePipeline(
            device,
            batch_size,
            input_layout,
            iter(eii2),
            anchor=anchor,
            shape=shape,
            axis_names=axis_names,
            axes=None,
            fill_value=0,
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1e-04,
    )


def test_operator_erase_with_normalized_coords():
    # layout, shape, axis_names, anchor, shape, anchor_norm, shape_norm
    rois = [
        (
            "HWC",
            (60, 80, 3),
            "HW",
            (4, 10),
            (40, 50),
            (4 / 60.0, 10 / 80.0),
            (40 / 60.0, 50 / 80.0),
            0,
        ),
        (
            "HWC",
            (60, 80, 3),
            "HW",
            (4, 10),
            (40, 50),
            (4 / 60.0, 10 / 80.0),
            (40 / 60.0, 50 / 80.0),
            -1,
        ),
        (
            "HWC",
            (60, 80, 3),
            "HW",
            (4, 10),
            (40, 50),
            (4 / 60.0, 10 / 80.0),
            (40 / 60.0, 50 / 80.0),
            (118, 186, 0),
        ),
        (
            "HWC",
            (60, 80, 3),
            "HW",
            (4, 2, 3, 4),
            (50, 10, 10, 50),
            (4 / 60.0, 2 / 80.0, 3 / 60.0, 4 / 80.0),
            (50 / 60.0, 10 / 80.0, 10 / 60.0, 50 / 80.0),
            0,
        ),
        ("HWC", (60, 80, 3), "H", (4,), (7,), (4 / 60.0,), (7 / 60.0,), 0),
        (
            "DHWC",
            (10, 60, 80, 3),
            "DHW",
            (2, 4, 10),
            (5, 40, 50),
            (2 / 10.0, 4 / 60.0, 10 / 80.0),
            (5 / 10.0, 40 / 60.0, 50 / 80.0),
            0,
        ),
        (
            "HWC",
            (60, 80, 1),
            "WH",
            (10, 4),
            (50, 40),
            (10 / 80.0, 4 / 60.0),
            (50 / 80.0, 40 / 60.0),
            0,
        ),
        ("XYZ", (60, 80, 3), "X", (4,), (7,), (4 / 60.0,), (7 / 60.0,), -10),
    ]

    for device in ["cpu", "gpu"]:
        for batch_size in [1, 8]:
            for (
                input_layout,
                input_shape,
                axis_names,
                anchor,
                shape,
                anchor_norm,
                shape_norm,
                fill_value,
            ) in rois:
                assert len(input_layout) == len(input_shape)
                assert len(anchor) == len(shape)
                assert len(anchor) % len(axis_names) == 0
                for normalized_anchor, normalized_shape in [
                    (True, True),
                    (True, False),
                    (False, True),
                ]:
                    anchor_norm_arg = anchor_norm if normalized_anchor else anchor
                    shape_norm_arg = shape_norm if normalized_shape else shape
                    yield (
                        check_operator_erase_with_normalized_coords,
                        device,
                        batch_size,
                        input_shape,
                        anchor,
                        shape,
                        axis_names,
                        input_layout,
                        anchor_norm_arg,
                        shape_norm_arg,
                        normalized_anchor,
                        normalized_shape,
                    )


def test_operator_erase_with_out_of_bounds_roi_coords():
    device = "cpu"
    batch_size = 8
    input_layout = "HWC"
    input_shape = (60, 80, 3)
    axis_names = "HW"
    anchor_arg = (4, 10, 10, 4)
    shape_arg = (40, 50, 50, 40)
    # second region is completely out of bounds
    anchor_norm_arg = (4 / 60.0, 10 / 80.0, 2000, 2000, 10 / 60.0, 4 / 80.0)
    shape_norm_arg = (40 / 60.0, 50 / 80.0, 200, 200, 50 / 60.0, 40 / 80.0)
    yield (
        check_operator_erase_with_normalized_coords,
        device,
        batch_size,
        input_shape,
        anchor_arg,
        shape_arg,
        axis_names,
        input_layout,
        anchor_norm_arg,
        shape_norm_arg,
        True,
        True,
    )
