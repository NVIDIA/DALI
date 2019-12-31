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
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
from functools import partial

from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path

class ErasePipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator,
                 anchor, shape, axis_names, axes,
                 normalized_anchor=False, normalized_shape=False,
                 num_threads=1, device_id=0, num_gpus=1):
        super(ErasePipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        if axis_names:
            self.erase = ops.Erase(device = self.device,
                                   anchor = anchor,
                                   shape = shape,
                                   axis_names = axis_names,
                                   normalized_anchor=normalized_anchor,
                                   normalized_shape=normalized_shape)
        else:
            self.erase = ops.Erase(device = self.device,
                                   anchor = anchor,
                                   shape = shape,
                                   axes = axes,
                                   normalized_anchor=normalized_anchor,
                                   normalized_shape=normalized_shape)


    def define_graph(self):
        self.data = self.inputs()
        random_data = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.erase(random_data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def erase_func(anchor, shape, axis_names, axes, layout, image):
    assert layout == "HWC"
    assert len(anchor) == len(shape)

    if not axis_names:
        assert(axes is not None)
        if axes == (0, 1):
            axis_names = "HW"
        elif axes == (1, 0):
            axis_names = "WH"
        elif axes == (0,):
            axis_names = "H"
        elif axes == (1,):
            axis_names = "W"
        else:
            assert(False)

    assert len(shape) % len(axis_names) == 0
    assert len(image.shape) == 3

    H = image.shape[0]
    W = image.shape[1]

    nregions = int(len(shape) / len(axis_names))
    region_length = int(len(shape) / nregions)
    if layout == "HWC":
        for n in range(nregions):
            start_0 = anchor[n*region_length+0]
            shape_0 = shape[n*region_length+0]
            end_0 = start_0 + shape_0

            if region_length > 1:
                start_1 = anchor[n*2+1]
                shape_1 = shape[n*2+1]
                end_1 = start_1 + shape_1

            if axis_names == "H":
                assert H >= end_0
                image[start_0:end_0, :, :] = 0
            elif axis_names == "W":
                assert W >= end_0
                image[:, start_0:end_0, :] = 0
            elif axis_names == "HW":
                assert H >= end_0
                assert W >= end_1
                image[start_0:end_0, start_1:end_1, :] = 0
            elif axis_names == "WH":
                assert H >= end_1
                assert W >= end_0
                image[start_1:end_1, start_0:end_0, :] = 0
            else:
                assert(False)  # should not happen
        return image
    else:
        assert(False)  # should not happen

class ErasePythonPipeline(Pipeline):
    def __init__(self, function, batch_size, data_layout, iterator,
                 anchor, shape, axis_names, axes,
                 erase_func=erase_func,
                 num_threads=1, device_id=0):
        super(ErasePythonPipeline, self).__init__(batch_size,
                                                  num_threads,
                                                  device_id,
                                                  exec_async=False,
                                                  exec_pipelined=False)
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_layout = data_layout

        function = partial(erase_func, anchor, shape, axis_names, axes, data_layout)

        self.erase = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.erase(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)


def check_operator_erase_vs_python(device, batch_size, input_shape,
                                   anchor, shape, axis_names, axes, input_layout):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        ErasePipeline(device, batch_size, input_layout, iter(eii1),
                      anchor=anchor, shape=shape, axis_names=axis_names, axes=axes),
        ErasePythonPipeline(device, batch_size, input_layout, iter(eii2),
                            anchor=anchor, shape=shape, axis_names=axis_names, axes=axes),
        batch_size=batch_size, N_iterations=5, eps=1e-04)


def test_operator_erase_vs_python():
    # layout, shape, axis_names, axes, anchor, shape
    rois = [("HWC", (60, 80, 3), "HW", None, (4, 10), (40, 50)),
            ("HWC", (60, 80, 3), "HW", None, (4, 2, 3, 4), (50, 10, 10, 50)),
            ("HWC", (60, 80, 3), "H", None, (4,), (7,)),
            ("HWC", (60, 80, 3), "H", None, (4, 15), (7, 8)),
            ("HWC", (60, 80, 3), "W", None, (4,), (7,)),
            ("HWC", (60, 80, 3), "W", None, (4, 15), (7, 8)),
            ("HWC", (60, 80, 3), None, (0, 1), (4, 10), (40, 50)),
            ("HWC", (60, 80, 3), None, (0, 1), (4, 2, 3, 4), (50, 10, 10, 50)),
            ("HWC", (60, 80, 3), None, (0,), (4,), (7,)),
            ("HWC", (60, 80, 3), None, (0,), (4, 15), (7, 8)),
            ("HWC", (60, 80, 3), None, (1,), (4,), (7,)),
            ("HWC", (60, 80, 3), None, (1,), (4, 15), (7, 8))]

    for device in ['cpu']:
        for batch_size in [1, 8]:
            for input_layout, input_shape, axis_names, axes, anchor, shape in rois:
                assert len(input_layout) == len(input_shape)
                assert len(anchor) == len(shape)
                if axis_names:
                    assert axes is None
                    assert len(anchor) % len(axis_names) == 0
                else:
                    assert len(axes) > 0
                    assert len(anchor) % len(axes) == 0

                yield check_operator_erase_vs_python, device, batch_size, input_shape, \
                    anchor, shape, axis_names, axes, input_layout


def check_operator_erase_with_normalized_coords(device, batch_size, input_shape,
                                                anchor, shape, axis_names, input_layout,
                                                anchor_norm, shape_norm, normalized_anchor, normalized_shape):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        ErasePipeline(device, batch_size, input_layout, iter(eii1),
                      anchor=anchor_norm, shape=shape_norm,
                      normalized_anchor=normalized_anchor, normalized_shape=normalized_shape,
                      axis_names=axis_names, axes=None),
        ErasePipeline(device, batch_size, input_layout, iter(eii1),
                      anchor=anchor, shape=shape, axis_names=axis_names, axes=None),
        batch_size=batch_size, N_iterations=5, eps=1e-04)


def test_operator_erase_with_normalized_coords():
    # layout, shape, axis_names, anchor, shape, anchor_norm, shape_norm
    rois = [("HWC", (60, 80, 3), "HW", (4, 10), (40, 50), (4/60.0, 10/80.0), (40/60.0, 50/80.0)),
            ("HWC", (60, 80, 3), "HW", (4, 2, 3, 4), (50, 10, 10, 50), (4/60.0, 2/80.0, 3/60.0, 4/80.0), (50/60.0, 10/80.0, 10/60.0, 50/80.0)),
            ("HWC", (60, 80, 3), "H", (4,), (7,), (4/60.0,), (7/60.0,))]

    for device in ['cpu']:
        for batch_size in [1, 8]:
            for input_layout, input_shape, axis_names, anchor, shape, anchor_norm, shape_norm in rois:
                assert len(input_layout) == len(input_shape)
                assert len(anchor) == len(shape)
                assert len(anchor) % len(axis_names) == 0
                for (normalized_anchor, normalized_shape) in [(True, True), (True, False), (False, True)]:
                    anchor_norm_arg = anchor_norm if normalized_anchor else anchor
                    shape_norm_arg = shape_norm if normalized_shape else shape
                    yield check_operator_erase_with_normalized_coords, device, batch_size, input_shape, \
                        anchor, shape, axis_names, input_layout, anchor_norm_arg, shape_norm_arg, \
                        normalized_anchor, normalized_shape

def test_operator_erase_with_out_of_bounds_roi_coords():
    device = 'cpu'
    batch_size = 8
    input_layout = "HWC"
    input_shape = (60, 80, 3)
    axis_names = "HW"
    anchor_arg = (4, 10, 10, 4)
    shape_arg = (40, 50, 50, 40)
    anchor_norm_arg = (4/60.0, 10/80.0, 2000, 2000, 10/60.0, 4/80.0)  # second region is completely out of bounds
    shape_norm_arg = (40/60.0, 50/80.0, 200, 200, 50/60.0, 40/80.0)
    yield check_operator_erase_with_normalized_coords, device, batch_size, input_shape, anchor_arg, shape_arg, \
        axis_names, input_layout, anchor_norm_arg, shape_norm_arg, True, True
