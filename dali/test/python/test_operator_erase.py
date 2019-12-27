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
                 anchor, shape, axis_names,
                 num_threads=1, device_id=0, num_gpus=1):
        super(ErasePipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.erase = ops.Erase(device = self.device,
                               anchor = anchor,
                               shape = shape,
                               axis_names = axis_names)

    def define_graph(self):
        self.data = self.inputs()
        random_data = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.erase(random_data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def erase_func(anchor, shape, axis_names, layout, image):
    assert layout == "HWC"
    assert len(anchor) == len(shape)
    assert axis_names == "HW"
    assert len(shape) % len(axis_names) == 0

    assert len(image.shape) == 3
    H = image.shape[0]
    W = image.shape[1]

    nregions = int(len(shape) / len(axis_names))
    if layout == "HWC":
        for n in range(nregions):
            start_y = anchor[n*2+0]
            erase_h = shape[n*2+0]
            end_y = start_y + erase_h
            assert H >= end_y

            start_x = anchor[n*2+1]
            erase_w = shape[n*2+1]
            end_x = start_x + erase_w
            assert W >= end_x

            image[start_y:end_y, start_x:end_x, :] = 0
        return image
    else:
        assert(False)  # should not happen

class ErasePythonPipeline(Pipeline):
    def __init__(self, function, batch_size, data_layout, iterator,
                 anchor, shape, axis_names,
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

        function = partial(erase_func, anchor, shape, axis_names, data_layout)

        self.erase = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.erase(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)


def check_operator_erase_vs_python(device, batch_size, input_shape,
                                   anchor, shape, axis_names,
                                   layout = "HWC"):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        ErasePipeline(device, batch_size, "HWC", iter(eii1),
                      anchor=anchor, shape=shape, axis_names=axis_names),
        ErasePythonPipeline(device, batch_size, "HWC", iter(eii2),
                            anchor=anchor, shape=shape, axis_names=axis_names),
        batch_size=batch_size, N_iterations=5, eps=1e-04)


def test_operator_erase_vs_python():
    layouts = ["HWC"]
    input_shapes = {
        "HWC" : [(60, 80, 3)],
    }

    axis_names = "HW"

    anchor_1_region = (4, 10)
    shape_1_region = (40, 50)

    anchor_2_region = (4, 2, 3, 4)     # y0, x0, y1, x1
    shape_2_region = (50, 10, 10, 50)  # h0, w0, h1, w1

    regions = [(anchor_1_region, shape_1_region), (anchor_2_region, shape_2_region)]

    for device in ['cpu']:
        for batch_size in [1, 8]:
            for input_layout in layouts:
                for anchor, shape in regions:
                    for input_shape in input_shapes[input_layout]:
                        assert len(input_layout) == len(input_shape)
                        yield check_operator_erase_vs_python, device, batch_size, input_shape, \
                            anchor, shape, axis_names, input_layout
