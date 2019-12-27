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
                 start_y, erase_h, start_x, erase_w,
                 num_threads=1, device_id=0, num_gpus=1):
        super(ErasePipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.erase = ops.Erase(device = self.device,
                               anchor = (start_y, start_x),
                               shape = (erase_h, erase_w),
                               axis_names = "HW")

    def define_graph(self):
        self.data = self.inputs()
        random_data = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.erase(random_data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

def erase_func(start_y, erase_h, start_x, erase_w, layout, image):
    assert layout == "HWC"  # TODO(janton): extend to other layouts
    assert len(image.shape) == 3
    H = image.shape[0]
    W = image.shape[1]

    # start_y = int(np.float32(crop_y) * np.float32(H - crop_h) + np.float32(0.5))
    end_y = start_y + erase_h
    # start_x = int(np.float32(crop_x) * np.float32(W - crop_w) + np.float32(0.5))
    end_x = start_x + erase_w

    assert H >= end_y
    assert W >= end_x

    if layout == "HWC":
        image[start_y:end_y, start_x:end_x, :] = 0
        return image
    else:
        assert(False)  # should not happen

class ErasePythonPipeline(Pipeline):
    def __init__(self, function, batch_size, data_layout, iterator,
                 start_y, erase_h, start_x, erase_w,
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

        function = partial(erase_func, start_y, erase_h, start_x, erase_w, data_layout)

        self.erase = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.erase(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)


def check_operator_erase_vs_python(device, batch_size, input_shape,
                                   start_y, erase_h, start_x, erase_w,
                                   layout = "HWC"):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        ErasePipeline(device, batch_size, "HWC", iter(eii1),
                      start_y=start_y, erase_h=erase_h, start_x=start_x, erase_w=erase_w),
        ErasePythonPipeline(device, batch_size, "HWC", iter(eii2),
                            start_y=start_y, erase_h=erase_h, start_x=start_x, erase_w=erase_w),
        batch_size=batch_size, N_iterations=5, eps=1e-04)


check_operator_erase_vs_python('cpu', 3, input_shape=(10, 10, 3),
                               start_y=1, erase_h=8, start_x=1, erase_w=8,
                               layout="HWC")

def test_erase_vs_numpy():
    layouts = ["HWC", "FHWC", "DHWC", "FDHWC"]
    input_shapes = {
        "HWC" : [(60, 80, 3)],
        "FHWC" : [(3, 60, 80, 3)],
        "DHWC" : [(10, 60, 80, 3)],
        "FDHWC" : [(3, 10, 60, 80, 3)]
    }

    for device in ['cpu']:
        for batch_size in [1, 8]:
            for input_layout in layouts:
                for input_shape in input_shapes[input_layout]:
                    assert len(input_layout) == len(input_shape)
                    yield check_erase_vs_numpy, device, batch_size, input_layout, input_shape
