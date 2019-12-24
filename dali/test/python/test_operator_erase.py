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
    def __init__(self, device, batch_size, layout, iterator, num_threads=1, device_id=0, num_gpus=1):
        super(ErasePipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.erase = ops.Erase(device = self.device,
                               anchor = (0, 0, 10, 10),
                               shape = (10, 10, 8, 8))

    def define_graph(self):
        self.data = self.inputs()
        random_data = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.erase(random_data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def check_erase_vs_numpy(device, batch_size, input_layout, input_shape):
    eii1 = RandomDataIterator(batch_size, shape=input_shape)
    pipe = ErasePipeline(device, batch_size, input_layout, iter(eii1)) 
    pipe.build()
    pipe.run()

check_erase_vs_numpy('cpu', 3, 'HWC', (30, 30, 3))

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
