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
from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator

def transpose_func(image):
    return image.transpose((1, 0, 2))

class TransposePipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(TransposePipeline, self).__init__(batch_size,
                                                num_threads,
                                                device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.transpose = ops.Transpose(device = self.device,
                                       perm = (1, 0, 2),
                                       transpose_layout = False)

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.transpose(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

class PythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(PythonOpPipeline, self).__init__(batch_size,
                                               num_threads,
                                               device_id,
                                               exec_async=False,
                                               exec_pipelined=False)
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.oper = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.oper(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

def check_transpose_vs_numpy(device, batch_size, shape):
    eii1 = RandomDataIterator(batch_size, shape=shape)
    eii2 = RandomDataIterator(batch_size, shape=shape)
    compare_pipelines(TransposePipeline(device, batch_size, types.NHWC, iter(eii1)),
                      PythonOpPipeline(transpose_func, batch_size, types.NHWC, iter(eii2)),
                      batch_size=batch_size, N_iterations=10)

def test_transpose_vs_numpy():
    # TODO(janton) Transpose not implemented for CPU
    for device in {'gpu'}:
        for batch_size in {1, 3}:
            for shape in {(2048, 512, 1), (2048, 512, 3), (2048, 512, 8)}:
                yield check_transpose_vs_numpy, device, batch_size, shape
