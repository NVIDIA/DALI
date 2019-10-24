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
import os

from test_utils import RandomlyShapedDataIterator

class PadSynthDataPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator,
                 layout="HWC", num_threads=1, device_id=0, num_gpus=1, axes=()):
        super(PadSynthDataPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=1234)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.pad = ops.Pad(device = self.device, axes=axes)

    def define_graph(self):
        self.data = self.inputs()
        input_data = self.data
        data = input_data.gpu() if self.device == 'gpu' else input_data
        out = self.pad(data)
        return input_data, out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def check_pad_synth_data(device, batch_size, input_max_shape, axes):
    eii = RandomlyShapedDataIterator(batch_size, max_shape=input_max_shape)
    pipe = PadSynthDataPipeline(device, batch_size, iter(eii), axes=axes)
    pipe.build()
    actual_axes = axes if len(axes) > 0 else range(len(input_max_shape))
    assert(len(actual_axes)>0)
    for k in range(5):
        out1, out2 = pipe.run()

        out1_data = out1.as_cpu() if isinstance(out1.at(0), dali.backend_impl.TensorGPU) else out1
        max_shape = [0] * len(input_max_shape)
        for i in range(batch_size):
            input_shape = out1_data.at(i).shape
            for dim in actual_axes:
                if input_shape[dim] > max_shape[dim]:
                    max_shape[dim] = input_shape[dim]

        out2_data = out2.as_cpu() if isinstance(out2.at(0), dali.backend_impl.TensorGPU) else out2
        for i in range(batch_size):
            output_shape = out2_data.at(i).shape
            for dim in range(len(max_shape)):
                if dim in actual_axes:
                    assert(output_shape[dim] == max_shape[dim])

def test_slice_synth_data_vs_numpy():
    for device in ["gpu"]:
        for batch_size in {1, 8, 100}:
            for input_max_shape, axes in \
                [((200,400,3), (0,)),
                 ((200,400,3), (1,)),
                 ((200,400,3), (0,1)),
                 ((200,400,3), ()),
                 ((200,400,3), [])]:
                yield check_pad_synth_data, device, batch_size, input_max_shape, axes
