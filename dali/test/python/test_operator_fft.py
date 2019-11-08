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
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator

class FftPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, axis=-1, num_threads=1, device_id=0):
        super(FftPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.fft = ops.Fft(device=self.device, axis=axis)

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.fft(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def check_operator_fft_complex_spectrum(device, batch_size, input_shape):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        FftPipeline(device, batch_size, iter(eii1)),
        FftPipeline(device, batch_size, iter(eii2)),
        batch_size=batch_size, N_iterations=5)

def test_operator_fft_complex_spectrum():
    for device in ['cpu']:
        for batch_size in [3]:
            for shape in [(1, 2, 1024)]:
                yield check_operator_fft_complex_spectrum, device, batch_size, shape

if __name__ == "__main__":
    check_operator_fft_complex_spectrum(device='cpu', batch_size=2, input_shape=(2, 1024))