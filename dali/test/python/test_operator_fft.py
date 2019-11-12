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
from functools import partial
from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator

class FftPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, axis, nfft, num_threads=1, device_id=0):
        super(FftPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.fft = ops.PowerSpectrum(device=self.device, axis=axis, nfft=nfft)

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.fft(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def fft_numpy(nfft, axis, waveform):
    fft_out = np.fft.fft(waveform, axis=axis, n=nfft)
    fft_pow = fft_out.real ** 2 + fft_out.imag ** 2
    shape = waveform.shape

    out_shape = list(shape)
    out_shape[axis] = nfft/2+1
    out_shape = tuple(out_shape)

    if len(out_shape) == 2:
        out = fft_pow[0:out_shape[0], 0:out_shape[1]]
    elif len(out_shape) == 3:
        out = fft_pow[0:out_shape[0], 0:out_shape[1], 0:out_shape[2]]
    return out

class FftNumpyPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, axis, nfft,
                 num_threads=1, device_id=0):
        super(FftNumpyPipeline, self).__init__(
              batch_size, num_threads, device_id,
              seed=12345, exec_async=False, exec_pipelined=False)
        self.device = "cpu"
        self.iterator = iterator
        self.inputs = ops.ExternalSource()

        function = partial(fft_numpy, nfft, axis)
        self.fft = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.fft(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def check_operator_fft_power_spectrum(device, batch_size, input_shape, nfft, axis):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        FftPipeline(device, batch_size, iter(eii1), axis=axis, nfft=nfft),
        FftNumpyPipeline(device, batch_size, iter(eii2), axis=axis, nfft=nfft),
        batch_size=batch_size, N_iterations=5, eps=1e-04)

def test_operator_fft_power_spectrum():
    for device in ['cpu']:
        for batch_size in [3]:
            for nfft, axis, shape in [(16, 1, (2, 16)),
                                      (1024, 1, (1, 1024)),
                                      (128, 1, (1, 100)),
                                      (16, 0, (16, 2)),
                                      (8, 1, (2, 8, 2))]:
                yield check_operator_fft_power_spectrum, device, batch_size, shape, nfft, axis

if __name__ == "__main__":
    check_operator_fft_power_spectrum(device='cpu', batch_size=3, input_shape=(2, 1024),
                                      nfft=1024, axis=1)