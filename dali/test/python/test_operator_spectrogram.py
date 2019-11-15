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
import torchaudio.transforms as torchaudio_transforms
import torch as torch

class SpectrogramPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, nfft, window_length, window_step,
                 num_threads=1, device_id=0):
        super(SpectrogramPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.fft = ops.Spectrogram(device = self.device,
                                   nfft = nfft,
                                   window_length = window_length,
                                   window_step = window_step)

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.fft(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def spectrogram_func(nfft, win_len, win_step, input):
    def rect_win(n):
        return torch.ones([n], dtype=torch.float32)
    spectrogram = torchaudio_transforms.Spectrogram(
        n_fft=nfft, win_length=win_len, hop_length=win_step, window_fn=rect_win)
    waveform = torch.FloatTensor(input)
    out = spectrogram.forward(waveform)
    print(out)
    return out

class SpectrogramPythonPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, nfft, window_length, window_step,
                 num_threads=1, device_id=0):
        super(SpectrogramPythonPipeline, self).__init__(
              batch_size, num_threads, device_id,
              seed=12345, exec_async=False, exec_pipelined=False)
        self.device = "cpu"
        self.iterator = iterator
        self.inputs = ops.ExternalSource()

        function = partial(spectrogram_func, nfft, window_length, window_step)
        self.spectrogram = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.spectrogram(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def check_operator_spectrogram_vs_python(device, batch_size, input_shape,
                                         nfft, window_length, window_step):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        SpectrogramPipeline(device, batch_size, iter(eii1), nfft=nfft,
                            window_length=window_length, window_step=window_step),
        SpectrogramPythonPipeline(device, batch_size, iter(eii2), nfft=nfft,
                                  window_length=window_length, window_step=window_step),
        batch_size=batch_size, N_iterations=5, eps=1e-04)

def test_operator_spectrogram_vs_python():
    for device in ['cpu']:
        for batch_size in [3]:
            for nfft, window_length, window_step, shape in [(256, 256, 128, (1, 4096))]:
                yield check_operator_spectrogram_vs_python, device, batch_size, shape, \
                    nfft, window_length, window_step

if __name__ == "__main__":
    check_operator_spectrogram_vs_python(device='cpu', batch_size=3, input_shape=(1, 16),
                                         nfft=16, window_length=16, window_step=16)