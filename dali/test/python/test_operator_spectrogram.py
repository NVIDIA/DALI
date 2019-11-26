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
from test_utils import DataIterator
import librosa as librosa
import math

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
                                   window_step = window_step,
                                   power = 2)

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.fft(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def spectrogram_func_torch(nfft, win_len, win_step, input):
    def hann_win(n):
        hann = torch.ones([n], dtype=torch.float32)
        a = (2.0 * math.pi / n)
        for t in range(n):
            phase = a * (t + 0.5)
            hann[t] = 0.5 * (1.0 - math.cos(phase))
        return hann

    waveform = torch.FloatTensor(input)
    assert nfft == win_len
    spectrogram = torchaudio_transforms.Spectrogram(
        n_fft=nfft, win_length=win_len, hop_length=win_step, power=2, window_fn=hann_win)
    out = spectrogram.forward(waveform)
    print(out)
    return out


def spectrogram_func_librosa(nfft, win_len, win_step, input_data):
    def hann_win(n):
        hann = np.ones([n], dtype=np.float32)
        a = (2.0 * math.pi / n)
        for t in range(n):
            phase = a * (t + 0.5)
            hann[t] = 0.5 * (1.0 - math.cos(phase))
        return hann

    has_channels_dim = len(input_data.shape) == 2

    if has_channels_dim:
        input_data = np.squeeze(input_data, axis=0)

    out = np.abs(
        librosa.stft(y=input_data, n_fft=nfft, hop_length=win_step, window=hann_win))**2

    # Alternative way to calculate the spectrogram:
    # out, _ = librosa.core.spectrum._spectrogram(
    #     y=input_data, n_fft=nfft, hop_length=win_step, window=hann_win, power=2)

    if has_channels_dim:
        out = np.expand_dims(out, axis=0)

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

        function = partial(spectrogram_func_librosa, nfft, window_length, window_step)
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
            for nfft, window_length, window_step, shape in [(256, 256, 128, (1, 4096)),
                                                            (256, 256, 128, (4096,)),
                                                            (16, 16, 8, (1, 1000)),
                                                            (10, 10, 5, (1, 1000)),
                                                            ]:
                yield check_operator_spectrogram_vs_python, device, batch_size, shape, \
                    nfft, window_length, window_step

def check_operator_spectrogram_vs_python_wave_1d(device, batch_size, input_length,
                                                 nfft, window_length, window_step):
    f = 4000  # [Hz]
    sr = 44100  # [Hz]
    x = np.arange(input_length, dtype=np.float32)
    y = np.sin(2 * np.pi * f * x / sr)

    data1 = DataIterator(batch_size, y, dtype=np.float32)
    data2 = DataIterator(batch_size, y, dtype=np.float32)

    compare_pipelines(
        SpectrogramPipeline(device, batch_size, iter(data1), nfft=nfft,
                            window_length=window_length, window_step=window_step),
        SpectrogramPythonPipeline(device, batch_size, iter(data2), nfft=nfft,
                                  window_length=window_length, window_step=window_step),
        batch_size=batch_size, N_iterations=5, eps=1e-04)

def test_operator_spectrogram_vs_python_wave():
    for device in ['cpu']:
        for batch_size in [3]:
            for nfft, window_length, window_step, length in [(256, 256, 128, 4096),
                                                             (16, 16, 8, 1000),
                                                             (10, 10, 5, 1000),
                                                             ]:
                yield check_operator_spectrogram_vs_python_wave_1d, device, batch_size, length, \
                    nfft, window_length, window_step

def debug_spectrogram(device, batch_size, data1, data2, nfft, window_length, window_step):
    pipe1 = SpectrogramPipeline(device, batch_size, iter(data1), nfft=nfft,
                                window_length=window_length, window_step=window_step)

    pipe2 = SpectrogramPythonPipeline(device, batch_size, iter(data2), nfft=nfft,
                                      window_length=window_length, window_step=window_step)

    pipe1.build()
    pipe2.build()

    out1 = pipe1.run()[0].at(0)
    out2 = pipe2.run()[0].at(0)

    import cv2;
    den1 = np.amax(out1)
    den2 = np.amax(out2)
    print(den1)
    print(den2)

    out11 = (out1) * 32767 / den1
    out22 = (out2) * 32767 / den2

    out11 = np.squeeze(out11, axis=0)
    out22 = np.squeeze(out22, axis=0)

    print(out11.shape)
    print(out22.shape)

    diff = np.abs(out1 - out2)
    diff_max = np.amax(diff)
    print(diff_max)
    diff = (diff * 32767 + 1000) / diff_max
    diff = np.squeeze(diff, axis=0)

    cv2.imwrite("dali.png", (out11).astype(np.int16))
    cv2.imwrite("reference.png", (out22).astype(np.int16))
    cv2.imwrite("diff.png", (diff).astype(np.int16))

def debug_spectrogram_wave(device, batch_size, input_shape,
                           nfft, window_length, window_step):
    input_length = input_shape[-1]
    f = 4000  # [Hz]
    sr = 44100  # [Hz]
    x = np.arange(input_length, dtype=np.float32)
    y = np.sin(2 * np.pi * f * x / sr)
    y = np.expand_dims(y, axis=0)

    data1 = DataIterator(batch_size, y, dtype=np.float32)
    data2 = DataIterator(batch_size, y, dtype=np.float32)
    debug_spectrogram(device, batch_size, data1, data2, nfft, window_length, window_step)

def debug_spectrogram_randn(device, batch_size, input_shape,
                            nfft, window_length, window_step):
    data1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    data2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)

    debug_spectrogram(device, batch_size, data1, data2, nfft, window_length, window_step)

if __name__ == "__main__":
    debug_spectrogram_randn(device='cpu', batch_size=3, input_shape=(1,4096),
                            nfft=256, window_length=256, window_step=128)