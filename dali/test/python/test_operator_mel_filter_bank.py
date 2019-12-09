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
import math
import librosa as librosa

class MelFilterBankPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, nfilter, sample_rate, freq_low, freq_high,
                 normalize, mel_formula, num_threads=1, device_id=0):
        super(MelFilterBankPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.fbank = ops.MelFilterBank(device = self.device,
                                       nfilter = nfilter,
                                       sample_rate = sample_rate,
                                       freq_low = freq_low,
                                       freq_high = freq_high,
                                       normalize = normalize,
                                       mel_formula = mel_formula)

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.fbank(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def mel_fbank_func(nfilter, sample_rate, freq_low, freq_high, normalize, mel_formula, input_data):
    in_shape = input_data.shape
    nfft = 2 * (input_data.shape[-2] - 1)
    librosa_norm = 1 if normalize else None
    librosa_htk = (mel_formula == 'htk')
    mel_transform = librosa.filters.mel(
        sr = sample_rate, n_mels=nfilter, n_fft = nfft,
        fmin=freq_low, fmax=freq_high,
        norm=librosa_norm, dtype=np.float32, htk=librosa_htk)

    out_shape = list(in_shape)
    out_shape[-2] = nfilter
    out_shape = tuple(out_shape)
    out = np.zeros(out_shape, dtype=np.float32)

    if len(in_shape) == 3:
        for i in range(in_shape[0]):
            out[i, :, :] = np.dot(mel_transform, input_data[i, :, :])
    elif len(in_shape) == 2:
        out = np.dot(mel_transform, input_data)
    return out

class MelFilterBankPythonPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, nfilter, sample_rate, freq_low, freq_high,
                 normalize, mel_formula, num_threads=1, device_id=0, func=mel_fbank_func):
        super(MelFilterBankPythonPipeline, self).__init__(
              batch_size, num_threads, device_id,
              seed=12345, exec_async=False, exec_pipelined=False)
        self.device = "cpu"
        self.iterator = iterator
        self.inputs = ops.ExternalSource()

        function = partial(func, nfilter, sample_rate, freq_low, freq_high, normalize, mel_formula)
        self.mel_fbank = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.mel_fbank(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)

def check_operator_mel_filter_bank_vs_python(device, batch_size, input_shape,
                                             nfilter, sample_rate, freq_low, freq_high,
                                             normalize, mel_formula):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        MelFilterBankPipeline(device, batch_size, iter(eii1),
                              nfilter=nfilter, sample_rate=sample_rate, freq_low=freq_low, freq_high=freq_high,
                              normalize=normalize, mel_formula=mel_formula),
        MelFilterBankPythonPipeline(device, batch_size, iter(eii2),
                                    nfilter=nfilter, sample_rate=sample_rate, freq_low=freq_low, freq_high=freq_high,
                                    normalize=normalize, mel_formula=mel_formula),
        batch_size=batch_size, N_iterations=5, eps=1e-03)

def test_operator_mel_filter_bank_vs_python():
    for device in ['cpu']:
        for batch_size in [1, 3]:
            for normalize in [True, False]:
                for mel_formula in ['htk', 'slaney']:
                    for nfilter, sample_rate, freq_low, freq_high, shape in \
                        [(4, 16000.0, 0.0, 8000.0, (17, 1)),
                        (128, 16000.0, 0.0, 8000.0, (513, 100)),
                        (128, 16000.0, 0.0, 8000.0, (10, 513, 100)),
                        (128, 48000.0, 0.0, 24000.0, (513, 100)),
                        (128, 48000.0, 4000.0, 24000.0, (513, 100)),
                        (128, 44100.0, 0.0, 22050.0, (513, 100)),
                        (128, 44100.0, 1000.0, 22050.0, (513, 100))]:
                        yield check_operator_mel_filter_bank_vs_python, device, batch_size, shape, \
                            nfilter, sample_rate, freq_low, freq_high, normalize, mel_formula
