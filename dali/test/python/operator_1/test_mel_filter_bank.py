# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
from functools import partial
from test_utils import compare_pipelines
from test_utils import RandomlyShapedDataIterator
import librosa as librosa
from nose_utils import attr
from nose2.tools import cartesian_params


class MelFilterBankPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        nfilter,
        sample_rate,
        freq_low,
        freq_high,
        normalize,
        mel_formula,
        layout="ft",
        num_threads=1,
        device_id=0,
    ):
        super(MelFilterBankPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.fbank = ops.MelFilterBank(
            device=self.device,
            nfilter=nfilter,
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
            normalize=normalize,
            mel_formula=mel_formula,
        )
        self.layout = layout

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == "gpu" else self.data
        out = self.fbank(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def mel_fbank_func(nfilter, sample_rate, freq_low, freq_high, normalize, mel_formula, input_data):
    in_shape = input_data.shape
    axis = -2 if len(in_shape) > 1 else 0
    fftbin_size = in_shape[axis]
    nfft = 2 * (fftbin_size - 1)
    librosa_norm = "slaney" if normalize else None
    librosa_htk = mel_formula == "htk"
    mel_transform = librosa.filters.mel(
        sr=sample_rate,
        n_mels=nfilter,
        n_fft=nfft,
        fmin=freq_low,
        fmax=freq_high,
        norm=librosa_norm,
        dtype=np.float32,
        htk=librosa_htk,
    )

    out_shape = list(in_shape)
    out_shape[axis] = nfilter
    out_shape = tuple(out_shape)
    out = np.zeros(out_shape, dtype=np.float32)

    if len(in_shape) == 3:
        for i in range(in_shape[0]):
            out[i, :, :] = np.dot(mel_transform, input_data[i, :, :])
    elif len(in_shape) <= 2:
        out = np.dot(mel_transform, input_data)
    return out


class MelFilterBankPythonPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        nfilter,
        sample_rate,
        freq_low,
        freq_high,
        normalize,
        mel_formula,
        layout="ft",
        num_threads=1,
        device_id=0,
        func=mel_fbank_func,
    ):
        super(MelFilterBankPythonPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12345, exec_async=False, exec_pipelined=False
        )
        self.device = "cpu"
        self.iterator = iterator
        self.inputs = ops.ExternalSource()

        function = partial(func, nfilter, sample_rate, freq_low, freq_high, normalize, mel_formula)
        self.mel_fbank = ops.PythonFunction(function=function)
        self.layout = layout
        self.freq_major = layout.find("f") != len(layout) - 1
        self.need_transpose = not self.freq_major and len(layout) > 1
        if self.need_transpose:
            perm = [i for i in range(len(layout))]
            f = layout.find("f")
            perm[f] = len(layout) - 2
            perm[-2] = f
            self.transpose = ops.Transpose(perm=perm)

    def _transposed(self, op):
        return lambda x: self.transpose(op(self.transpose(x)))

    def define_graph(self):
        self.data = self.inputs()
        mel_fbank = self._transposed(self.mel_fbank) if self.need_transpose else self.mel_fbank
        out = mel_fbank(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)


def check_operator_mel_filter_bank_vs_python(
    device,
    batch_size,
    max_shape,
    nfilter,
    sample_rate,
    freq_low,
    freq_high,
    normalize,
    mel_formula,
    layout,
):
    f_axis = layout.find("f")
    min_shape = [1 for _ in max_shape]
    min_shape[f_axis] = max_shape[f_axis]
    eii1 = RandomlyShapedDataIterator(
        batch_size, min_shape=min_shape, max_shape=max_shape, dtype=np.float32
    )
    eii2 = RandomlyShapedDataIterator(
        batch_size, min_shape=min_shape, max_shape=max_shape, dtype=np.float32
    )
    compare_pipelines(
        MelFilterBankPipeline(
            device,
            batch_size,
            iter(eii1),
            nfilter=nfilter,
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
            normalize=normalize,
            mel_formula=mel_formula,
            layout=layout,
        ),
        MelFilterBankPythonPipeline(
            device,
            batch_size,
            iter(eii2),
            nfilter=nfilter,
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
            normalize=normalize,
            mel_formula=mel_formula,
            layout=layout,
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1e-03,
    )


mel_filter_bank_cases = [
    (4, 16000.0, 0.0, 8000.0, (17,), "f"),
    (4, 16000.0, 0.0, 8000.0, (17, 1), "ft"),
    (128, 16000.0, 0.0, 8000.0, (513, 100), "ft"),
    (128, 48000.0, 0.0, 24000.0, (513, 100), "ft"),
    (128, 16000.0, 0.0, 8000.0, (10, 513, 100), "Ctf"),
    (128, 48000.0, 4000.0, 24000.0, (513, 100), "tf"),
    (128, 44100.0, 0.0, 22050.0, (513, 100), "tf"),
    (128, 44100.0, 1000.0, 22050.0, (513, 100), "tf"),
]


@cartesian_params(
    ["cpu", "gpu"],  # device
    [1, 3],  # batch_size
    ["htk", "slaney"],  # mel_formula
    mel_filter_bank_cases,  # (nfilter, sample_rate, freq_low, freq_high, shape, layout)
)
def test_operator_mel_filter_bank_vs_python_normalize(device, batch_size, mel_formula, case):
    nfilter, sample_rate, freq_low, freq_high, shape, layout = case
    check_operator_mel_filter_bank_vs_python(
        device,
        batch_size,
        shape,
        nfilter,
        sample_rate,
        freq_low,
        freq_high,
        True,
        mel_formula,
        layout,
    )


@attr("sanitizer_skip")
@cartesian_params(
    ["cpu", "gpu"],  # device
    [1, 3],  # batch_size
    ["htk", "slaney"],  # mel_formula
    mel_filter_bank_cases,  # (nfilter, sample_rate, freq_low, freq_high, shape, layout)
)
def test_operator_mel_filter_bank_vs_python_wo_normalize(device, batch_size, mel_formula, case):
    nfilter, sample_rate, freq_low, freq_high, shape, layout = case
    check_operator_mel_filter_bank_vs_python(
        device,
        batch_size,
        shape,
        nfilter,
        sample_rate,
        freq_low,
        freq_high,
        False,
        mel_formula,
        layout,
    )
