# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from functools import partial
from test_utils import get_files
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import ConstantDataIterator
import math
from test_audio_utils_librosa_ref import stft

from nose2.tools import params
from nose_utils import assert_raises


audio_files = get_files("db/audio/wav", "wav")


class SpectrogramPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        nfft,
        window_length,
        window_step,
        window=None,
        center=None,
        num_threads=1,
        device_id=0,
    ):
        super(SpectrogramPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        window_fn = window(window_length).tolist() if window is not None else None
        self.fft = ops.Spectrogram(
            device=self.device,
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
            window_fn=window_fn,
            center_windows=center,
            power=2,
        )
        # randomly insert extra axis (channels?)
        self.r = np.random.randint(-1, 2)

    def define_graph(self):
        self.data = self.inputs()
        out = self.data.gpu() if self.device == "gpu" else self.data
        out = self.fft(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        if self.r == 0:
            data = [x[np.newaxis, :] for x in data]
        elif self.r == 1:
            data = [x[:, np.newaxis] for x in data]

        self.feed_input(self.data, data)


def hann_win(n):
    hann = np.ones([n], dtype=np.float32)
    a = 2.0 * math.pi / n
    for t in range(n):
        phase = a * (t + 0.5)
        hann[t] = 0.5 * (1.0 - math.cos(phase))
    return hann


def cos_win(n):
    phase = (np.arange(n) + 0.5) * (math.pi / n)
    return np.sin(phase).astype(np.float32)


def spectrogram_func_librosa(nfft, win_len, win_step, window, center, input_data):
    # Squeeze to 1d
    if len(input_data.shape) > 1:
        input_data = np.squeeze(input_data)

    if window is None:
        window = hann_win

    out = (
        np.abs(
            stft(
                y=input_data,
                n_fft=nfft or win_len,
                center=center,
                win_length=win_len,
                hop_length=win_step,
                window=window,
                pad_mode="reflect",
            )
        )
        ** 2
    )

    # Alternative way to calculate the spectrogram:
    # out, _ = librosa.core.spectrum._spectrogram(
    #     y=input_data, n_fft=nfft, hop_length=win_step, window=hann_win, power=2)

    return out


class SpectrogramPythonPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        nfft,
        window_length,
        window_step,
        window=None,
        center=None,
        num_threads=1,
        device_id=0,
        spectrogram_func=spectrogram_func_librosa,
    ):
        super(SpectrogramPythonPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12345, exec_async=False, exec_pipelined=False
        )
        self.device = "cpu"
        self.iterator = iterator
        self.inputs = ops.ExternalSource()

        function = partial(spectrogram_func, nfft, window_length, window_step, window, center)
        self.spectrogram = ops.PythonFunction(function=function, output_layouts=["ft"])

    def define_graph(self):
        self.data = self.inputs()
        out = self.spectrogram(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data)


def check_operator_spectrogram_vs_python(
    device, batch_size, input_shape, nfft, window_length, window_step, center
):
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(
        SpectrogramPipeline(
            device,
            batch_size,
            iter(eii1),
            nfft=nfft,
            window=None,
            window_length=window_length,
            window_step=window_step,
            center=center,
        ),
        SpectrogramPythonPipeline(
            device,
            batch_size,
            iter(eii2),
            window=None,
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
            center=center,
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1e-04,
    )


def test_operator_spectrogram_vs_python():
    for device in ["cpu", "gpu"]:
        for batch_size in [3]:
            for center in [False, True]:
                for nfft, window_length, window_step, shape in [
                    (256, 256, 128, (1, 4096)),
                    (256, 256, 128, (4096,)),
                    (256, 256, 128, (4096, 1)),
                    (256, 256, 128, (1, 1, 4096, 1)),
                    (16, 16, 8, (1, 1000)),
                    (10, 10, 5, (1, 1000)),
                    (None, 10, 5, (1, 1000)),
                ]:
                    yield (
                        check_operator_spectrogram_vs_python,
                        device,
                        batch_size,
                        shape,
                        nfft,
                        window_length,
                        window_step,
                        center,
                    )


def check_operator_spectrogram_vs_python_wave_1d(
    device, batch_size, input_length, nfft, window_length, window_step, window, center
):
    f = 4000  # [Hz]
    sr = 44100  # [Hz]
    x = np.arange(input_length, dtype=np.float32)
    y = np.sin(2 * np.pi * f * x / sr)

    data1 = ConstantDataIterator(batch_size, y, dtype=np.float32)
    data2 = ConstantDataIterator(batch_size, y, dtype=np.float32)

    compare_pipelines(
        SpectrogramPipeline(
            device,
            batch_size,
            iter(data1),
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
            window=window,
            center=center,
        ),
        SpectrogramPythonPipeline(
            device,
            batch_size,
            iter(data2),
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
            window=window,
            center=center,
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1e-04,
    )


def test_operator_spectrogram_vs_python_wave():
    for device in ["cpu", "gpu"]:
        for window in [None, hann_win, cos_win]:
            for batch_size in [3]:
                for nfft, window_length, window_step, length in [
                    (256, 256, 128, 4096),
                    (128, 100, 61, 1000),
                    (10, 10, 5, 1000),
                ]:
                    # Note: center_windows=False and nfft > window_length doesn't work like librosa.
                    # Librosa seems to disregard window_length
                    # and extract windows of nfft size regardless
                    for center in [False, True] if nfft == window_length else [True]:
                        yield (
                            check_operator_spectrogram_vs_python_wave_1d,
                            device,
                            batch_size,
                            length,
                            nfft,
                            window_length,
                            window_step,
                            window,
                            center,
                        )


class AudioSpectrogramPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        nfft,
        window_length,
        window_step,
        center,
        layout="ft",
        num_threads=1,
        device_id=0,
    ):
        super(AudioSpectrogramPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.readers.File(device="cpu", files=audio_files)
        self.decode = ops.decoders.Audio(device="cpu", dtype=types.FLOAT, downmix=True)
        self.fft = ops.Spectrogram(
            device=device,
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
            power=2,
            center_windows=center,
            layout=layout,
        )

    def define_graph(self):
        read, _ = self.input()
        audio, rate = self.decode(read)
        if self.fft.device == "gpu":
            audio = audio.gpu()
        spec = self.fft(audio)
        return spec


class AudioSpectrogramPythonPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        nfft,
        window_length,
        window_step,
        center,
        layout="ft",
        num_threads=1,
        device_id=0,
        spectrogram_func=spectrogram_func_librosa,
    ):
        super(AudioSpectrogramPythonPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=12345, exec_async=False, exec_pipelined=False
        )

        self.input = ops.readers.File(device="cpu", files=audio_files)
        self.decode = ops.decoders.Audio(device="cpu", dtype=types.FLOAT, downmix=True)

        function = partial(spectrogram_func, nfft, window_length, window_step, None, center)
        self.spectrogram = ops.PythonFunction(function=function, output_layouts=["ft"])
        self.layout = layout

    def define_graph(self):
        read, _ = self.input()
        audio, rate = self.decode(read)
        out = self.spectrogram(audio)
        if self.layout == "tf":
            out = dali.fn.transpose(out, perm=[1, 0], transpose_layout=True)

        return out


def check_operator_decoder_and_spectrogram_vs_python(
    device, batch_size, nfft, window_length, window_step, center, layout
):
    compare_pipelines(
        AudioSpectrogramPipeline(
            device=device,
            batch_size=batch_size,
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
            center=center,
            layout=layout,
        ),
        AudioSpectrogramPythonPipeline(
            batch_size,
            nfft=nfft,
            window_length=window_length,
            window_step=window_step,
            center=center,
            layout=layout,
        ),
        batch_size=batch_size,
        N_iterations=3,
        eps=1e-04,
    )


def test_operator_decoder_and_spectrogram():
    for device in ["cpu", "gpu"]:
        for layout in ["tf", "ft"]:
            for batch_size in [3]:
                for nfft, window_length, window_step in [
                    (256, 256, 128),
                    (256, 256, 128),
                    (256, 256, 128),
                    (
                        256,
                        256,
                        128,
                    ),
                    (
                        256,
                        256,
                        128,
                    ),
                    (
                        16,
                        16,
                        8,
                    ),
                    (
                        10,
                        10,
                        5,
                    ),
                ]:
                    # Note: center_windows=False and nfft > window_length doesn't work like librosa.
                    # Librosa seems to disregards window_length
                    # and extract windows of nfft size regardless
                    for center in [False, True] if nfft == window_length else [True]:
                        yield (
                            check_operator_decoder_and_spectrogram_vs_python,
                            device,
                            batch_size,
                            nfft,
                            window_length,
                            window_step,
                            center,
                            layout,
                        )


@params(
    *[
        (device, inp, center)
        for device in ["cpu", "gpu"]
        for inp in ["arange", "zero_vol", "empty_batch"]
        for center in [False, True]
    ],
)
def test_no_windows(device, inp, center):
    def sample(sample_info):
        if inp == "arange":
            return np.arange(sample_info.idx_in_batch + 1, dtype=np.float32)
        elif inp in ("zero_vol", "empty_batch"):
            return np.arange(sample_info.idx_in_batch, dtype=np.float32)
        else:
            raise AssertionError(f"Invalid input: {inp}")

    enable_conditionals = inp == "empty_batch"

    @dali.pipeline_def(
        batch_size=1, num_threads=4, device_id=0, enable_conditionals=enable_conditionals
    )
    def pipeline(device):
        sig = dali.fn.external_source(source=sample, batch=False)
        if device == "gpu":
            sig = sig.gpu()
        if enable_conditionals:
            dummy = dali.fn.external_source(
                source=lambda x: np.array(42, dtype=np.int32), batch=False
            )
            if dummy < 0:
                spec = dali.fn.spectrogram(
                    sig, window_length=512, center_windows=center, window_step=3
                )
            else:
                spec = sig
        else:
            spec = dali.fn.spectrogram(sig, window_length=512, center_windows=center, window_step=3)
        return spec

    pipe = pipeline(device=device)
    pipe.build()
    # empty batch should be supported by any op
    # and non-empty sample with centered window always ends up
    # with a positive number of windows
    if inp == "empty_batch" or (inp == "arange" and center):
        pipe.run()
    else:
        error_msg = (
            "Signal is too short"
            if inp == "arange"
            else "Spectogram does not support empty (0-volume) samples"
        )
        with assert_raises(RuntimeError, glob=error_msg):
            pipe.run()
