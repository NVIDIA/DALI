# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali as dali
from nvidia.dali import fn, pipeline_def, types

import numpy as np
import scipy.io.wavfile
from test_audio_decoder_utils import generate_waveforms, rosa_resample
from test_utils import compare_pipelines, get_files, check_batch

names = [
  "/tmp/dali_test_1C.wav",
  "/tmp/dali_test_2C.wav",
  "/tmp/dali_test_4C.wav"
]

freqs = [
  np.array([0.02]),
  np.array([0.01, 0.012]),
  np.array([0.01, 0.012, 0.013, 0.014])
]
rates = [ 16000, 22050, 12347 ]
lengths = [ 10000, 54321, 12345 ]

def create_test_files():
  for i in range(len(names)):
    wave = generate_waveforms(lengths[i], freqs[i])
    wave = (wave * 32767).round().astype(np.int16)
    scipy.io.wavfile.write(names[i], rates[i], wave)


create_test_files()


@pipeline_def
def audio_decoder_pipe():
    encoded, _ = fn.readers.file(files=names)
    audio0, sr0 = fn.decoders.audio(encoded, dtype=types.FLOAT)
    out_sr = 15000
    audio1, sr1 = fn.decoders.audio(encoded, dtype=types.FLOAT, sample_rate=out_sr)
    audio2 = fn.experimental.audio_resample(audio0, in_rate=sr0, out_rate=out_sr)
    audio3 = fn.experimental.audio_resample(audio0, scale=out_sr/sr0)
    audio4 = fn.experimental.audio_resample(audio0, out_length=fn.shapes(audio1)[0])
    return audio1, audio2, audio3, audio4

def test_standalone_vs_fused():
    pipe = audio_decoder_pipe(batch_size=2, num_threads=1, device_id=0)
    pipe.build()
    for _ in range(2):
        outs = pipe.run()
        # two sampling rates - should be bit-exact
        check_batch(outs[0], outs[1], eps=0, max_allowed_error=0)
        # numerical round-off error in rate
        check_batch(outs[0], outs[2], eps=1e-6, max_allowed_error=1e-4)
        # here, the sampling rate is slightly different, so we can tolerate larger errors
        check_batch(outs[0], outs[3], eps=1e-4, max_allowed_error=1)
