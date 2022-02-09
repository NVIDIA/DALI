# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from functools import partial
import itertools
import librosa
import numpy as np
import nvidia.dali.types as types
import test_utils
import os
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

audio_files = test_utils.get_files(os.path.join('db', 'audio', 'wav'), 'wav')

def trim_ref(cutoff_db, ref, frame_length, hop_length, input_data):
    yt, index = librosa.effects.trim(y=input_data, top_db=-cutoff_db, ref=ref,
                                     frame_length=frame_length,
                                     hop_length=hop_length)
    # librosa's trim function calculates power with reference to center of window,
    # while DALI uses beginning of window. Hence the subtraction below
    begin = index[0] - frame_length // 2
    length = index[1] - index[0]
    if length != 0:
        length += frame_length - 1
    return np.array(begin), np.array(length)

@pipeline_def
def nonsilent_region_pipe(cutoff_value, window_size, reference_power, reset_interval):
    raw, _ = fn.readers.file(files=audio_files)
    audio, _ = fn.decoders.audio(raw, dtype=types.FLOAT, downmix=True)
    begin, len = fn.nonsilent_region(
        audio, cutoff_db=cutoff_value, window_length=window_size,
        reference_power=reference_power,
        reset_interval=reset_interval
    )
    return audio, begin, len

def check_nonsilence_operator(batch_size, cutoff_value, window_size, reference_power,
                              reset_interval, eps):
    pipe = nonsilent_region_pipe(
        cutoff_value, window_size, reference_power, reset_interval,
        batch_size=batch_size, num_threads=3, device_id=0, seed=42,
    )
    hop_length = 1
    ref = np.max if not reference_power else reference_power
    pipe.build()
    for _ in range(3):
        audio_batch, begin_batch, len_batch = pipe.run()
        for s in range(batch_size):
            audio = np.array(audio_batch[s])
            begin = np.array(begin_batch[s])
            len = np.array(len_batch[s])
            ref_begin, ref_len = trim_ref(
                cutoff_value, ref, window_size, hop_length, audio
            )
            np.testing.assert_allclose(ref_begin, begin, atol=eps)
            np.testing.assert_allclose(ref_len, len, atol=eps)

def test_nonsilence_operator():
    batch_size = 3
    window_sizes = [512, 1024]
    reset_intervals = [-1, 2048, 8192]
    references_power = [None, .0003]
    cutoff_coeffs = [-10, -60, -80]
    for ws in window_sizes:
        for ri in reset_intervals:
            for rp in references_power:
                for cc in cutoff_coeffs:
                    yield check_nonsilence_operator, \
                          batch_size, cc, ws, rp, ri, ws
