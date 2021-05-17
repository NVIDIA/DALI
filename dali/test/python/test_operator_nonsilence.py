# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from nvidia.dali.pipeline import Pipeline, pipeline_def
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import os
import test_utils
import re

dali_extra = test_utils.get_dali_extra_path()
audio_files_path = os.path.join(dali_extra, "db", "audio", "wav")
audio_files = [os.path.join(audio_files_path, f) for f in os.listdir(audio_files_path) if
               re.match(".*\.wav", f) is not None]


def trim_ref(top_db, ref, frame_length, hop_length, input_data):
    yt, index = librosa.effects.trim(y=input_data, top_db=top_db, ref=ref,
                                     frame_length=frame_length,
                                     hop_length=hop_length)

    # librosa's trim function calculates power with reference to center of window,
    # while DALI uses beginning of window. Hence the subtraction below
    begin = index[0] - frame_length / 2
    length = index[1] - index[0]
    if length != 0:
        length += frame_length - 1
    return np.array(begin), np.array(length)


@pipeline_def
def nonsilence_dali(cutoff_value, window_size, reference_power, reset_interval):
    read, _ = fn.readers.file(device="cpu", files=audio_files)
    audio, rate = fn.decoders.audio(read, device="cpu", dtype=types.FLOAT, downmix=True)
    begin, len = fn.nonsilent_region(audio, cutoff_db=cutoff_value, window_length=window_size,
                                     reference_power=reference_power,
                                     reset_interval=reset_interval,
                                     device="cpu")
    return begin, len


@pipeline_def
def nonsilence_rosa(cutoff_value, window_size, reference_power, reset_interval):
    read, _ = fn.readers.file(device="cpu", files
    =audio_files)
    audio, rate = fn.decoders.audio(read, device="cpu", dtype=types.FLOAT, downmix=True)
    hop_length = 1
    function = partial(trim_ref, cutoff_value,
                       np.max if not reference_power else reference_power,
                       window_size, hop_length)
    begin, len = fn.python_function(audio, function=function, num_outputs=2)
    return begin, len


def check_nonsilence_operator(batch_size, cutoff_value, window_size, reference_power,
                              reset_interval, eps):
    dali = nonsilence_dali(cutoff_value, window_size, reference_power, reset_interval,
                           batch_size=batch_size, num_threads=1, device_id=0)
    rosa = nonsilence_rosa(-cutoff_value, window_size, reference_power, reset_interval,
                           batch_size=batch_size, num_threads=1, device_id=0, exec_async=False,
                           exec_pipelined=False)
    test_utils.compare_pipelines(dali, rosa, batch_size=batch_size, N_iterations=3, eps=eps)


def test_nonsilence_operator():
    batch_size = 3
    window_sizes = [512, 1024]
    reset_intervals = [-1, 2048, 8192]
    references_power = [None, .0003]
    cutoff_coeffs = [-10, -20, -30]
    for ws in window_sizes:
        for ri in reset_intervals:
            for rp in references_power:
                for cc in cutoff_coeffs:
                    yield check_nonsilence_operator, \
                          batch_size, cc, ws, rp, ri, ws
