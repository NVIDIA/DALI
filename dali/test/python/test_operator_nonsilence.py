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
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import test_utils
import os

audio_files = test_utils.get_files(os.path.join('db', 'audio', 'wav'), 'wav')

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


class NonsilencePipeline(Pipeline):
    def __init__(self, batch_size, num_threads=1, exec_async=True, exec_pipelined=True):
        super(NonsilencePipeline, self).__init__(batch_size, num_threads, 0, seed=42,
                                                 exec_async=exec_async,
                                                 exec_pipelined=exec_pipelined)
        self.input = ops.readers.File(device="cpu", files=audio_files)
        self.decode = ops.decoders.Audio(device="cpu", dtype=types.FLOAT, downmix=True)

        self.nonsilence = None

    def define_graph(self):
        if self.nonsilence is None:
            raise RuntimeError(
                "Error: you need to derive from this class and define `self.nonsilence` operator")
        read, _ = self.input()
        audio, rate = self.decode(read)
        begin, len = self.nonsilence(audio)
        return begin, len


class NonsilenceDaliPipeline(NonsilencePipeline):
    def __init__(self, batch_size, cutoff_value, window_size, reference_power,
                 reset_interval):
        super(NonsilenceDaliPipeline, self).__init__(batch_size, num_threads=1)
        self.nonsilence = ops.NonsilentRegion(cutoff_db=cutoff_value, window_length=window_size,
                                              reference_power=reference_power,
                                              reset_interval=reset_interval,
                                              device="cpu")


class NonsilenceRosaPipeline(NonsilencePipeline):
    def __init__(self, batch_size, cutoff_value, window_size, reference_power, reset_interval):
        super(NonsilenceRosaPipeline, self).__init__(batch_size, num_threads=1,
                                                     exec_async=False, exec_pipelined=False)
        hop_length = 1
        function = partial(trim_ref, cutoff_value,
                           np.max if not reference_power else reference_power,
                           window_size, hop_length)
        self.nonsilence = ops.PythonFunction(function=function, num_outputs=2)


def check_nonsilence_operator(batch_size, cutoff_value, window_size, reference_power,
                              reset_interval, eps):
    test_utils.compare_pipelines(
        NonsilenceDaliPipeline(batch_size, cutoff_value, window_size, reference_power,
                               reset_interval),
        NonsilenceRosaPipeline(batch_size, -cutoff_value, window_size, reference_power,
                               reset_interval),
        batch_size=batch_size, N_iterations=3, eps=eps)


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
