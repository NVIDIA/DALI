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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import test_utils
from functools import partial
import os
import librosa

dali_extra = test_utils.get_dali_extra_path()
audio_files = os.path.join(dali_extra, "db", "audio")


def trim_ref(top_db, ref, frame_length, hop_length, input_data):
    yt, index = librosa.effects.trim(y=input_data, top_db=top_db, ref=ref,
                                     frame_length=frame_length,
                                     hop_length=hop_length)
    print(np.array([index[0]]), np.array([index[1] - index[0]]))
    return np.array([index[0]]), np.array([index[1] - index[0]])


class NonsilencePipeline(Pipeline):
    def __init__(self, batch_size, num_threads=1, exec_async=True, exec_pipelined=True):
        super(NonsilencePipeline, self).__init__(batch_size, num_threads, 0, seed=42,
                                                 exec_async=exec_async,
                                                 exec_pipelined=exec_pipelined)
        self.input = ops.FileReader(device="cpu", file_root=audio_files)
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True)

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
    def __init__(self, batch_size, cutoff_value, window_size, reference_db, reference_max,
                 reset_interval):
        super(NonsilenceDaliPipeline, self).__init__(batch_size, num_threads=1)
        self.nonsilence = ops.NonsilenceRegion(cutoff_db=cutoff_value, window_length=window_size,
                                               reference_db=reference_db,
                                               reference_max=reference_max,
                                               reset_interval=reset_interval,
                                               device="cpu")


class NonsilenceRosaPipeline(NonsilencePipeline):
    def __init__(self, batch_size, cutoff_value, window_size, reference_db, reference_max,
                 reset_interval):
        super(NonsilenceRosaPipeline, self).__init__(batch_size, num_threads=1,
                                                     exec_async=False, exec_pipelined=False)
        hop_length = 1
        function = partial(trim_ref, cutoff_value, np.max if reference_max else reference_db,
                           window_size, hop_length)
        self.nonsilence = ops.PythonFunction(function=function, num_outputs=2)


def check_nonsilence_operator(batch_size, cutoff_value, window_size, reference_db, reference_max,
                              reset_interval):
    test_utils.compare_pipelines(
        NonsilenceDaliPipeline(batch_size, cutoff_value, window_size, reference_db, reference_max,
                               reset_interval),
        NonsilenceRosaPipeline(batch_size, cutoff_value, window_size, reference_db, reference_max,
                               reset_interval),
        batch_size=batch_size, N_iterations=1, eps=window_size)


def test_nonsilence_operator():
    batch_size = 1
    window_sizes = [2048]
    reset_intervals = [-1, 2048*4]
    references_max = [True]
    references_db = [1.]
    cutoff_coeffs = [20]
    for ws in window_sizes:
        for ri in reset_intervals:
            for rm in references_max:
                for rd in references_db:
                    for cc in cutoff_coeffs:
                        yield check_nonsilence_operator, batch_size, cc, ws, rd, rm, ri
