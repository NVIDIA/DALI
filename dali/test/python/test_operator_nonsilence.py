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


def trim_ref(top_db, frame_length, hop_length, input_data):
    y, sr = librosa.load(librosa.util.example_audio_file())
    yt, index = librosa.effects.trim(y=input_data, top_db=top_db, frame_length=frame_length,
                                     hop_length=hop_length)
    return np.array([index[0]]), np.array([index[1] - index[0]])


# TODO refator derive
class NonsilenceDaliPipeline(Pipeline):
    def __init__(self, batch_size, cutoff_value, num_threads=1):
        super(NonsilenceDaliPipeline, self).__init__(batch_size, num_threads, 0)
        self.input = ops.FileReader(device="cpu", file_root=audio_files)
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True)

        self.nonsilence = ops.NonsilenceRegion(cutoff_value=cutoff_value, device="cpu")

    def define_graph(self):
        read, _ = self.input()
        audio, rate = self.decode(read)
        out = self.nonsilence(audio)
        return out


class NonsilenceRosaPipeline(Pipeline):
    def __init__(self, batch_size, cutoff_value, trim_ref_function=trim_ref, num_threads=1):
        frame_length = 2048
        hop_length = 512
        super(NonsilenceRosaPipeline, self).__init__(batch_size, num_threads, 0, seed=42,
                                                     exec_async=False, exec_pipelined=False)
        self.input = ops.FileReader(device="cpu", file_root=audio_files)
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True)

        function = partial(trim_ref_function, cutoff_value, frame_length, hop_length)
        self.nonsilence = ops.PythonFunction(function=function, num_outputs=2)

    def define_graph(self):
        read, _ = self.input()
        audio, rate = self.decode(read)
        out = self.nonsilence(audio)
        return out


def check_nonsilence_operator(batch_size, cutoff_value):
    test_utils.compare_pipelines(NonsilenceDaliPipeline(batch_size, cutoff_value),
                                 NonsilenceRosaPipeline(batch_size, cutoff_value),
                                 batch_size=batch_size, N_iterations=1)


def test_nonsilence_operator():
    batch_size = 1
    for coef in [20]:
        yield check_nonsilence_operator, batch_size, coef
