# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import nvidia.dali.types as types
import test_utils
import os
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from test_audio_utils_librosa_ref import nonsilent_region

audio_files = test_utils.get_files(os.path.join("db", "audio", "wav"), "wav")


def trim_ref(cutoff_db, ref, frame_length, hop_length, input_data):
    start, length = nonsilent_region(
        y=input_data, top_db=-cutoff_db, ref=ref, frame_length=frame_length, hop_length=hop_length
    )
    return np.array(start), np.array(length)


@pipeline_def
def nonsilent_region_pipe(cutoff_value, window_size, reference_power, reset_interval):
    raw, _ = fn.readers.file(files=audio_files)
    audio, _ = fn.decoders.audio(raw, dtype=types.FLOAT, downmix=True)
    begin_cpu, len_cpu = fn.nonsilent_region(
        audio,
        cutoff_db=cutoff_value,
        window_length=window_size,
        reference_power=reference_power,
        reset_interval=reset_interval,
    )
    begin_gpu, len_gpu = fn.nonsilent_region(
        audio.gpu(),
        cutoff_db=cutoff_value,
        window_length=window_size,
        reference_power=reference_power,
        reset_interval=reset_interval,
    )
    return audio, begin_cpu, len_cpu, begin_gpu, len_gpu


def check_nonsilence_operator(
    batch_size, cutoff_value, window_size, reference_power, reset_interval, eps
):
    pipe = nonsilent_region_pipe(
        cutoff_value,
        window_size,
        reference_power,
        reset_interval,
        batch_size=batch_size,
        num_threads=3,
        device_id=0,
        seed=42,
    )
    hop_length = 1
    ref = np.max if not reference_power else reference_power
    for _ in range(3):
        audio_batch_cpu, begin_batch_cpu, len_batch_cpu, begin_batch_gpu, len_batch_gpu = pipe.run()
        for s in range(batch_size):
            audio_cpu = test_utils.as_array(audio_batch_cpu[s])
            begin_cpu = test_utils.as_array(begin_batch_cpu[s])
            len_cpu = test_utils.as_array(len_batch_cpu[s])
            begin_gpu = test_utils.as_array(begin_batch_gpu[s])
            len_gpu = test_utils.as_array(len_batch_gpu[s])

            ref_begin, ref_len = trim_ref(cutoff_value, ref, window_size, hop_length, audio_cpu)
            np.testing.assert_allclose(ref_begin, begin_cpu, atol=eps)
            np.testing.assert_allclose(ref_begin, begin_gpu, atol=eps)
            np.testing.assert_allclose(ref_len, len_cpu, atol=eps)
            np.testing.assert_allclose(ref_len, len_gpu, atol=eps)

            np.testing.assert_allclose(begin_cpu, begin_gpu, atol=1)
            np.testing.assert_allclose(len_cpu, len_gpu, atol=10)


def test_nonsilence_operator():
    batch_size = 3
    window_sizes = [512, 1024]
    reset_intervals = [-1, 2048, 8192]
    references_power = [None, 0.0003]
    cutoff_coeffs = [-10, -60, -80]
    for ws in window_sizes:
        for ri in reset_intervals:
            for rp in references_power:
                for cc in cutoff_coeffs:
                    yield check_nonsilence_operator, batch_size, cc, ws, rp, ri, ws


def test_cpu_vs_gpu():
    batch_size = 8

    @pipeline_def
    def nonsilent_pipe(data_arr=None, window_size=256, cutoff_value=-10, reference_power=None):
        if data_arr is None:
            raw, _ = fn.readers.file(files=audio_files)
            audio, _ = fn.decoders.audio(raw, dtype=types.INT16, downmix=True)
        else:
            audio = types.Constant(device="cpu", value=data_arr)

        begin_cpu, len_cpu = fn.nonsilent_region(
            audio,
            cutoff_db=cutoff_value,
            window_length=window_size,
            reference_power=reference_power,
        )
        begin_gpu, len_gpu = fn.nonsilent_region(
            audio.gpu(),
            cutoff_db=cutoff_value,
            window_length=window_size,
            reference_power=reference_power,
        )
        return begin_cpu, len_cpu, begin_gpu, len_gpu

    audio_arr = np.zeros([10 + 1 + 10], dtype=np.int16)
    audio_arr[10] = 3000
    pipe = nonsilent_pipe(
        data_arr=audio_arr,
        window_size=1,
        cutoff_value=-80,
        batch_size=1,
        num_threads=3,
        device_id=0,
    )
    begin_cpu, len_cpu, begin_gpu, len_gpu = [test_utils.as_array(out[0]) for out in pipe.run()]
    assert begin_cpu == begin_gpu == 10
    assert len_cpu == len_gpu == 1

    audio_arr[10:15] = 3000
    pipe = nonsilent_pipe(
        data_arr=audio_arr, window_size=1, batch_size=1, num_threads=3, device_id=0
    )
    begin_cpu, len_cpu, begin_gpu, len_gpu = [test_utils.as_array(out[0]) for out in pipe.run()]
    assert begin_cpu == begin_gpu == 10
    assert len_cpu == len_gpu == 5

    window = 5
    pipe = nonsilent_pipe(
        data_arr=audio_arr, window_size=5, batch_size=1, num_threads=3, device_id=0
    )
    outputs = pipe.run()
    begin_cpu, len_cpu, begin_gpu, len_gpu = [test_utils.as_array(out[0]) for out in outputs]
    assert begin_cpu == begin_gpu == (10 - window + 1)
    assert len_cpu == len_gpu == 13

    pipe = nonsilent_pipe(batch_size=batch_size, num_threads=3, device_id=0, seed=42)
    for _ in range(3):
        begin_batch_cpu, len_batch_cpu, begin_batch_gpu, len_batch_gpu = pipe.run()
        for s in range(batch_size):
            begin_cpu = test_utils.as_array(begin_batch_cpu[s])
            len_cpu = test_utils.as_array(len_batch_cpu[s])
            begin_gpu = test_utils.as_array(begin_batch_gpu[s])
            len_gpu = test_utils.as_array(len_batch_gpu[s])

            np.testing.assert_array_equal(begin_cpu, begin_gpu)
            np.testing.assert_array_equal(len_cpu, len_gpu)
