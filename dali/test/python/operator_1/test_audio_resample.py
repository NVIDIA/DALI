# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import scipy.io.wavfile
from nvidia.dali import fn, pipeline_def, types

from test_audio_decoder_utils import generate_waveforms
from test_utils import check_batch, dali_type_to_np, as_array

names = ["/tmp/dali_test_1C.wav", "/tmp/dali_test_2C.wav", "/tmp/dali_test_4C.wav"]

freqs = [np.array([0.02]), np.array([0.01, 0.012]), np.array([0.01, 0.012, 0.013, 0.014])]
rates = [16000, 22050, 12347]
lengths = [10000, 54321, 12345]


def create_files():
    for i in range(len(names)):
        wave = generate_waveforms(lengths[i], freqs[i])
        wave = (wave * 32767).round().astype(np.int16)
        scipy.io.wavfile.write(names[i], rates[i], wave)


create_files()


@pipeline_def
def audio_decoder_pipe(device):
    encoded, _ = fn.readers.file(files=names)
    audio0, sr0 = fn.decoders.audio(encoded, dtype=types.FLOAT)
    out_sr = 15000
    audio1, sr1 = fn.decoders.audio(encoded, dtype=types.FLOAT, sample_rate=out_sr)
    if device == "gpu":
        audio0 = audio0.gpu()
    audio2 = fn.audio_resample(audio0, in_rate=sr0, out_rate=out_sr)
    audio3 = fn.audio_resample(audio0, scale=out_sr / sr0)
    audio4 = fn.audio_resample(audio0, out_length=audio1.shape()[0])
    return audio1, audio2, audio3, audio4


def _test_standalone_vs_fused(device):
    pipe = audio_decoder_pipe(device=device, batch_size=2, num_threads=1, device_id=0)
    is_gpu = device == "gpu"
    for _ in range(2):
        outs = pipe.run()
        # two sampling rates - should be bit-exact
        check_batch(
            outs[0], outs[1], eps=1e-6 if is_gpu else 0, max_allowed_error=1e-4 if is_gpu else 0
        )
        # numerical round-off error in rate
        check_batch(outs[0], outs[2], eps=1e-6, max_allowed_error=1e-4)
        # here, the sampling rate is slightly different, so we can tolerate larger errors
        check_batch(outs[0], outs[3], eps=1e-4, max_allowed_error=1)


def test_standalone_vs_fused():
    for device in ("gpu", "cpu"):
        yield _test_standalone_vs_fused, device


def _test_type_conversion(device, src_type, in_values, dst_type, out_values, eps):
    src_nptype = dali_type_to_np(src_type)
    dst_nptype = dali_type_to_np(dst_type)
    assert len(out_values) == len(in_values)
    in_data = [np.full((100 + 10 * i,), x, src_nptype) for i, x in enumerate(in_values)]

    @pipeline_def(batch_size=len(in_values))
    def test_pipe(device):
        input = fn.external_source(in_data, batch=False, cycle="quiet", device=device)
        return fn.audio_resample(input, dtype=dst_type, scale=1, quality=0)

    pipe = test_pipe(device, device_id=0, num_threads=4)
    for _ in range(2):
        (out,) = pipe.run()
        assert len(out) == len(out_values)
        assert out.dtype == dst_type
        for i in range(len(out_values)):
            ref = np.full_like(in_data[i], out_values[i], dst_nptype)
            out_arr = as_array(out[i])
            if not np.allclose(out_arr, ref, 1e-6, eps):
                print("Actual: ", out_arr)
                print(out_arr.dtype, out_arr.shape)
                print("Reference: ", ref)
                print(ref.dtype, ref.shape)
                print("Diff: ", out_arr.astype(float) - ref)
                assert np.allclose(out_arr, ref, 1e-6, eps)


def test_dynamic_ranges():
    for type, values, eps in [
        (
            types.FLOAT,
            [-1.0e30, -1 - 1.0e-6, -1, -0.5, -1.0e-30, 0, 1.0e-30, 0.5, 1, 1 + 1.0e-6, 1e30],
            0,
        ),
        (types.UINT8, [0, 1, 128, 254, 255], 0),
        (types.INT8, [-128, -127, -1, 0, 1, 127], 0),
        (types.UINT16, [0, 1, 32767, 32768, 65534, 65535], 0),
        (types.INT16, [-32768, -32767, -100, -1, 0, 1, 100, 32767], 0),
        (types.UINT32, [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF], 128),
        (types.INT32, [-0x80000000, -0x7FFFFFFF, -100, -1, 0, 1, 0x7FFFFFFF], 128),
    ]:
        for device in ("cpu", "gpu"):
            yield _test_type_conversion, device, type, values, type, values, eps


def test_type_conversion():
    type_ranges = [
        (types.FLOAT, [-1, 1]),
        (types.UINT8, [0, 255]),
        (types.INT8, [-127, 127]),
        (types.UINT16, [0, 65535]),
        (types.INT16, [-32767, 32767]),
        (types.INT32, [-0x7FFFFFFF, 0x7FFFFFFF]),
        (types.UINT32, [0, 0xFFFFFFFF]),
    ]

    for src_type, src_range in type_ranges:
        i_lo, i_hi = src_range
        if i_lo == -i_hi:
            in_values = [i_lo, 0, i_hi]
        else:
            in_values = [i_lo, (i_lo + i_hi) // 2, (i_lo + i_hi + 1) // 2, i_hi]

        for dst_type, dst_range in type_ranges:
            o_lo, o_hi = dst_range
            if len(in_values) == 3:
                if o_lo != -o_hi:
                    out_values = [o_lo, (o_hi + o_lo + 1) / 2, o_hi]  # rounding
                else:
                    out_values = [o_lo, 0, o_hi]
            else:
                out_values = [
                    o_lo,
                    o_lo + (o_hi - o_lo) * in_values[1] / (i_hi - i_lo),
                    o_lo + (o_hi - o_lo) * in_values[2] / (i_hi - i_lo),
                    o_hi,
                ]
            if dst_type != types.FLOAT:
                out_values = list(map(int, out_values))
            eps = (o_hi - o_lo) / 2**24 + (i_hi - i_lo) / 2**24
            print(src_type, in_values, dst_type, out_values)

            # the result will be halfway - add epsilon of 1
            if eps < 1 and (o_lo != -o_hi or (i_hi != i_lo and dst_type != types.FLOAT)):
                eps = 1

            for device in ("cpu", "gpu"):
                yield _test_type_conversion, device, src_type, in_values, dst_type, out_values, eps
