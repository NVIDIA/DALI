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

from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import scipy.io.wavfile
import numpy as np
import os
from test_audio_decoder_utils import generate_waveforms, rosa_resample
from test_utils import compare_pipelines, get_files
from nose_utils import attr

names = ["/tmp/dali_test_1C.wav", "/tmp/dali_test_2C.wav", "/tmp/dali_test_4C.wav"]

freqs = [np.array([0.02]), np.array([0.01, 0.012]), np.array([0.01, 0.012, 0.013, 0.014])]
rates = [16000, 22050, 12347]
lengths = [10000, 54321, 12345]


def create_test_files():
    for i in range(len(names)):
        wave = generate_waveforms(lengths[i], freqs[i])
        wave = (wave * 32767).round().astype(np.int16)
        scipy.io.wavfile.write(names[i], rates[i], wave)


create_test_files()

rate1 = 16000
rate2 = 12999


class DecoderPipeline(Pipeline):
    def __init__(self):
        super().__init__(
            batch_size=8,
            num_threads=3,
            device_id=0,
            exec_async=True,
            exec_pipelined=True,
            output_dtype=[
                types.INT16,
                types.INT16,
                types.INT16,
                types.FLOAT,
                types.FLOAT,
                types.FLOAT,
                types.FLOAT,
                types.FLOAT,
            ],
            output_ndim=[2, 2, 1, 1, 0, 0, 0, 0],
        )
        self.file_source = ops.ExternalSource()
        self.plain_decoder = ops.decoders.Audio(dtype=types.INT16)
        self.resampling_decoder = ops.decoders.Audio(sample_rate=rate1, dtype=types.INT16)
        self.downmixing_decoder = ops.decoders.Audio(downmix=True, dtype=types.INT16)
        self.resampling_downmixing_decoder = ops.decoders.Audio(
            sample_rate=rate2, downmix=True, quality=50, dtype=types.FLOAT
        )

    def define_graph(self):
        self.raw_file = self.file_source()
        dec_plain, rates_plain = self.plain_decoder(self.raw_file)
        dec_res, rates_res = self.resampling_decoder(self.raw_file)
        dec_mix, rates_mix = self.downmixing_decoder(self.raw_file)
        dec_res_mix, rates_res_mix = self.resampling_downmixing_decoder(self.raw_file)
        out = [
            dec_plain,
            dec_res,
            dec_mix,
            dec_res_mix,
            rates_plain,
            rates_res,
            rates_mix,
            rates_res_mix,
        ]
        return out

    def iter_setup(self):
        list = []
        for i in range(self.batch_size):
            idx = i % len(names)
            with open(names[idx], mode="rb") as f:
                list.append(np.array(bytearray(f.read()), np.uint8))
        self.feed_input(self.raw_file, list)


@attr("sanitizer_skip")
def test_decoded_vs_generated():
    pipeline = DecoderPipeline()
    idx = 0
    for iter in range(1):
        out = pipeline.run()
        for i in range(len(out[0])):
            plain = out[0].at(i)
            res = out[1].at(i)
            mix = out[2].at(i)[:, np.newaxis]
            res_mix = out[3].at(i)[:, np.newaxis]

            ref_len = [0, 0, 0, 0]
            ref_len[0] = lengths[idx]
            ref_len[1] = lengths[idx] * rate1 / rates[idx]
            ref_len[2] = lengths[idx]
            ref_len[3] = lengths[idx] * rate2 / rates[idx]

            ref0 = generate_waveforms(ref_len[0], freqs[idx]) * 32767
            ref1 = generate_waveforms(ref_len[1], freqs[idx] * (rates[idx] / rate1)) * 32767
            ref2 = generate_waveforms(ref_len[2], freqs[idx]) * 32767
            ref2 = ref2.mean(axis=1, keepdims=1)
            ref3 = generate_waveforms(ref_len[3], freqs[idx] * (rates[idx] / rate2))
            ref3 = ref3.mean(axis=1, keepdims=1)

            assert out[4].at(i) == rates[idx]
            assert out[5].at(i) == rate1
            assert out[6].at(i) == rates[idx]
            assert out[7].at(i) == rate2

            # just reading - allow only for rounding
            assert np.allclose(plain, ref0, rtol=0, atol=0.5)
            # resampling - allow for 1e-3 dynamic range error
            assert np.allclose(res, ref1, rtol=0, atol=32767 * 1e-3)
            # downmixing - allow for 2 bits of error
            # - one for quantization of channels, one for quantization of result
            assert np.allclose(mix, ref2, rtol=0, atol=2)
            # resampling with weird ratio - allow for 3e-3 dynamic range error
            assert np.allclose(res_mix, ref3, rtol=0, atol=3e-3)

            rosa_in1 = plain.astype(np.float32)
            rosa1 = rosa_resample(rosa_in1, rates[idx], rate1)
            rosa_in3 = rosa_in1 / 32767
            rosa3 = rosa_resample(rosa_in3.mean(axis=1, keepdims=1), rates[idx], rate2)

            assert np.allclose(res, rosa1, rtol=0, atol=32767 * 1e-3)
            assert np.allclose(res_mix, rosa3, rtol=0, atol=3e-3)

            idx = (idx + 1) % len(names)


batch_size_alias_test = 16


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def decoder_pipe(decoder_op, fnames, sample_rate, downmix, quality, dtype):
    encoded, _ = fn.readers.file(files=fnames)
    decoded, rates = decoder_op(
        encoded, sample_rate=sample_rate, downmix=downmix, quality=quality, dtype=dtype
    )
    return decoded, rates


def check_audio_decoder_alias(sample_rate, downmix, quality, dtype):
    new_pipe = decoder_pipe(fn.decoders.audio, names, sample_rate, downmix, quality, dtype)
    legacy_pipe = decoder_pipe(fn.audio_decoder, names, sample_rate, downmix, quality, dtype)
    compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 10)


def test_audio_decoder_alias():
    for sample_rate in [None, 16000, 12999]:
        for downmix in [False, True]:
            for quality in [0, 50, 100]:
                for dtype in [types.INT16, types.INT32, types.FLOAT]:
                    yield check_audio_decoder_alias, sample_rate, downmix, quality, dtype


def check_audio_decoder_correctness(fmt, dtype):
    batch_size = 16
    niterations = 10

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def audio_decoder_pipe(fnames, dtype, downmix=False):
        encoded, _ = fn.readers.file(files=fnames)
        decoded, _ = fn.decoders.audio(encoded, dtype=dtype, downmix=downmix)
        return decoded

    audio_files = get_files(os.path.join("db", "audio", fmt), fmt)
    npy_files = [os.path.splitext(fpath)[0] + ".npy" for fpath in audio_files]
    pipe = audio_decoder_pipe(audio_files, dtype)
    for it in range(niterations):
        data = pipe.run()
        for s in range(batch_size):
            sample_idx = (it * batch_size + s) % len(audio_files)
            ref = np.load(npy_files[sample_idx])
            if len(ref.shape) == 1:
                ref = np.expand_dims(ref, 1)
            arr = np.array(data[0][s])
            assert arr.shape == ref.shape
            if fmt == "ogg":
                # For OGG Vorbis, we consider errors any value that is off by more than 1
                # TODO(janton): There is a bug in libsndfile that produces underflow/overflow.
                #               Remove this when the bug is fixed.
                # Tuple with two arrays, we just need the first dimension
                wrong_values = np.where(np.abs(arr - ref) > 1)[0]
                nerrors = len(wrong_values)
                assert nerrors <= 1
                # TODO(janton): Uncomment this when the bug is fixed
                # np.testing.assert_allclose(arr, ref, atol=1)
            else:
                np.testing.assert_equal(arr, ref)


def test_audio_decoder_correctness():
    dtype = types.INT16
    for fmt in ["wav", "flac", "ogg"]:
        yield check_audio_decoder_correctness, fmt, dtype
