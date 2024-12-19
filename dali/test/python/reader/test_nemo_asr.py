# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali import fn
import nvidia.dali.types as types
import scipy.io.wavfile
import numpy as np
import json
import tempfile
import os
from test_audio_decoder_utils import generate_waveforms
from test_utils import compare_pipelines


def create_manifest_file(manifest_file, names, lengths, rates, texts):
    assert len(names) == len(lengths) == len(rates) == len(texts)
    data = []
    for idx in range(len(names)):
        entry_i = {}
        entry_i["audio_filepath"] = names[idx]
        entry_i["duration"] = lengths[idx] * (1.0 / rates[idx])
        entry_i["text"] = texts[idx]
        data.append(entry_i)
    with open(manifest_file, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


tmp_dir = tempfile.TemporaryDirectory()

names = [
    os.path.join(tmp_dir.name, "dali_test_1C.wav"),
    os.path.join(tmp_dir.name, "dali_test_2C.wav"),
    os.path.join(tmp_dir.name, "dali_test_4C.wav"),
]

freqs = [np.array([0.02]), np.array([0.01, 0.012]), np.array([0.01, 0.012, 0.013, 0.014])]
rates = [22050, 22050, 12347]
lengths = [10000, 54321, 12345]


def create_ref():
    ref = []
    for i in range(len(names)):
        wave = generate_waveforms(lengths[i], freqs[i])
        wave = (wave * 32767).round().astype(np.int16)
        ref.append(wave)
    return ref


ref_i = create_ref()


def create_wav_files():
    for i in range(len(names)):
        scipy.io.wavfile.write(names[i], rates[i], ref_i[i])


create_wav_files()

ref_text_literal = [
    "dali test 1C",
    "dali test 2C",
    "dali test 4C",
]
nemo_asr_manifest = os.path.join(tmp_dir.name, "nemo_asr_manifest.json")
create_manifest_file(nemo_asr_manifest, names, lengths, rates, ref_text_literal)
ref_text = [np.frombuffer(bytes(s, "utf8"), dtype=np.uint8) for s in ref_text_literal]

ref_text_non_ascii_literal = [
    "dzień dobry",
    "доброе утро",
    "这是一个测试",
]
nemo_asr_manifest_non_ascii = os.path.join(tmp_dir.name, "nemo_asr_manifest_non_ascii.json")
create_manifest_file(nemo_asr_manifest_non_ascii, names, lengths, rates, ref_text_non_ascii_literal)
ref_text_non_ascii = [
    np.frombuffer(bytes(s, "utf8"), dtype=np.uint8) for s in ref_text_non_ascii_literal
]

rate1 = 16000
rate2 = 44100


class NemoAsrReaderPipeline(Pipeline):
    def __init__(self, batch_size=8):
        super(NemoAsrReaderPipeline, self).__init__(
            batch_size=batch_size, num_threads=1, device_id=0, exec_async=True, exec_pipelined=True
        )

    def define_graph(self):
        fixed_seed = 12345
        audio_plain_i = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.INT16,
            downmix=False,
            read_sample_rate=False,
            read_text=False,
            seed=fixed_seed,
        )
        audio_plain_f = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.FLOAT,
            downmix=False,
            read_sample_rate=False,
            read_text=False,
            seed=fixed_seed,
        )
        audio_downmix_i = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.INT16,
            downmix=True,
            read_sample_rate=False,
            read_text=False,
            seed=fixed_seed,
        )
        audio_downmix_f = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.FLOAT,
            downmix=True,
            read_sample_rate=False,
            read_text=False,
            seed=fixed_seed,
        )
        audio_resampled1_i, _ = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.INT16,
            downmix=True,
            sample_rate=rate1,
            read_sample_rate=True,
            read_text=False,
            seed=fixed_seed,
        )
        audio_resampled1_f, _ = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.FLOAT,
            downmix=True,
            sample_rate=rate1,
            read_sample_rate=True,
            read_text=False,
            seed=fixed_seed,
        )
        audio_resampled2_i, _ = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.INT16,
            downmix=True,
            sample_rate=rate2,
            read_sample_rate=True,
            read_text=False,
            seed=fixed_seed,
        )
        audio_resampled2_f, _ = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.FLOAT,
            downmix=True,
            sample_rate=rate2,
            read_sample_rate=True,
            read_text=False,
            seed=fixed_seed,
        )
        _, _, text = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            dtype=types.INT16,
            downmix=True,
            read_sample_rate=True,
            read_text=True,
            seed=fixed_seed,
        )
        _, _, text_non_ascii = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest_non_ascii],
            dtype=types.INT16,
            downmix=True,
            read_sample_rate=True,
            read_text=True,
            seed=fixed_seed,
        )
        return (
            audio_plain_i,
            audio_plain_f,
            audio_downmix_i,
            audio_downmix_f,
            audio_resampled1_i,
            audio_resampled1_f,
            audio_resampled2_i,
            audio_resampled2_f,
            text,
            text_non_ascii,
        )


def test_decoded_vs_generated():
    batch_size = 3
    pipeline = NemoAsrReaderPipeline(batch_size=batch_size)

    for iter in range(1):
        out = pipeline.run()
        for idx in range(batch_size):
            audio_plain_i = out[0].at(idx)
            audio_plain_f = out[1].at(idx)
            audio_downmix_i = out[2].at(idx)
            audio_downmix_f = out[3].at(idx)
            audio_resampled1_i = out[4].at(idx)
            audio_resampled1_f = out[5].at(idx)
            audio_resampled2_i = out[6].at(idx)
            audio_resampled2_f = out[7].at(idx)
            text = out[8].at(idx)
            text_non_ascii = out[9].at(idx)

            ref_plain_i = ref_i[idx]
            np.testing.assert_allclose(audio_plain_i, ref_plain_i, rtol=1e-7)

            ref_plain_f = ref_i[idx].astype(np.float32) / 32767
            np.testing.assert_allclose(audio_plain_f, ref_plain_f, rtol=1e-4)

            ref_downmix_i_float = ref_i[idx].astype(np.float32).mean(axis=1, keepdims=1)

            ref_downmix_i = ref_downmix_i_float.astype(np.int16).flatten()
            np.testing.assert_allclose(audio_downmix_i, ref_downmix_i, atol=1)

            ref_downmix_f = (ref_downmix_i_float / 32767).flatten()
            np.testing.assert_allclose(audio_downmix_f, ref_downmix_f, rtol=1e-4)

            ref_resampled1_float = generate_waveforms(
                lengths[idx] * rate1 / rates[idx], freqs[idx] * (rates[idx] / rate1)
            )
            ref_resampled1_downmix = ref_resampled1_float.astype(np.float32).mean(
                axis=1, keepdims=1
            )
            ref_resampled1_i = (ref_resampled1_downmix * 32767).astype(np.int16).flatten()
            # resampling - allow for 1e-3 dynamic range error
            np.testing.assert_allclose(
                audio_resampled1_i, ref_resampled1_i, atol=round(32767 * 1e-3)
            )

            ref_resampled1_f = ref_resampled1_downmix.flatten()
            # resampling - allow for 1e-3 dynamic range error
            np.testing.assert_allclose(audio_resampled1_f, ref_resampled1_f, atol=1e-3)

            ref_resampled2_float = generate_waveforms(
                lengths[idx] * rate2 / rates[idx], freqs[idx] * (rates[idx] / rate2)
            )
            ref_resampled2_downmix = ref_resampled2_float.astype(np.float32).mean(
                axis=1, keepdims=1
            )
            ref_resampled2_i = (ref_resampled2_downmix * 32767).astype(np.int16).flatten()
            # resampling - allow for 1e-3 dynamic range error
            np.testing.assert_allclose(
                audio_resampled2_i, ref_resampled2_i, atol=round(32767 * 1e-3)
            )

            ref_resampled2_f = ref_resampled2_downmix.flatten()
            # resampling - allow for 1e-3 dynamic range error
            np.testing.assert_allclose(audio_resampled2_f, ref_resampled2_f, atol=1e-3)

            np.testing.assert_equal(text, ref_text[idx])

            np.testing.assert_equal(text_non_ascii, ref_text_non_ascii[idx])
            text_non_ascii_str = str(text_non_ascii.tobytes(), encoding="utf8")

            # Checking that we don't have any trailing zeros
            # (those won't be caught by the string comparison)
            ref_text_non_ascii_literal_bytes = bytes(ref_text_non_ascii_literal[idx], "utf8")
            assert (
                text_non_ascii.tobytes() == ref_text_non_ascii_literal_bytes
            ), f"'{text_non_ascii.tobytes()}' != '{ref_text_non_ascii_literal_bytes}'"

            # String comparison (utf-8)
            assert (
                text_non_ascii_str == ref_text_non_ascii_literal[idx]
            ), f"'{text_non_ascii_str}' != '{ref_text_non_ascii_literal[idx]}'"


batch_size_alias_test = 64


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def nemo_pipe(nemo_op, path, read_text, read_sample_rate, dtype, downmix):
    if read_sample_rate:
        audio, sr = nemo_op(
            manifest_filepaths=path,
            read_sample_rate=read_sample_rate,
            read_text=read_text,
            dtype=dtype,
            downmix=downmix,
        )
        return audio, sr
    elif read_text:
        audio, text = nemo_op(
            manifest_filepaths=path,
            read_sample_rate=read_sample_rate,
            read_text=read_text,
            dtype=dtype,
            downmix=downmix,
        )
        return audio, text
    else:
        audio = nemo_op(
            manifest_filepaths=path,
            read_sample_rate=read_sample_rate,
            read_text=read_text,
            dtype=dtype,
            downmix=downmix,
        )
        return audio


def test_nemo_asr_reader_alias():
    for read_sr, read_text in [(True, False), (False, True), (False, False)]:
        for dtype in [types.INT16, types.FLOAT]:
            for downmix in [True, False]:
                new_pipe = nemo_pipe(
                    fn.readers.nemo_asr, [nemo_asr_manifest], read_sr, read_text, dtype, downmix
                )
                legacy_pipe = nemo_pipe(
                    fn.nemo_asr_reader, [nemo_asr_manifest], read_sr, read_text, dtype, downmix
                )
                compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 50)


def test_nemo_asr_reader_pad_last_batch():
    @pipeline_def(device_id=0, num_threads=4)
    def nemo_asr_pad_last_batch_pipe():
        audio = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            pad_last_batch=True,
            read_sample_rate=False,
            read_text=False,
        )

        return audio

    def _testimpl_nemo_asr_reader_pad_last_batch(batch_size):
        pipe = nemo_asr_pad_last_batch_pipe(batch_size=batch_size)

        dataset_len = len(names)
        assert dataset_len % batch_size > 0  # Checking that we need to pad
        sample_idx = 0
        for it in range(10):
            audio = pipe.run()[0]
            sample_idx = it * batch_size
            if sample_idx > dataset_len:
                sample_idx = 0
            last_sample = None
            padded_sample = None
            for i in range(batch_size):
                if sample_idx == dataset_len - 1:
                    last_sample = np.array(audio[i])
                elif sample_idx >= dataset_len:
                    padded_sample = np.array(audio[i])
                    np.testing.assert_array_equal(padded_sample, last_sample)
                sample_idx += 1

    # The manifest has 3 samples, of lengths 10000, 54321, 12345

    # With batch size 2, batches will contain lengths:
    # [10000, 54321], [12345, 12345], [10000, 54321], ...
    # This is meant to reproduce an error found when combining pad_last_sample=True,
    # using ShareData to replicate the last sample, and trying to resize to a bigger
    # buffer after ShareData.
    yield _testimpl_nemo_asr_reader_pad_last_batch, 2

    # Trying to catch race conditions (A lot of samples in the batch to be replicated)
    yield _testimpl_nemo_asr_reader_pad_last_batch, 128


def test_read_idxs():
    batch_size = 10
    reader_seed = 12345

    @pipeline_def(device_id=0, num_threads=4)
    def nemo_asr_reader_read_idxs(reader_seed=reader_seed):
        audio, idx = fn.readers.nemo_asr(
            manifest_filepaths=[nemo_asr_manifest],
            random_shuffle=True,
            seed=reader_seed,
            read_sample_rate=False,
            read_text=False,
            read_idxs=True,
        )
        return audio, idx

    seed = 12345
    pipe1 = nemo_asr_reader_read_idxs(batch_size=batch_size, reader_seed=seed)
    pipe2 = nemo_asr_reader_read_idxs(batch_size=batch_size, reader_seed=seed)

    total_samples = len(names)

    for iter in range(3):
        audio1, idx1 = pipe1.run()
        audio2, idx2 = pipe2.run()
        for s in range(batch_size):
            np.testing.assert_array_equal(np.array(audio1[s]), np.array(audio2[s]))
            np.testing.assert_array_equal(np.array(idx1[s]), np.array(idx2[s]))
            idx = np.array(idx1[s])[0]
            assert idx >= 0 and idx < total_samples
