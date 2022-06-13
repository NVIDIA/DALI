# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import numpy as np
import tempfile
import scipy
import scipy.io.wavfile

from test_audio_decoder_utils import generate_waveforms


def setup_test_nemo_asr_reader_cpu():
    tmp_dir = tempfile.TemporaryDirectory()
    dir_name = tmp_dir.name

    def create_manifest_file(manifest_file, names, lengths, rates, texts):
        assert(len(names) == len(lengths) == len(rates) == len(texts))
        data = []
        for idx in range(len(names)):
            entry_i = {}
            entry_i['audio_filepath'] = names[idx]
            entry_i['duration'] = lengths[idx] * (1.0 / rates[idx])
            entry_i["text"] = texts[idx]
            data.append(entry_i)
        with open(manifest_file, 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')
    nemo_asr_manifest = os.path.join(dir_name, "nemo_asr_manifest.json")
    names = [
        os.path.join(dir_name, "dali_test_1C.wav"),
        os.path.join(dir_name, "dali_test_2C.wav"),
        os.path.join(dir_name, "dali_test_4C.wav")
    ]

    freqs = [
        np.array([0.02]),
        np.array([0.01, 0.012]),
        np.array([0.01, 0.012, 0.013, 0.014])
    ]
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
    nemo_asr_manifest = os.path.join(dir_name, "nemo_asr_manifest.json")
    create_manifest_file(nemo_asr_manifest, names, lengths, rates, ref_text_literal)

    return tmp_dir, nemo_asr_manifest


def setup_test_numpy_reader_cpu():
    tmp_dir = tempfile.TemporaryDirectory()
    dir_name = tmp_dir.name

    rng = np.random.default_rng(12345)

    def create_numpy_file(filename, shape, typ, fortran_order):
        # generate random array
        arr = rng.random(shape) * 10.
        arr = arr.astype(typ)
        if fortran_order:
            arr = np.asfortranarray(arr)
        np.save(filename, arr)

    num_samples = 20
    filenames = []
    for index in range(0, num_samples):
        filename = os.path.join(dir_name, "test_{:02d}.npy".format(index))
        filenames.append(filename)
        create_numpy_file(filename, (5, 2, 8), np.float32, False)

    return tmp_dir
