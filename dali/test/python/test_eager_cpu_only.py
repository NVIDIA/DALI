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

import json
import numpy as np
import os
import tempfile
import scipy.io.wavfile
from functools import reduce

from nvidia.dali import pipeline_def
import nvidia.dali.experimental.eager as eager
import nvidia.dali.fn as fn
import nvidia.dali.tensors as tensors
import nvidia.dali.types as types
from test_audio_decoder_utils import generate_waveforms
from test_utils import check_batch, get_dali_extra_path, get_files
from webdataset_base import generate_temp_index_file as generate_temp_wds_index

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
caffe_dir = os.path.join(data_root, 'db', 'lmdb')
caffe2_dir = os.path.join(data_root, 'db', 'c2lmdb')
recordio_dir = os.path.join(data_root, 'db', 'recordio')
webdataset_dir = os.path.join(data_root, 'db', 'webdataset')
coco_dir = os.path.join(data_root, 'db', 'coco', 'images')
coco_annotation = os.path.join(data_root, 'db', 'coco', 'instances.json')
sequence_dir = os.path.join(data_root, 'db', 'sequence', 'frames')
video_files = get_files(os.path.join('db', 'video', 'vfr'), 'mp4')

rng = np.random.default_rng()

batch_size = 2


def get_ops(op_path, fn_op=None, eager_op=None):
    import_path = op_path.split('.')
    if fn_op is None:
        fn_op = reduce(getattr, [fn] + import_path)
    if eager_op is None:
        eager_op = reduce(getattr, [eager] + import_path)
    return fn_op, eager_op


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def no_input_pipeline(op, kwargs):
    out = op(**kwargs)
    if isinstance(out, list):
        out = tuple(out)
    return out


def check_reader(op_path, *, fn_op=None, eager_op=None, batch_size=batch_size, N_iterations=2, **kwargs):
    fn_op, eager_op = get_ops(op_path, fn_op, eager_op)
    pipe = no_input_pipeline(fn_op, kwargs)
    pipe.build()

    iter_eager = eager_op(batch_size=batch_size, **kwargs)

    for _ in range(N_iterations):
        for i, out2 in enumerate(iter_eager):
            out1 = pipe.run()

            if not isinstance(out2, (tuple, list)):
                out2 = (out2,)

            for o1, o2 in zip(out1, out2):
                if i == len(iter_eager) - 1:
                    o1 = type(o1)([o1[j] for j in range(len(o2))])
                out1_data = o1.as_cpu() if isinstance(o1, tensors.TensorListGPU) else o1
                out2_data = o2.as_cpu() if isinstance(o2, tensors.TensorListGPU) else o2

                check_batch(out1_data, out2_data, len(o2))


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=None)
def reader_op_pipeline(op, kwargs, source=None, layout=None):
    if source is None:
        raise RuntimeError('No source for file reader.')
    data, _ = fn.readers.file(file_root=source)
    out = op(data, **kwargs)
    if isinstance(out, list):
        out = tuple(out)
    return out


def test_file_reader_cpu():
    check_reader('readers.file', file_root=images_dir)


def test_mxnet_reader_cpu():
    check_reader('readers.mxnet', path=os.path.join(recordio_dir, 'train.rec'),
                 index_path=os.path.join(recordio_dir, 'train.idx'), shard_id=0, num_shards=1)


def test_webdataset_reader_cpu():
    webdataset = os.path.join(webdataset_dir, 'MNIST', 'devel-0.tar')
    webdataset_idx = generate_temp_wds_index(webdataset)
    check_reader('readers.webdataset',
                 paths=webdataset,
                 index_paths=webdataset_idx.name,
                 ext=['jpg', 'cls'],
                 shard_id=0, num_shards=1)


def test_coco_reader_cpu():
    check_reader('readers.coco', file_root=coco_dir,
                 annotations_file=coco_annotation, shard_id=0, num_shards=1)


def test_caffe_reader_cpu():
    check_reader('readers.caffe', path=caffe_dir, shard_id=0, num_shards=1)


def test_caffe2_reader_cpu():
    check_reader('readers.caffe2', path=caffe2_dir, shard_id=0, num_shards=1)


def test_nemo_asr_reader_cpu():
    with tempfile.TemporaryDirectory() as tmp_dir:
        def create_manifest_file(manifest_file, names, lengths, rates, texts):
            assert(len(names) == len(lengths) == len(rates) == len(texts))
            data = []
            for idx in range(len(names)):
                entry_i = {}
                entry_i['audio_filepath'] = names[idx]
                entry_i['duration'] = lengths[idx] * (1.0 / rates[idx])
                entry_i['text'] = texts[idx]
                data.append(entry_i)
            with open(manifest_file, 'w') as f:
                for entry in data:
                    json.dump(entry, f)
                    f.write('\n')
        nemo_asr_manifest = os.path.join(tmp_dir, 'nemo_asr_manifest.json')
        names = [
            os.path.join(tmp_dir, 'dali_test_1C.wav'),
            os.path.join(tmp_dir, 'dali_test_2C.wav'),
            os.path.join(tmp_dir, 'dali_test_4C.wav')
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
            'dali test 1C',
            'dali test 2C',
            'dali test 4C',
        ]
        nemo_asr_manifest = os.path.join(tmp_dir, 'nemo_asr_manifest.json')
        create_manifest_file(nemo_asr_manifest, names, lengths, rates, ref_text_literal)

        fixed_seed = 1234
        check_reader('readers.nemo_asr', manifest_filepaths=[nemo_asr_manifest], dtype=types.INT16,
                     downmix=False, read_sample_rate=True, read_text=True, seed=fixed_seed)


def test_video_reader():
    check_reader('experimental.readers.video', filenames=video_files,
                 labels=[0, 1], sequence_length=10)


def test_numpy_reader_cpu():
    with tempfile.TemporaryDirectory() as test_data_root:
        def create_numpy_file(filename, shape, typ, fortran_order):
            arr = rng.random(shape) * 10.
            arr = arr.astype(typ)
            if fortran_order:
                arr = np.asfortranarray(arr)
            np.save(filename, arr)

        num_samples = 20
        filenames = []
        for index in range(0, num_samples):
            filename = os.path.join(test_data_root, 'test_{:02d}.npy'.format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)

        check_reader('readers.numpy', file_root=test_data_root)


def test_sequence_reader_cpu():
    check_reader('readers.sequence', file_root=sequence_dir,
                 sequence_length=2, shard_id=0, num_shards=1)
