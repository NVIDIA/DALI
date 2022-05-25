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

import numpy as np
import os
from functools import reduce

from nvidia.dali import pipeline_def
import nvidia.dali.experimental.eager as eager
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from test_utils import check_batch, get_dali_extra_path, get_files
from test_dali_cpu_only_utils import *
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
def reader_pipeline(op, kwargs):
    out = op(pad_last_batch=True, **kwargs)
    if isinstance(out, list):
        out = tuple(out)
    return out


def check_reader(op_path, *, fn_op=None, eager_op=None, batch_size=batch_size, N_iterations=2, **kwargs):
    fn_op, eager_op = get_ops(op_path, fn_op, eager_op)
    pipe = reader_pipeline(fn_op, kwargs)
    pipe.build()

    iter_eager = eager_op(batch_size=batch_size, **kwargs)

    for _ in range(N_iterations):
        for i, out_eager in enumerate(iter_eager):
            out_fn = pipe.run()

            if not isinstance(out_eager, (tuple, list)):
                out_eager = (out_eager,)

            for tensor_out_fn, tensor_out_eager in zip(out_fn, out_eager):
                if i == len(iter_eager) - 1:
                    tensor_out_fn = type(tensor_out_fn)(
                        [tensor_out_fn[j] for j in range(len(tensor_out_eager))])

                assert type(tensor_out_fn) == type(tensor_out_eager)
                check_batch(tensor_out_fn, tensor_out_eager, len(tensor_out_eager))


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
    tmp_dir, nemo_asr_manifest = setup_test_nemo_asr_reader_cpu()

    with tmp_dir:
        check_reader('readers.nemo_asr', manifest_filepaths=[nemo_asr_manifest], dtype=types.INT16,
                     downmix=False, read_sample_rate=True, read_text=True, seed=1234)


def test_video_reader():
    check_reader('experimental.readers.video', filenames=video_files,
                 labels=[0, 1], sequence_length=10)


def test_numpy_reader_cpu():
    with setup_test_numpy_reader_cpu() as test_data_root:
        check_reader('readers.numpy', file_root=test_data_root)


def test_sequence_reader_cpu():
    check_reader('readers.sequence', file_root=sequence_dir,
                 sequence_length=2, shard_id=0, num_shards=1)
