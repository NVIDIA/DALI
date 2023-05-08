# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import os.path
import tempfile
import numpy as np
from test_utils import compare_pipelines, get_dali_extra_path
from nose_utils import assert_raises


def skip_second(src, dst):
    with open(src, 'r') as tmp_f:
        with open(dst, 'w') as f:
            second = False
            for x in tmp_f:
                if not second:
                    f.write(x)
                second = not second


def test_tfrecord():
    class TFRecordPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, data, data_idx):
            super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.TFRecord(
                path=data,
                index_path=data_idx,
                features={
                    "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                    "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)})

        def define_graph(self):
            inputs = self.input(name="Reader")
            images = inputs["image/encoded"]
            return images

    tfrecord = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train')
    tfrecord_idx_org = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train.idx')
    tfrecord_idx = "tfr_train.idx"

    idx_files_dir = tempfile.TemporaryDirectory()
    idx_file = os.path.join(idx_files_dir.name, tfrecord_idx)

    skip_second(tfrecord_idx_org, idx_file)

    pipe = TFRecordPipeline(1, 1, 0, 1, tfrecord, idx_file)
    pipe_org = TFRecordPipeline(1, 1, 0, 1, tfrecord, tfrecord_idx_org)
    pipe.build()
    pipe_org.build()
    iters = pipe.epoch_size("Reader")
    for _ in range(iters):
        out = pipe.run()
        out_ref = pipe_org.run()
        for a, b in zip(out, out_ref):
            assert np.array_equal(a.as_array(), b.as_array())
        _ = pipe_org.run()


def test_recordio():
    class MXNetReaderPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, data, data_idx):
            super(MXNetReaderPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.readers.MXNet(path=[data], index_path=[data_idx],
                                           shard_id=device_id, num_shards=num_gpus)

        def define_graph(self):
            images, _ = self.input(name="Reader")
            return images

    recordio = os.path.join(get_dali_extra_path(), 'db', 'recordio', 'train.rec')
    recordio_idx_org = os.path.join(get_dali_extra_path(), 'db', 'recordio', 'train.idx')
    recordio_idx = "rio_train.idx"

    idx_files_dir = tempfile.TemporaryDirectory()
    idx_file = os.path.join(idx_files_dir.name, recordio_idx)

    skip_second(recordio_idx_org, idx_file)

    pipe = MXNetReaderPipeline(1, 1, 0, 1, recordio, idx_file)
    pipe_org = MXNetReaderPipeline(1, 1, 0, 1, recordio, recordio_idx_org)
    pipe.build()
    pipe_org.build()
    iters = pipe.epoch_size("Reader")
    for _ in range(iters):
        out = pipe.run()
        out_ref = pipe_org.run()
        for a, b in zip(out, out_ref):
            assert np.array_equal(a.as_array(), b.as_array())
        _ = pipe_org.run()


def test_wrong_feature_shape():
    features = {
        'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
        'image/object/bbox': tfrec.FixedLenFeature([], tfrec.float32, -1.0),
        'image/object/class/label': tfrec.FixedLenFeature([], tfrec.int64, -1),
    }
    test_dummy_data_path = os.path.join(get_dali_extra_path(), 'db', 'coco_dummy')
    pipe = Pipeline(1, 1, 0)
    with pipe:
        input = fn.readers.tfrecord(
            path=os.path.join(test_dummy_data_path, 'small_coco.tfrecord'),
            index_path=os.path.join(test_dummy_data_path, 'small_coco_index.idx'),
            features=features)
    pipe.set_outputs(input['image/encoded'],
                     input['image/object/class/label'],
                     input['image/object/bbox'])
    pipe.build()
    # the error is raised because FixedLenFeature is used with insufficient shape to house the input
    assert_raises(RuntimeError,
                  pipe.run,
                  glob="Error when executing CPU operator*readers*tfrecord*"
                  "Output tensor shape is too small*[]*Expected at least 4 elements")


batch_size_alias_test = 64


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def mxnet_pipe(mxnet_op, path, index_path):
    files, labels = mxnet_op(path=path, index_path=index_path)
    return files, labels


def test_mxnet_reader_alias():
    recordio = [os.path.join(get_dali_extra_path(), 'db', 'recordio', 'train.rec')]
    recordio_idx = [os.path.join(get_dali_extra_path(), 'db', 'recordio', 'train.idx')]
    new_pipe = mxnet_pipe(fn.readers.mxnet, recordio, recordio_idx)
    legacy_pipe = mxnet_pipe(fn.mxnet_reader, recordio, recordio_idx)
    compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 50)


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def tfrecord_pipe(tfrecord_op, path, index_path):
    inputs = tfrecord_op(
        path=path,
        index_path=index_path,
        features={
            "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1)})
    return inputs["image/encoded"]


def test_tfrecord_reader_alias():
    tfrecord = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train')
    tfrecord_idx = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train.idx')
    new_pipe = tfrecord_pipe(fn.readers.tfrecord, tfrecord, tfrecord_idx)
    legacy_pipe = tfrecord_pipe(fn.tfrecord_reader, tfrecord, tfrecord_idx)
    compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 50)


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def tfrecord_pipe_empty_fields(path, index_path):
    inputs = fn.readers.tfrecord(
        path=path, index_path=index_path,
        features={"image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                  "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
                  "does/not/exists": tfrec.VarLenFeature(tfrec.int64, -1),
                  "does/not/exists/as/well": tfrec.FixedLenFeature([1], tfrec.float32, .0)})
    return inputs["image/encoded"], inputs["does/not/exists"], inputs["does/not/exists/as/well"]


def test_tfrecord_reader_alias2():
    tfrecord = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train')
    tfrecord_idx = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train.idx')
    pipe = tfrecord_pipe_empty_fields(tfrecord, tfrecord_idx)
    pipe.build()
    out = pipe.run()
    for tensor in out[0]:
        data = np.array(tensor)
        assert len(data) != 0
        assert data.dtype == np.uint8
    for tensor in out[1]:
        data = np.array(tensor)
        assert len(data.shape) == 0
        assert data.dtype == np.int64
    for tensor in out[2]:
        data = np.array(tensor)
        assert len(data.shape) == 0
        assert data.dtype == np.float32


def test_tfrecord_reader_scalars():
    test_dummy_data_path = os.path.join(get_dali_extra_path(), 'db', 'coco_dummy')

    @pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
    def tfrecord_pipe_scalars():
        data = fn.readers.tfrecord(
            path=os.path.join(test_dummy_data_path, 'small_coco.tfrecord'),
            index_path=os.path.join(test_dummy_data_path, 'small_coco_index.idx'),
            features={
                'image/height': tfrec.FixedLenFeature((), tfrec.int64, -1),
            })
        return data['image/height']
    pipe = tfrecord_pipe_scalars()
    pipe.build()
    out = pipe.run()

    for tensor in out[0]:
        data = np.array(tensor)
        assert data.dtype == np.int64
        assert data.shape == (), f"Unexpected shape. Expected scalar, got {data.shape}"


def test_conditionals():
    tfrecord = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train')
    tfrecord_idx = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train.idx')

    @pipeline_def()
    def get_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, num_gpus):

        inputs = fn.readers.tfrecord(
            path=tfrec_filenames, index_path=tfrec_idx_filenames, random_shuffle=True,
            shard_id=shard_id, num_shards=num_gpus, initial_fill=10000, seed=42, features={
                'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                'image/class/text': tfrec.FixedLenFeature([], tfrec.string, ''),
                'image/object/bbox/xmin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'image/object/bbox/ymin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'image/object/bbox/xmax': tfrec.VarLenFeature(tfrec.float32, 0.0),
                'image/object/bbox/ymax': tfrec.VarLenFeature(tfrec.float32, 0.0)
            })

        encoded = inputs["image/encoded"]
        images = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
        images = fn.resize(images, device="gpu", resize_shorter=256)

        labels = inputs["image/class/label"].gpu()

        labels -= 1  # Change to 0-based (don't use background class)
        return images, labels

    pipe_base = get_dali_pipeline(tfrecord, tfrecord_idx, shard_id=0, num_gpus=1, device_id=0,
                                  num_threads=4, batch_size=32)

    pipe_cond = get_dali_pipeline(tfrecord, tfrecord_idx, shard_id=0, num_gpus=1, device_id=0,
                                  num_threads=4, batch_size=32, enable_conditionals=True)
    for pipe in [pipe_base, pipe_cond]:
        pipe.build()
    compare_pipelines(pipe_base, pipe_cond, 32, 5)
