# Copyright (c) 2019, 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from numpy.testing import assert_array_equal
import os
import tempfile
import shutil

from test_utils import compare_pipelines
from test_utils import get_dali_extra_path
from nose_utils import assert_raises, SkipTest

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, "db", "lmdb")
c2lmdb_db_folder = os.path.join(test_data_root, "db", "c2lmdb")
c2lmdb_no_label_db_folder = os.path.join(test_data_root, "db", "c2lmdb_no_label")


class CaffeReaderPipeline(Pipeline):
    def __init__(self, path, batch_size, num_threads=1, device_id=0, num_gpus=1):
        super(CaffeReaderPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.readers.Caffe(path=path, shard_id=device_id, num_shards=num_gpus)

        self.decode = ops.decoders.ImageCrop(
            device="cpu", crop=(224, 224), crop_pos_x=0.3, crop_pos_y=0.2, output_type=types.RGB
        )

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        return images, labels


def check_reader_path_vs_paths(paths, batch_size1, batch_size2, num_threads1, num_threads2):
    """
    test: compare caffe_db_folder with [caffe_db_folder] and [caffe_db_folder, caffe_db_folder],
    with different batch_size and num_threads
    """
    pipe1 = CaffeReaderPipeline(caffe_db_folder, batch_size1, num_threads1)

    pipe2 = CaffeReaderPipeline(paths, batch_size2, num_threads2)

    def Seq(pipe):
        while True:
            pipe_out = pipe.run()
            for idx in range(len(pipe_out[0])):
                yield pipe_out[0].at(idx), pipe_out[1].at(idx)

    seq1 = Seq(pipe1)
    seq2 = Seq(pipe2)

    num_entries = 100
    for i in range(num_entries):
        image1, label1 = next(seq1)
        image2, label2 = next(seq2)
        assert_array_equal(image1, image2)
        assert_array_equal(label1, label2)


def test_reader_path_vs_paths():
    for paths in [[caffe_db_folder], [caffe_db_folder, caffe_db_folder]]:
        for batch_size1 in {1}:
            for batch_size2 in {1, 16, 31}:
                for num_threads1 in {1}:
                    for num_threads2 in {1, 2}:
                        yield (
                            check_reader_path_vs_paths,
                            paths,
                            batch_size1,
                            batch_size2,
                            num_threads1,
                            num_threads2,
                        )


batch_size_alias_test = 64


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def caffe_pipe(caffe_op, path):
    data, label = caffe_op(path=path)
    return data, label


def test_caffe_reader_alias():
    new_pipe = caffe_pipe(fn.readers.caffe, caffe_db_folder)
    legacy_pipe = caffe_pipe(fn.caffe_reader, caffe_db_folder)
    compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 50)


def test_caffe_sharding():
    @pipeline_def(batch_size=1, device_id=0, seed=123, num_threads=1)
    def pipeline(shard_id, num_shards, stick_to_shard):
        images, _ = fn.readers.caffe(
            name="Reader",
            path=caffe_db_folder,
            pad_last_batch=True,
            random_shuffle=False,
            shard_id=shard_id,
            stick_to_shard=stick_to_shard,
            num_shards=num_shards,
        )
        return images

    def get_data(shard_id, num_shards, stick_to_shard):
        p = pipeline(shard_id, num_shards, stick_to_shard)
        size = p.reader_meta()["Reader"]["epoch_size_padded"]

        # This should return some unique number for each sample
        def sample_id(sample):
            return sample.as_array().sum()

        return {sample_id(p.run()[0]) for _ in range(size)}

    dataset = get_data(0, 1, False)

    num_shards = 3

    # Shards do not overlap
    shard = [get_data(i, num_shards, True) for i in range(num_shards)]
    for i in range(num_shards):
        for j in range(num_shards):
            if i != j:
                assert len(shard[i] & shard[j]) == 0, "overlapping shards"

    # Shards add up to whole dataset
    assert set().union(*shard) == dataset, "shards don't add up"

    # With stick_to_shard = False we traverse whole dataset
    for i in range(num_shards):
        result = get_data(i, num_shards, False)
        assert result == dataset, "starting shard changes the data"


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def caffe2_pipe(caffe2_op, path, label_type):
    if label_type == 4:
        data = caffe2_op(path=path, label_type=label_type)
        return data
    else:
        data, label = caffe2_op(path=path, label_type=label_type)
        return data, label


def check_caffe2(label_type):
    path = c2lmdb_no_label_db_folder if label_type == 4 else c2lmdb_db_folder
    new_pipe = caffe2_pipe(fn.readers.caffe2, path, label_type)
    legacy_pipe = caffe2_pipe(fn.caffe2_reader, path, label_type)
    compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 50)


def test_caffe2_reader_alias():
    for label_type in [0, 4]:
        yield check_caffe2, label_type


# Minimal protobuf wire encoding for Caffe2 TensorProtos (used for negative tests).
# TensorProtos: field 1 = repeated TensorProto.
# TensorProto: 1=dims, 2=data_type, 4=int32_data, 5=byte_data.
def _tensor_protos_bytes_label_indices_out_of_bounds():
    """One TensorProto: dims=[1], data_type=INT32(2), int32_data=[999].
    For MULTI_LABEL_SPARSE with num_labels=10."""
    # TensorProto: 08 01 (dims 1) 10 02 (INT32) 22 02 E7 07 (int32_data packed: 999)
    tp = bytes([0x08, 0x01, 0x10, 0x02, 0x22, 0x02, 0xE7, 0x07])
    return bytes([0x0A, len(tp)]) + tp


def _tensor_protos_bytes_label_weighted_indices_out_of_bounds():
    """Two TensorProtos: indices [999], weights [1.0].
    For MULTI_LABEL_WEIGHTED_SPARSE with num_labels=10."""
    # indices: dims=[1], INT32, int32_data=[999]
    tp0 = bytes([0x08, 0x01, 0x10, 0x02, 0x22, 0x02, 0xE7, 0x07])
    # weights: dims=[1], FLOAT, float_data=[1.0] (4 bytes LE)
    tp1 = bytes([0x08, 0x01, 0x10, 0x01, 0x22, 0x04, 0x00, 0x00, 0x80, 0x3F])
    return bytes([0x0A, len(tp0)]) + tp0 + bytes([0x0A, len(tp1)]) + tp1


def _tensor_protos_bytes_image_byte_data_size_mismatch():
    """Image BYTE dims [2,2,1] (H*W*C=4) but byte_data of length 3. Then single label [0]."""
    # Image: dims 2,2,1; data_type BYTE(3); byte_data (field 5) length 3
    img_tp = bytes([0x08, 0x02, 0x08, 0x02, 0x08, 0x01, 0x10, 0x03, 0x2A, 0x03, 0x00, 0x00, 0x00])
    # Label: dims=[1], INT32, int32_data=[0]
    label_tp = bytes([0x08, 0x01, 0x10, 0x02, 0x22, 0x01, 0x00])
    return bytes([0x0A, len(img_tp)]) + img_tp + bytes([0x0A, len(label_tp)]) + label_tp


def _create_c2_lmdb_with_bytes(serialized_tensor_protos):
    """Create a temp Caffe2 LMDB dir with one key-value. Returns path to the dir."""
    try:
        import lmdb
    except ImportError:
        raise SkipTest("lmdb package required for Caffe2 parser negative tests")
    tmpdir = tempfile.mkdtemp(prefix="dali_c2lmdb_")
    try:
        env = lmdb.open(tmpdir, map_size=1024 * 1024, subdir=True)
        with env.begin(write=True) as txn:
            txn.put(b"00000000", serialized_tensor_protos)
        env.close()
        return tmpdir
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def test_caffe2_parser_label_index_out_of_bounds_sparse():
    data = _tensor_protos_bytes_label_indices_out_of_bounds()
    path = _create_c2_lmdb_with_bytes(data)
    try:

        @pipeline_def(batch_size=1, device_id=0, num_threads=1)
        def pipe():
            data, labels = fn.readers.caffe2(
                path=path,
                image_available=False,
                label_type=1,  # MULTI_LABEL_SPARSE
                num_labels=10,
            )
            return data, labels

        p = pipe()
        p.build()
        with assert_raises(RuntimeError, glob="Label index out of bounds*num_labels*"):
            p.run()
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_caffe2_parser_label_index_out_of_bounds_weighted_sparse():
    data = _tensor_protos_bytes_label_weighted_indices_out_of_bounds()
    path = _create_c2_lmdb_with_bytes(data)
    try:

        @pipeline_def(batch_size=1, device_id=0, num_threads=1)
        def pipe():
            data, labels = fn.readers.caffe2(
                path=path,
                image_available=False,
                label_type=3,  # MULTI_LABEL_WEIGHTED_SPARSE
                num_labels=10,
            )
            return data, labels

        p = pipe()
        p.build()
        with assert_raises(RuntimeError, glob="Label index out of bounds*num_labels*"):
            p.run()
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_caffe2_parser_image_byte_data_size_mismatch():
    data = _tensor_protos_bytes_image_byte_data_size_mismatch()
    path = _create_c2_lmdb_with_bytes(data)
    try:

        @pipeline_def(batch_size=1, device_id=0, num_threads=1)
        def pipe():
            data, labels = fn.readers.caffe2(
                path=path,
                image_available=True,
                label_type=0,  # SINGLE_LABEL
                num_labels=1,
            )
            return data, labels

        p = pipe()
        p.build()
        with assert_raises(RuntimeError, glob="Image data size mismatch*"):
            p.run()
    finally:
        shutil.rmtree(path, ignore_errors=True)
