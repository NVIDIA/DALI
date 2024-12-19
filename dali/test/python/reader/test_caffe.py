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

from test_utils import compare_pipelines
from test_utils import get_dali_extra_path

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
