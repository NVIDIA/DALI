# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os

from test_utils import check_batch
from test_utils import save_image
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')

class CaffeReaderPipeline(Pipeline):
    def __init__(self, path, batch_size, num_threads=1, device_id=0, num_gpus=1):
        super(CaffeReaderPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.input = ops.CaffeReader(path = path, shard_id = device_id, num_shards = num_gpus)

        self.decode = ops.ImageDecoderCrop(device = "cpu",
                                           crop = (224, 224),
                                           crop_pos_x = 0.3,
                                           crop_pos_y = 0.2,
                                           output_type = types.RGB)
    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        return images, labels

# test 1: compare caffe_db_folder with [caffe_db_folder] and [caffe_db_folder, caffe_db_folder]
def check_reader_vs_paths(path, batch_size, num_threads):
 
    pipe1 = CaffeReaderPipeline(caffe_db_folder, batch_size, num_threads)
    pipe1.build()

    pipe2 = CaffeReaderPipeline(path, batch_size, num_threads)
    pipe2.build()

    num_batches = 2
    for i in range(num_batches):
        pipe1_out = pipe1.run()
        pipe2_out = pipe2.run()
        for idx in range(len(pipe1_out[0])):
            image1 = pipe1_out[0].at(idx)
            label1 = pipe1_out[1].at(idx)
            image2 = pipe2_out[0].at(idx)
            label2 = pipe2_out[1].at(idx)

            assert_array_equal(image1, image2)
            assert_array_equal(label1, label2)

def test_reader_vs_paths():
    for path in [[caffe_db_folder], [caffe_db_folder, caffe_db_folder]]:
        for batch_size in {1, 32, 44}:
            for num_threads in {1, 2}:
                yield check_reader_vs_paths, path, batch_size, num_threads


# test 2: compare with a simple CaffeReader with batch_size=1
def check_reader_vs_simple(path, batch_size, num_threads, num_gpus, images, labels):
    pipelines = [CaffeReaderPipeline(path, batch_size, num_threads, device_id, num_gpus) for device_id in range(num_gpus)]
    num_paths = 1 if isinstance(path, str) else len(path)

    num_batches = 2
    for pipe in pipelines:
        pipe.build()
        count = 0
        for i in range(num_batches):
            pipe_out = pipe.run()
            for idx in range(len(pipe_out[0])):
                image = pipe_out[0].at(idx)
                label = pipe_out[1].at(idx)
                dstid = count % len(images)
                assert_array_equal(image, images[dstid])
                assert_array_equal(label, labels[dstid])
                count+=1

def test_reader_vs_simple():
    pipe = CaffeReaderPipeline(caffe_db_folder, 1)
    pipe.build()
    images = []
    labels = []
    num = 43
    for i in range(num):
        pipe_out = pipe.run()
        image = pipe_out[0].at(0)
        label = pipe_out[1].at(0)
        images.append(image)
        labels.append(label)

    num_gpus = 1
    for path in [caffe_db_folder, [caffe_db_folder], [caffe_db_folder, caffe_db_folder]]:
        for batch_size in {1, 32, 44}:
            for num_threads in {1, 2}:
                yield check_reader_vs_simple, path, batch_size, num_threads, num_gpus, images, labels


if __name__ == '__main__':
    # run this using previous CaffeReader to get the db statistics
    num = 43
    pipe = CaffeReaderPipeline(caffe_db_folder, 1)
    pipe.build()
    db_stat = []
    for i in range(num):
        pipe_out = pipe.run()
        image = pipe_out[0].at(0)
        label = pipe_out[1].at(0)
        print(['%.2f' % image.mean(), label[0]])


