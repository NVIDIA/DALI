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

# test 1: rewind
def check_reader_vs_rewind(path):
    # the db has 43 entries, 44 should trigger rewind
    if isinstance(path, str):
        num_paths = 1
    else:
        num_paths = len(path)
    num = 43*num_paths + 1
    pipe = CaffeReaderPipeline(path, 1)
    pipe.build()
    for i in range(num):
        pipe_out = pipe.run()
        if i == 0:
            image0 = pipe_out[0].at(0)
            label0 = pipe_out[1].at(0)
        if i == num-1:
            image0_rewind = pipe_out[0].at(0)
            label0_rewind = pipe_out[1].at(0)
        
    assert_array_equal(image0, image0_rewind)
    assert_array_equal(label0, label0_rewind)

def test_reader_vs_rewind():
    for path in [caffe_db_folder, [caffe_db_folder], [caffe_db_folder, caffe_db_folder]]:
        yield check_reader_vs_rewind, path

# test 2: compare with previous CaffeReader
# run the previous CaffeReader to get the db statistics [image.mean(), label], 
# regarded as ground truth
db_stat = [
    [118.10, 1],
    [95.21, 1],
    [105.89, 1],
    [135.78, 1],
    [152.72, 1],
    [93.08, 0],
    [153.97, 0],
    [177.89, 0],
    [163.62, 1],
    [82.04, 1],
    [214.37, 1],
    [77.89, 1],
    [231.43, 0],
    [100.98, 1],
    [157.01, 1],
    [167.31, 1],
    [98.75, 0],
    [101.37, 1],
    [125.64, 1],
    [120.80, 1],
    [128.53, 1],
    [95.86, 1],
    [56.78, 0],
    [89.86, 1],
    [86.71, 1],
    [118.33, 1],
    [76.55, 1],
    [108.08, 0],
    [147.95, 0],
    [188.69, 0],
    [135.47, 1],
    [164.58, 0],
    [172.21, 1],
    [111.00, 1],
    [175.26, 0],
    [99.32, 1],
    [138.10, 1],
    [97.69, 1],
    [108.19, 0],
    [60.03, 0],
    [145.58, 1],
    [135.89, 1],
    [141.06, 0],
]

def check_reader_vs_db_stat(path, batch_size, num_threads, num_gpus):
    pipelines = [CaffeReaderPipeline(path, batch_size, num_threads, device_id, num_gpus) for device_id in range(num_gpus)]
    
    num_batches = 2
    for pipe in pipelines:
        pipe.build()
        count = 0
        for i in range(num_batches):
            pipe_out = pipe.run()
            for idx in range(len(pipe_out[0])):
                image = pipe_out[0].at(idx)
                label = pipe_out[1].at(idx)
                dstid = count % len(db_stat)
                assert_allclose(image.mean(), db_stat[dstid][0], rtol=1e-3)
                assert_array_equal(label[0], db_stat[dstid][1])
                count+=1

def test_reader_vs_db_stat():
    num_gpus = 1
    for path in [caffe_db_folder, [caffe_db_folder], [caffe_db_folder, caffe_db_folder]]:
        for batch_size in {1, 32, 44}:
            for num_threads in {1, 2}:
                yield check_reader_vs_db_stat, path, batch_size, num_threads, num_gpus

# test 3: compare with a simple CaffeReader with batch_size=1
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


