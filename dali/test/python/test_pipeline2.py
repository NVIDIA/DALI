# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

import glob
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import nvidia.dali as dali
from timeit import default_timer as timer
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os

caffe_db_folder = "/data/imagenet/train-lmdb-256x256"

test_data_root = os.environ['DALI_EXTRA_PATH']
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')

def check_batch(batch1, batch2, batch_size, eps = 0.0000001):
    for i in range(batch_size):
        is_failed = False
        try:
            err = np.mean( np.abs(batch1.at(i) - batch2.at(i)) )
        except:
            is_failed = True
        if (is_failed or err > eps ):
            try:
                print("failed[{}] err[{}]".format(is_failed, err))
                plt.imsave("err_1.png", batch1.at(i))
                plt.imsave("err_2.png", batch2.at(i))
            except:
                print("Batch at {} can't be saved as an image".format(i))
                print(batch1.at(i))
                print(batch2.at(i))
            assert(False)

def compare_pipelines(pipe1, pipe2, batch_size, N_iterations):
    pipe1.build()
    pipe2.build()
    for k in range(N_iterations):
        out1 = pipe1.run()
        out2 = pipe2.run()
        assert len(out1) == len(out2)
        for i in range(len(out1)):
            out1_data = out1[i].as_cpu() if isinstance(out1[i].at(0), dali.backend_impl.TensorGPU) else out1[i]
            out2_data = out2[i].as_cpu() if isinstance(out2[i].at(0), dali.backend_impl.TensorGPU) else out2[i]
            check_batch(out1_data, out2_data, batch_size)
    print("OK: ({} iterations)".format(N_iterations))

class CropPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, is_old_crop=True):
        super(CropPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)

        if is_old_crop:
            self.crop = ops.Crop(device = device,
                                 crop = (224, 224),
                                 crop_pos_x = 0.3,
                                 crop_pos_y = 0.2,
                                 image_type = types.RGB)
        else:
            self.crop = ops.NewCrop(device = device,
                                    crop = (224, 224),
                                    crop_pos_x = 0.3,
                                    crop_pos_y = 0.2,
                                    image_type = types.RGB)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        out = self.crop(images.gpu()) if self.device == 'gpu' else self.crop(images)
        return out

def test_old_crop_vs_new_crop_cpu():
    for batch_size in {1, 32, 100}:
        compare_pipelines(CropPipeline('cpu', batch_size, is_old_crop=True),
                          CropPipeline('cpu', batch_size, is_old_crop=False),
                          batch_size=batch_size, N_iterations=20)

def test_old_crop_vs_new_crop_gpu():
    for batch_size in {1, 32, 100}:
        compare_pipelines(CropPipeline('gpu', batch_size, is_old_crop=True),
                          CropPipeline('gpu', batch_size, is_old_crop=False),
                          batch_size=batch_size, N_iterations=20)


class CropSequencePipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, is_old_crop=True):
        super(CropSequencePipeline, self).__init__(batch_size,
                                                   num_threads,
                                                   device_id)
        self.device = device
        VIDEO_FILES = [test_data_video + '/' + file for file in ['sintel_trailer_short.mp4']]
        self.input = ops.VideoReader(device='gpu', filenames=VIDEO_FILES, sequence_length=10,
                                     shard_id=0, num_shards=1, random_shuffle=False,
                                     normalized=True, image_type=types.RGB, dtype=types.UINT8)
        if is_old_crop:
            self.crop = ops.Crop(device = device,
                                 crop = (224, 224),
                                 crop_pos_x = 0.3,
                                 crop_pos_y = 0.2,
                                 image_type = types.RGB)
        else:
            self.crop = ops.NewCrop(device = device,
                                    crop = (224, 224),
                                    crop_pos_x = 0.3,
                                    crop_pos_y = 0.2,
                                    image_type = types.RGB)

    def define_graph(self):
        input_data = self.input(name='Reader')
        out = self.crop(input_data.gpu()) if self.device == 'gpu' else self.crop(input_data)
        return out

def test_crop_sequence_old_crop_vs_new_crop_gpu():
    batch_size = 4
    compare_pipelines(CropSequencePipeline('gpu', batch_size, is_old_crop=True),
                      CropSequencePipeline('gpu', batch_size, is_old_crop=False),
                      batch_size=batch_size, N_iterations=10)

class SlicePipeline(Pipeline):
    def __init__(self, device, batch_size, pos_size_iter, num_threads=1, device_id=0, num_gpus=1, is_old_slice=True):
        super(SlicePipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.pos_size_iter = pos_size_iter
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()
        self.input_crop = ops.ExternalSource()
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)

        if is_old_slice:
            self.slice = ops.Slice(device = device,
                                   image_type = types.RGB)
        else:
            self.slice = ops.NewSlice(device = device,
                                      image_type = types.RGB)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()
        if self.device == 'gpu':
            images = images.gpu()
        out = self.slice(images, self.crop_pos, self.crop_size)
        return out

    def iter_setup(self):
        (crop_pos, crop_size) = self.pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)


class SliceArgsIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        pos = []
        size = []
        for k in range(self.batch_size):
            pos.append(np.asarray([0.4, 0.2], dtype=np.float32)) # xy
            size.append(np.asarray([0.3, 0.5], dtype=np.float32)) # WH
            self.i = (self.i + 1) % self.n
        return (pos, size)
    next = __next__

def test_old_slice_vs_new_slice_gpu():
    for batch_size in {1, 13, 64}:
        eii1 = SliceArgsIterator(batch_size)
        pos_size_iter1 = iter(eii1)

        eii2 = SliceArgsIterator(batch_size)
        pos_size_iter2 = iter(eii2)

        compare_pipelines(SlicePipeline('gpu', batch_size, pos_size_iter1, is_old_slice=True),
                          SlicePipeline('gpu', batch_size, pos_size_iter2, is_old_slice=False),
                          batch_size=batch_size, N_iterations=10)

class SliceArgsIteratorAllDims(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        pos = []
        size = []
        for k in range(self.batch_size):
            pos.append(np.asarray([0.2, 0.4, 0.0], dtype=np.float32)) # yxc
            size.append(np.asarray([0.5, 0.3, 1.0], dtype=np.float32)) # HWC
            self.i = (self.i + 1) % self.n
        return (pos, size)
    next = __next__

def test_new_slice_gpu_args_WH_vs_args_HWC():
    for batch_size in {3, 32, 64}:
        eii1 = SliceArgsIterator(batch_size)
        pos_size_iter1 = iter(eii1)

        eii2 = SliceArgsIteratorAllDims(batch_size)
        pos_size_iter2 = iter(eii2)

        compare_pipelines(SlicePipeline('gpu', batch_size, pos_size_iter1, is_old_slice=False),
                          SlicePipeline('gpu', batch_size, pos_size_iter2, is_old_slice=False),
                          batch_size=batch_size, N_iterations=10)

def test_new_slice_cpu_vs_gpu():
    for batch_size in {3, 32, 64}:
        eii1 = SliceArgsIterator(batch_size)
        pos_size_iter1 = iter(eii1)

        eii2 = SliceArgsIterator(batch_size)
        pos_size_iter2 = iter(eii2)

        compare_pipelines(SlicePipeline('gpu', batch_size, pos_size_iter1, is_old_slice=False),
                          SlicePipeline('cpu', batch_size, pos_size_iter2, is_old_slice=False),
                          batch_size=batch_size, N_iterations=10)

test_new_slice_cpu_vs_gpu()
