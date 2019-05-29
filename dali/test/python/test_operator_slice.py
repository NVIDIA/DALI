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
from nvidia.dali.edge import EdgeReference
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os

from test_utils import compare_pipelines

test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')

class SlicePipeline(Pipeline):
    def __init__(self, device, batch_size, pos_size_iter, num_threads=1, device_id=0, is_old_slice=True):
        super(SlicePipeline, self).__init__(batch_size,
                                            num_threads,
                                            device_id,
                                            seed=1234)
        self.pos_size_iter = pos_size_iter
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle=False)
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

class SliceArgsIteratorExtractFirstChannel(object):
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
            pos.append(np.asarray([0.0, 0.0, 0.0], dtype=np.float32)) # yxc
            size.append(np.asarray([1.0, 1.0, 1./3.], dtype=np.float32)) # HWC
            self.i = (self.i + 1) % self.n
        return (pos, size)
    next = __next__

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads=1, device_id=0):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id,
                                             exec_async=False,
                                             exec_pipelined=False,
                                             seed=1234)
        self.input = ops.CaffeReader(path = caffe_db_folder, random_shuffle=False)
        self.decode = ops.HostDecoder(device = 'cpu', output_type = types.RGB)

    def load(self):
        jpegs, _ = self.input()
        decoded = self.decode(jpegs)
        return decoded

class PythonOperatorPipeline(CommonPipeline):
    def __init__(self, function, batch_size, num_threads=1, device_id=0):
        super(PythonOperatorPipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id)
        self.python_function = ops.PythonFunction(function=function)

    def define_graph(self):
        images = self.load()
        processed = self.python_function(images)
        assert isinstance(processed, EdgeReference)
        return processed

def extract_first_channel(image):
    return image[:,:,0].reshape(image.shape[0:2] + (1,))

def test_slice_extract_channel_cpu():
    for batch_size in {1, 32, 64}:
        eii = SliceArgsIteratorExtractFirstChannel(batch_size)
        compare_pipelines(SlicePipeline('cpu', batch_size, iter(eii), is_old_slice=False),
                          PythonOperatorPipeline(extract_first_channel, batch_size),
                          batch_size=batch_size, N_iterations=10)

def test_slice_extract_channel_gpu():
    for batch_size in {1, 32, 64}:
        eii = SliceArgsIteratorExtractFirstChannel(batch_size)
        compare_pipelines(SlicePipeline('gpu', batch_size, iter(eii), is_old_slice=False),
                          PythonOperatorPipeline(extract_first_channel, batch_size),
                          batch_size=batch_size, N_iterations=10)

def slice_func(image):
    start_y = int(np.float32(image.shape[0]) * np.float32(0.2))
    end_y = int(np.float32(image.shape[0]) * np.float32(0.2 + 0.5))
    start_x = int(np.float32(image.shape[1]) * np.float32(0.4))
    end_x = int(np.float32(image.shape[1]) * np.float32(0.4 + 0.3))
    return image[start_y:end_y, start_x:end_x, :]

def test_slice_vs_numpy_slice_gpu():
    for batch_size in {1, 32, 64}:
        eii = SliceArgsIteratorAllDims(batch_size)
        compare_pipelines(SlicePipeline('gpu', batch_size, iter(eii), is_old_slice=False),
                          PythonOperatorPipeline(slice_func, batch_size),
                          batch_size=batch_size, N_iterations=10)

def test_slice_vs_numpy_slice_cpu():
    for batch_size in {1, 32, 64}:
        eii = SliceArgsIteratorAllDims(batch_size)
        compare_pipelines(SlicePipeline('cpu', batch_size, iter(eii), is_old_slice=False),
                          PythonOperatorPipeline(slice_func, batch_size),
                          batch_size=batch_size, N_iterations=10)
