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
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
import cv2

test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')

def crop_func_help(image, layout, crop_y = 0.2, crop_x = 0.3, crop_h = 220, crop_w = 224):
    if layout == types.NFHWC:
        assert len(image.shape) == 4
        H = image.shape[1]
        W = image.shape[2]
    elif layout == types.NHWC:
        assert len(image.shape) == 3
        H = image.shape[0]
        W = image.shape[1]

    assert H >= crop_h
    assert W >= crop_w

    start_y = int(np.round(np.float32(crop_y) * np.float32(H - crop_h)))
    end_y = start_y + crop_h
    start_x = int(np.round(np.float32(crop_x) * np.float32(W - crop_w)))
    end_x = start_x + crop_w

    if layout == types.NFHWC:
        return image[:, start_y:end_y, start_x:end_x, :]
    elif layout == types.NHWC:
        return image[start_y:end_y, start_x:end_x, :]
    else:
        assert(False)  # should not happen

def crop_NHWC_func(image):
    return crop_func_help(image, types.NHWC)

def resize_func_help(image, size_x = 300, size_y = 900):
    res = cv2.resize(image, (size_x, size_y))
    return res

def resize_func(image):
    return resize_func_help(image)

class MultichannelPipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator, num_threads=1, device_id=0, should_resize=False, should_crop=False):
        super(MultichannelPipeline, self).__init__(batch_size,
                                                   num_threads,
                                                   device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.should_crop = should_crop
        self.should_resize = should_resize
        if self.should_resize:
            self.resize = ops.Resize(device = self.device,
                                     resize_y = 900,
                                     resize_x = 300,
                                     min_filter=types.DALIInterpType.INTERP_LINEAR)
        if self.should_crop:
            self.crop = ops.Crop(device = self.device,
                                 crop = (220, 224),
                                 crop_pos_x = 0.3,
                                 crop_pos_y = 0.2,
                                 image_type = types.RGB)

    def define_graph(self):
        self.data = self.inputs()
        sequence = self.data.gpu() if self.device == 'gpu' else self.data
        out = sequence
        if self.should_resize:
            out = self.resize(out)
        if self.should_crop:
            out = self.crop(out)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

class MultichannelPythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(MultichannelPythonOpPipeline, self).__init__(batch_size,
                                                           num_threads,
                                                           device_id,
                                                           exec_async=False,
                                                           exec_pipelined=False)
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.crop = ops.PythonFunction(function=function)

    def define_graph(self):
        self.data = self.inputs()
        out = self.crop(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

def check_resize_multichannel_vs_numpy(device, batch_size, shape):
    eii1 = RandomDataIterator(batch_size, shape=shape)
    eii2 = RandomDataIterator(batch_size, shape=shape)
    compare_pipelines(MultichannelPipeline(device, batch_size, types.NHWC, iter(eii1), should_resize=True),
                      MultichannelPythonOpPipeline(resize_func, batch_size, types.NHWC, iter(eii2)),
                      batch_size=batch_size, N_iterations=10,
                      eps = 0.2)

def test_resize_multichannel_vs_numpy():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 3}:
            for shape in {(2048, 512, 3), (2048, 512, 8)}:
                yield check_resize_multichannel_vs_numpy, device, batch_size, shape


def check_crop_multichannel_vs_numpy(device, batch_size, shape):
    eii1 = RandomDataIterator(batch_size, shape=shape)
    eii2 = RandomDataIterator(batch_size, shape=shape)
    compare_pipelines(MultichannelPipeline(device, batch_size, types.NHWC, iter(eii1), should_crop=True),
                      MultichannelPythonOpPipeline(crop_NHWC_func, batch_size, types.NHWC, iter(eii2)),
                      batch_size=batch_size, N_iterations=10)

def test_crop_multichannel_vs_numpy():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 3}:
            for shape in {(2048, 512, 3), (2048, 512, 8)}:
                yield check_crop_multichannel_vs_numpy, device, batch_size, shape
