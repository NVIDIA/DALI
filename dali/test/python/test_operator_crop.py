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
from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')

class CropPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, is_fused_decoder=False):
        super(CropPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.is_fused_decoder = is_fused_decoder
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)

        if self.is_fused_decoder:
            self.decode = ops.ImageDecoderCrop(device = "cpu",
                                              crop = (224, 224),
                                              crop_pos_x = 0.3,
                                              crop_pos_y = 0.2,
                                              output_type = types.RGB)
        else:
            self.decode = ops.ImageDecoder(device = "cpu",
                                          output_type = types.RGB)
            self.crop = ops.Crop(device = self.device,
                                 crop = (224, 224),
                                 crop_pos_x = 0.3,
                                 crop_pos_y = 0.2,
                                 image_type = types.RGB)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        if self.is_fused_decoder:
            images = self.decode(inputs)
        else:
            images = self.decode(inputs)
            if self.device == 'gpu':
                images = images.gpu()
            images = self.crop(images)
        return images

def check_crop_vs_fused_decoder(device, batch_size):
    compare_pipelines(CropPipeline(device, batch_size, is_fused_decoder=True),
                      CropPipeline(device, batch_size, is_fused_decoder=False),
                      batch_size=batch_size, N_iterations=10)

def test_crop_vs_fused_decoder():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 32, 100}:
            yield check_crop_vs_fused_decoder, device, batch_size

def check_crop_cpu_vs_gpu(batch_size):
    compare_pipelines(CropPipeline('cpu', batch_size),
                      CropPipeline('gpu', batch_size),
                      batch_size=batch_size, N_iterations=10)

def test_crop_cpu_vs_gpu():
    for batch_size in {1, 32, 100}:
        yield check_crop_cpu_vs_gpu, batch_size

class CropSequencePipeline(Pipeline):
    def __init__(self, device, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(CropSequencePipeline, self).__init__(batch_size,
                                                   num_threads,
                                                   device_id)
        self.device = device
        self.layout = layout
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.crop = ops.Crop(device = self.device,
                             crop = (224, 224),
                             crop_pos_x = 0.3,
                             crop_pos_y = 0.2,
                             image_type = types.RGB)

    def define_graph(self):
        self.data = self.inputs()
        sequence = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.crop(sequence)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.layout)

class CropSequencePythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, layout, iterator, num_threads=1, device_id=0):
        super(CropSequencePythonOpPipeline, self).__init__(batch_size,
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

def crop_func_help(image, layout, crop_y = 0.2, crop_x = 0.3, crop_h = 224, crop_w = 224):
    if layout == "FHWC":
        assert len(image.shape) == 4
        H = image.shape[1]
        W = image.shape[2]
    elif layout == "HWC":
        assert len(image.shape) == 3
        H = image.shape[0]
        W = image.shape[1]

    assert H >= crop_h
    assert W >= crop_w

    start_y = int(np.float32(crop_y) * np.float32(H - crop_h) + np.float32(0.5))
    end_y = start_y + crop_h
    start_x = int(np.float32(crop_x) * np.float32(W - crop_w) + np.float32(0.5))
    end_x = start_x + crop_w

    if layout == "FHWC":
        return image[:, start_y:end_y, start_x:end_x, :]
    elif layout == "HWC":
        return image[start_y:end_y, start_x:end_x, :]
    else:
        assert(False)  # should not happen

def crop_NFHWC_func(image):
    return crop_func_help(image, "FHWC")

def crop_NHWC_func(image):
    return crop_func_help(image, "HWC")

def check_crop_NFHWC_vs_python_op_crop(device, batch_size):
    eii1 = RandomDataIterator(batch_size, shape=(10, 300, 400, 3))
    eii2 = RandomDataIterator(batch_size, shape=(10, 300, 400, 3))
    compare_pipelines(CropSequencePipeline(device, batch_size, "FHWC", iter(eii1)),
                      CropSequencePythonOpPipeline(crop_NFHWC_func, batch_size, "FHWC", iter(eii2)),
                      batch_size=batch_size, N_iterations=10)

def test_crop_NFHWC_vs_python_op_crop():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 4}:
            yield check_crop_NFHWC_vs_python_op_crop, device, batch_size

def check_crop_NHWC_vs_python_op_crop(device, batch_size):
    eii1 = RandomDataIterator(batch_size, shape=(300, 400, 3))
    eii2 = RandomDataIterator(batch_size, shape=(300, 400, 3))
    compare_pipelines(CropSequencePipeline(device, batch_size, "HWC", iter(eii1)),
                      CropSequencePythonOpPipeline(crop_NHWC_func, batch_size, "HWC", iter(eii2)),
                      batch_size=batch_size, N_iterations=10)

def test_crop_NHWC_vs_python_op_crop():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 4}:
            yield check_crop_NHWC_vs_python_op_crop, device, batch_size

class CropCastPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, should_perform_cast=False):
        super(CropCastPipeline, self).__init__(batch_size,
                                               num_threads,
                                               device_id)
        self.should_perform_cast = should_perform_cast
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "cpu",
                                      output_type = types.RGB)

        if self.should_perform_cast:
            self.crop = ops.Crop(device = self.device,
                                crop = (224, 224),
                                crop_pos_x = 0.3,
                                crop_pos_y = 0.2,
                                image_type = types.RGB,
                                output_dtype = types.FLOAT)
            self.crop2 = ops.Crop(device = self.device,
                                  crop = (224, 224),
                                  crop_pos_x = 0.0,
                                  crop_pos_y = 0.0,
                                  image_type = types.RGB,
                                  output_dtype = types.UINT8)
        else:
            self.crop = ops.Crop(device = self.device,
                    crop = (224, 224),
                    crop_pos_x = 0.3,
                    crop_pos_y = 0.2,
                    image_type = types.RGB)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        if self.should_perform_cast:
            images_float = self.crop(images)
            images = self.crop2(images_float)
        else:
            images = self.crop(images)
        return images

def check_crop_no_cast_vs_cast_to_float_and_back(device, batch_size):
    compare_pipelines(CropCastPipeline(device, batch_size, should_perform_cast=False),
                      CropCastPipeline(device, batch_size, should_perform_cast=True),
                      batch_size=batch_size, N_iterations=10)

def test_crop_no_cast_vs_cast_to_float_and_back():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 4}:
            yield check_crop_no_cast_vs_cast_to_float_and_back, device, batch_size

class Crop3dPipeline(Pipeline):
    def __init__(self, device, batch_size, iterator, data_shape, data_layout, num_threads=1, device_id=0):
        super(Crop3dPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_shape = data_shape
        self.data_layout = data_layout

        if self.data_layout == "DHWC":
            D, H, W = self.data_shape[0], self.data_shape[1], self.data_shape[2]
        elif self.data_layout == "CDHW":
            D, H, W = self.data_shape[1], self.data_shape[2], self.data_shape[3]
        else:
            assert(False)

        self.crop = ops.Crop(device = self.device,
                             crop_pos_z = 0.1,
                             crop_pos_y = 0.2,
                             crop_pos_x = 0.3,
                             crop_d = D * 0.91,
                             crop_h = H * 0.85,
                             crop_w = W * 0.75,
                             image_type = types.RGB)

    def define_graph(self):
        self.data = self.inputs()
        sequence = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.crop(sequence)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)

class Crop3dPythonOpPipeline(Pipeline):
    def __init__(self, function, batch_size, iterator, data_shape, data_layout, num_threads=1, device_id=0):
        super(Crop3dPythonOpPipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     exec_async=False,
                                                     exec_pipelined=False)
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_shape = data_shape
        self.data_layout = data_layout

        def crop_func(image):
            return function(image, layout=self.data_layout, shape=self.data_shape)

        self.crop = ops.PythonFunction(function=crop_func)

    def define_graph(self):
        self.data = self.inputs()
        out = self.crop(self.data)
        return out

    def iter_setup(self):
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)

def crop_3d_func(image, layout, shape, crop_anchor=(0.1, 0.2, 0.3), crop_shape=(0.91, 0.85, 0.75)):
    assert len(image.shape) == 4
    assert len(crop_anchor) == 3
    assert len(crop_shape) == 3

    if layout == "DHWC":
        D, H, W = image.shape[0], image.shape[1], image.shape[2]
    elif layout == "CDHW":
        D, H, W = image.shape[1], image.shape[2], image.shape[3]
    else:
        assert(False)

    crop_d, crop_h, crop_w = int(crop_shape[0]*D), int(crop_shape[1]*H), int(crop_shape[2]*W),
    assert D >= crop_d
    assert H >= crop_h
    assert W >= crop_w
    crop_z, crop_y, crop_x = crop_anchor[0], crop_anchor[1], crop_anchor[2]

    start_z = int(np.float32(0.5) + np.float32(crop_z) * np.float32(D - crop_d))
    end_z = start_z + crop_d
    start_y = int(np.float32(0.5) + np.float32(crop_y) * np.float32(H - crop_h))
    end_y = start_y + crop_h
    start_x = int(np.float32(0.5) + np.float32(crop_x) * np.float32(W - crop_w))
    end_x = start_x + crop_w

    if layout == "DHWC":
        return image[start_z:end_z, start_y:end_y, start_x:end_x, :]
    elif layout == "CDHW":
        return image[:, start_z:end_z, start_y:end_y, start_x:end_x]
    else:
        assert(False)

def check_crop_3d_vs_python_op_crop(device, batch_size, layout, shape):
    eii1 = RandomDataIterator(batch_size, shape=shape)
    eii2 = RandomDataIterator(batch_size, shape=shape)
    compare_pipelines(Crop3dPipeline(device, batch_size, iter(eii1), data_shape=shape, data_layout=layout),
                      Crop3dPythonOpPipeline(crop_3d_func, batch_size, iter(eii2), data_shape=shape, data_layout=layout),
                      batch_size=batch_size, N_iterations=3)

def test_crop_3d_vs_python_op_crop():
    for device in {'cpu', 'gpu'}:
        for batch_size in {1, 4}:
            for layout, shape in {("DHWC", (300, 100, 10, 3)),
                                  ("DHWC", (100, 300, 10, 1)),
                                  ("DHWC", (10, 30, 300, 1)),
                                  ("DHWC", (20, 50, 60, 8)),
                                  ("CDHW", (3, 300, 100, 10)),
                                  ("CDHW", (3, 300, 10, 100)),
                                  ("CDHW", (8, 30, 10, 50))}:
               yield check_crop_3d_vs_python_op_crop, device, batch_size, layout, shape
