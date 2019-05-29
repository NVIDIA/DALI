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

from test_utils import check_batch
from test_utils import compare_pipelines

test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')

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
