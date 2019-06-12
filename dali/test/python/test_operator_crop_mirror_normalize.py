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

test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')

class CropMirrorNormalizePipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1,
                 is_new_cmn = False, output_dtype = types.FLOAT, output_layout = types.NHWC,
                 mirror_probability = 0.0, mean=[0., 0., 0.], std=[1., 1., 1.], pad_output=False):
        super(CropMirrorNormalizePipeline, self).__init__(batch_size, num_threads, device_id, seed=7865)
        self.device = device
        self.is_new_cmn = is_new_cmn
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        if self.is_new_cmn:
            self.cmn = ops.NewCropMirrorNormalize(device = self.device,
                                                  output_dtype = output_dtype,
                                                  output_layout = output_layout,
                                                  crop = (224, 224),
                                                  crop_pos_x = 0.3,
                                                  crop_pos_y = 0.2,
                                                  image_type = types.RGB,
                                                  mean = mean,
                                                  std = std,
                                                  pad_output = pad_output)
        else:
            self.cmn = ops.CropMirrorNormalize(device = self.device,
                                               output_dtype = output_dtype,
                                               output_layout = output_layout,
                                               crop = (224, 224),
                                               crop_pos_x = 0.3,
                                               crop_pos_y = 0.2,
                                               image_type = types.RGB,
                                               mean = mean,
                                               std = std,
                                               pad_output = pad_output)
        self.coin = ops.CoinFlip(probability=mirror_probability, seed=7865)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        rng = self.coin()
        images = self.cmn(images, mirror=rng)
        return images

def check_cmn_cpu_vs_gpu(batch_size, output_dtype, output_layout, mirror_probability, mean, std, pad_output, is_new_cmn):
    iterations = 8 if batch_size == 1 else 1
    compare_pipelines(CropMirrorNormalizePipeline('cpu', batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=pad_output,
                                                  is_new_cmn=is_new_cmn),
                      CropMirrorNormalizePipeline('gpu', batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=pad_output,
                                                  is_new_cmn=is_new_cmn),
                      batch_size=batch_size, N_iterations=iterations)

def test_cmn_cpu_vs_gpu():
    for batch_size in [1, 8]:
        for output_dtype in [types.FLOAT, types.INT32]:
            for output_layout in [types.NHWC, types.NCHW]:
                for mirror_probability in [0.0, 0.5, 1.0]:
                    for (mean, std) in [ ([0., 0., 0.], [1., 1., 1.]),
                                         ([0.5 * 255], [0.225 * 255]),
                                         ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]) ]:
                        for pad_output in [False, True]:
                            for is_new_cmn in [False, True]:
                                yield check_cmn_cpu_vs_gpu, batch_size, output_dtype, output_layout, mirror_probability, mean, std, pad_output, True

def check_cmn_cpu_old_vs_new(device_new, device_old, batch_size, output_dtype, output_layout, mirror_probability, mean, std, pad_output):
    iterations = 8 if batch_size == 1 else 1
    compare_pipelines(CropMirrorNormalizePipeline(device_old, batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=pad_output,
                                                  is_new_cmn=False),
                      CropMirrorNormalizePipeline(device_new, batch_size, output_dtype=output_dtype,
                                                  output_layout=output_layout, mirror_probability=mirror_probability,
                                                  mean=mean, std=std, pad_output=pad_output,
                                                  is_new_cmn=True),
                      batch_size=batch_size, N_iterations=iterations)

def test_cmn_cpu_old_vs_new():
    for device_new in ['cpu', 'gpu']:
        for device_old in ['cpu', 'gpu']:
            for batch_size in [1, 8]:
                for output_dtype in [types.FLOAT, types.INT32]:
                    for output_layout in [types.NHWC, types.NCHW]:
                        for mirror_probability in [0.0, 0.5, 1.0]:
                            norm_data = [ ([0., 0., 0.], [1., 1., 1.]),
                                          ([0.5 * 255], [0.225 * 255]),
                                          ([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]) ] \
                                        if output_dtype != types.INT32 else \
                                        [ ([0., 0., 0.], [1., 1., 1.]),
                                          ([9, 8, 10], [10, 8, 9]),
                                          ([10, 10, 10], [10, 10, 10]) ]
                            for (mean, std) in norm_data:
                                for pad_output in [False, True] if device_old != 'cpu' else [False]: # padding doesn't work in the old CPU version
                                    yield check_cmn_cpu_old_vs_new, device_new, device_old, batch_size, output_dtype, \
                                        output_layout, mirror_probability, mean, std, pad_output


class NoCropPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, decoder_only=False):
        super(NoCropPipeline, self).__init__(batch_size, num_threads, device_id)
        self.decoder_only = decoder_only
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        if not self.decoder_only:
            self.cmn = ops.NewCropMirrorNormalize(device = self.device,
                                                  image_type = types.RGB,
                                                  output_dtype = types.UINT8,
                                                  output_layout = types.NHWC)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if not self.decoder_only:
            images = self.decode(inputs)
            if self.device == 'gpu':
                images = images.gpu()
            images = self.cmn(images)
        return images

def check_cmn_no_crop_args_vs_decoder_only(device, batch_size):
    compare_pipelines(NoCropPipeline(device, batch_size, decoder_only=True),
                      NoCropPipeline(device, batch_size, decoder_only=False),
                      batch_size=batch_size, N_iterations=10)

def test_cmn_no_crop_args_vs_decoder_only():
    for device in {'cpu'}:
        for batch_size in {1, 4}:
            yield check_cmn_no_crop_args_vs_decoder_only, device, batch_size
