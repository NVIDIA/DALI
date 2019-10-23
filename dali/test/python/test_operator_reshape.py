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
import math
from numpy.testing import assert_array_equal, assert_allclose
import os
import cv2
from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator

test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')


class ReshapePipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=3, device_id=0, num_gpus=1):
        super(ReshapePipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=True, exec_pipelined=True)
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.resize = ops.Resize(device = "cpu", resize_x = 224, resize_y = 224);
        self.reshape = ops.Reshape(device = device, shape = (224, 224 * 3), layout = "ab");

    def define_graph(self):
        jpegs, labels = self.input(name = "Reader")
        images = self.resize(self.decode(jpegs))
        if self.device == "gpu":
          images = images.gpu()
        reshaped = self.reshape(images)

        return [images, reshaped]

def CollapseChannels(image):
  new_shape = np.array([ image.shape[0], image.shape[1] * image.shape[2] ]).astype(np.int)
  return new_shape

class ReshapeWithInput(Pipeline):
    def __init__(self, device, batch_size, num_threads=3, device_id=0, num_gpus=1):
        super(ReshapeWithInput, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.gen_shapes = ops.PythonFunction(function=CollapseChannels)
        self.reshape = ops.Reshape(device = device, layout = "ab");

    def define_graph(self):
        jpegs, labels = self.input(name = "Reader")
        images_cpu = self.decode(jpegs)
        shapes = self.gen_shapes(images_cpu)
        images = images_cpu.gpu() if self.device == "gpu" else images_cpu
        reshaped = self.reshape(images, shapes)

        return [images, reshaped]

def verify_tensor_layouts(imgs, reshaped):
  assert imgs.layout() == "HWC"
  assert reshaped.layout() == "ab"
  for i in range(len(imgs)):
    assert imgs.at(i).layout() == "HWC"
    assert reshaped.at(i).layout() == "ab"

def verify(imgs, reshaped, src_shape = None):
  assert imgs.layout() == "HWC"
  assert reshaped.layout() == "ab"
  for i in range(len(imgs)):
    if src_shape is not None:
      assert imgs.at(i).shape == src_shape
    img_shape = imgs.at(i).shape
    # collapse width and channels
    ref_shape = (img_shape[0], img_shape[1] * img_shape[2])
    assert reshaped.at(i).shape == ref_shape
    assert_array_equal(imgs.at(i).flatten(), reshaped.at(i).flatten())


def check_reshape(device, batch_size):
  pipe = ReshapePipeline(device, batch_size)
  pipe.build()
  for iter in range(10):
    imgs, reshaped = pipe.run()
    if device == "gpu":
      verify_tensor_layouts(imgs, reshaped)
      imgs = imgs.as_cpu()
      reshaped = reshaped.as_cpu()
    verify(imgs, reshaped, (224, 224, 3))

def check_reshape_with_input(device, batch_size):
  pipe = ReshapeWithInput(device, batch_size)
  pipe.build()
  for iter in range(10):
    imgs, reshaped = pipe.run()
    if device == "gpu":
      verify_tensor_layouts(imgs, reshaped)
      imgs = imgs.as_cpu()
      reshaped = reshaped.as_cpu()
    verify(imgs, reshaped)

def test_reshape_arg():
  for device in ["cpu", "gpu"]:
    for batch_size in [16]:
      yield check_reshape, device, batch_size

def test_reshape_input():
  for device in ["cpu", "gpu"]:
    for batch_size in [16]:
      yield check_reshape_with_input, device, batch_size

def main():
  for test in test_reshape_arg():
    test[0](*test[1:])
  for test in test_reshape_input():
    test[0](*test[1:])

if __name__ == '__main__':
  main()
