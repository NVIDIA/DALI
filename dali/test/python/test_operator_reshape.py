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

def verify_tensor_layouts(imgs, reshaped):
  assert imgs.layout() == "HWC"
  assert reshaped.layout() == "ab"
  for i in range(len(imgs)):
    assert imgs.at(i).layout() == "HWC"
    assert reshaped.at(i).layout() == "ab"

def verify(imgs, reshaped):
  assert imgs.layout() == "HWC"
  assert reshaped.layout() == "ab"
  for i in range(len(imgs)):
    assert imgs.at(i).shape == (224, 224, 3)
    assert reshaped.at(i).shape == (224, 224 * 3)
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
    verify(imgs, reshaped)

def test_reshape():
  for device in ["cpu", "gpu"]:
    for batch_size in [16]:
      yield check_reshape, device, batch_size

def main():
  for test in test_reshape():
    test[0](*test[1:])

if __name__ == '__main__':
  main()
