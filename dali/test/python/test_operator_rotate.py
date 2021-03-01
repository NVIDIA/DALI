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

def get_output_size(angle, input_size):
    cosa = abs(math.cos(angle))
    sina = abs(math.sin(angle))
    (h, w) = input_size[0:2]
    eps = 1e-2
    out_w = int(math.ceil(w*cosa + h*sina - eps))
    out_h = int(math.ceil(h*cosa + w*sina - eps))
    if sina <= cosa:
      if out_w % 2 != w % 2:
        out_w += 1
      if out_h % 2 != h % 2:
        out_h += 1
    else:
      if out_w % 2 != h % 2:
        out_w += 1
      if out_h % 2 != w % 2:
        out_h += 1
    return (out_h, out_w)

def get_transform(angle, input_size, output_size):
    cosa = math.cos(angle)
    sina = math.sin(angle)
    (out_h, out_w) = output_size[0:2]
    (in_h,  in_w) = input_size[0:2]
    t1 = np.array([
        [1, 0, -out_w*0.5],
        [0, 1, -out_h*0.5],
        [0, 0, 1]])
    r = np.array([
        [cosa, -sina, 0],
        [sina, cosa, 0],
        [0, 0, 1]])
    t2 = np.array([
        [1, 0, in_w*0.5],
        [0, 1, in_h*0.5],
        [0, 0, 1]])

    return (np.matmul(t2, np.matmul(r, t1)))[0:2,0:3]

def ToCVMatrix(matrix):
  offset = np.matmul(matrix, np.array([[0.5], [0.5], [1]]))
  result = matrix.copy()
  result[0][2] = offset[0] - 0.5
  result[1][2] = offset[1] - 0.5
  return result

def CVRotate(output_type, input_type, fixed_size):
  def warp_fn(img, angle):
    in_size = img.shape[0:2]
    angle = math.radians(angle)
    out_size = fixed_size if fixed_size is not None else get_output_size(angle, in_size)
    matrix = get_transform(angle, in_size, out_size)
    matrix = ToCVMatrix(matrix)
    if output_type == dali.types.FLOAT or input_type == dali.types.FLOAT:
      img = np.float32(img)
    out_size_wh = (out_size[1], out_size[0])
    out = cv2.warpAffine(img, matrix, out_size_wh, borderMode = cv2.BORDER_CONSTANT, borderValue = [42,42,42],
      flags = (cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP));
    if output_type == dali.types.UINT8 and input_type == dali.types.FLOAT:
      out = np.uint8(np.clip(out, 0, 255))
    return out

  return warp_fn

class RotatePipeline(Pipeline):
    def __init__(self, device, batch_size, output_type, input_type, fixed_size=None, num_threads=3, device_id=0, num_gpus=1):
        super(RotatePipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.name = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        if input_type != dali.types.UINT8:
          self.cast = ops.Cast(device = device, dtype = input_type)
        else:
          self.cast = None

        self.uniform = ops.random.Uniform(range = (-180.0, 180.0), seed = 42);
        self.rotate = ops.Rotate(device = device, size=fixed_size, fill_value = 42, dtype = output_type)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        if self.rotate.device == "gpu":
          images = images.gpu()
        if self.cast:
          images = self.cast(images)

        outputs = self.rotate(images, angle = self.uniform())
        return outputs

class CVPipeline(Pipeline):
    def __init__(self, batch_size, output_type, input_type, fixed_size, num_threads=3, device_id=0, num_gpus=1):
        super(CVPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.name = "cv"
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        self.rotate = ops.PythonFunction(function=CVRotate(output_type, input_type, fixed_size),
                                         output_layouts="HWC")
        self.uniform = ops.random.Uniform(range = (-180.0, 180.0), seed = 42)
        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        angles = self.uniform()
        outputs = self.rotate(images, angles)
        return outputs


def compare(pipe1, pipe2, eps):
  pipe1.build()
  pipe2.build()
  epoch_size = pipe1.epoch_size("Reader")
  batch_size = pipe1.batch_size
  niter = 1 if batch_size >= epoch_size else 2
  compare_pipelines(pipe1, pipe2, batch_size, niter, eps);

io_types = [
  (dali.types.UINT8, dali.types.UINT8),
  (dali.types.UINT8, dali.types.FLOAT),
  (dali.types.FLOAT, dali.types.UINT8),
  (dali.types.FLOAT, dali.types.FLOAT)
]

def create_pipeline(backend, *args):
  if backend == "cv":
    return CVPipeline(*args)
  else:
    return RotatePipeline(backend, *args)

def run_cases(backend1, backend2, epsilon):
  for batch_size in [1, 4, 19]:
    for output_size in [None, (160,240)]:
      for (itype, otype) in io_types:
        def run_case(backend1, backend2, *args):
          pipe1 = create_pipeline(backend1, *args)
          pipe2 = create_pipeline(backend2, *args)
          compare(pipe1, pipe2, epsilon)
        yield run_case, backend1, backend2, batch_size, otype, itype, output_size

def test_gpu_vs_cv():
  for test in run_cases("gpu", "cv", 8):
    yield test

def test_cpu_vs_cv():
  for test in run_cases("cpu", "cv", 8):
    yield test

def test_gpu_vs_cpu():
  for test in run_cases("gpu", "cpu", 1):
    yield test

