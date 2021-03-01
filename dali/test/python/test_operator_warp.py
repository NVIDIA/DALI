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
import nvidia.dali.fn as fn
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
import random

test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')

def gen_transform(angle, zoom, dst_cx, dst_cy, src_cx, src_cy):
    t1 = np.array([[1, 0, -dst_cx], [0, 1, -dst_cy], [0, 0, 1]])
    cosa = math.cos(angle)/zoom
    sina = math.sin(angle)/zoom
    r = np.array([
        [cosa, -sina, 0],
        [sina, cosa, 0],
        [0, 0, 1]])
    t2 = np.array([[1, 0, src_cx], [0, 1, src_cy], [0, 0, 1]])
    return (np.matmul(t2, np.matmul(r, t1)))[0:2,0:3]

def gen_transforms(n, step):
    a = 0.0
    step = step * (math.pi/180)
    out = np.zeros([n, 2, 3])
    for i in range(n):
        out[i,:,:] = gen_transform(a, 2, 160, 120, 100, 100)
        a = a + step
    return out.astype(np.float32)

def ToCVMatrix(matrix):
  offset = np.matmul(matrix, np.array([[0.5], [0.5], [1]]))
  result = matrix.copy()
  result[0][2] = offset[0] - 0.5
  result[1][2] = offset[1] - 0.5
  return result

def CVWarp(output_type, input_type, warp_matrix = None, inv_map = False):
  def warp_fn(img, matrix):
    size = (320, 240)
    matrix = ToCVMatrix(matrix)
    if output_type == dali.types.FLOAT or input_type == dali.types.FLOAT:
      img = np.float32(img)
    out = cv2.warpAffine(img, matrix, size, borderMode = cv2.BORDER_CONSTANT, borderValue = [42,42,42],
      flags = (cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP) if inv_map else cv2.INTER_LINEAR);
    if output_type == dali.types.UINT8 and input_type == dali.types.FLOAT:
      out = np.uint8(np.clip(out, 0, 255))
    return out

  if warp_matrix:
    m = np.array(warp_matrix)
    def warp_fixed(img):
      return warp_fn(img, m)
    return warp_fixed

  return warp_fn


class WarpPipeline(Pipeline):
    def __init__(self, device, batch_size, output_type, input_type, use_input, num_threads=3, device_id=0, num_gpus=1, inv_map=False):
        super(WarpPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.use_input = use_input
        self.use_dynamic_size = use_input  # avoid Cartesian product
        self.name = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        if input_type != dali.types.UINT8:
          self.cast = ops.Cast(device = device, dtype = input_type)
        else:
          self.cast = None

        static_size = None if self.use_dynamic_size else (240,320)

        if use_input:
          self.transform_source = ops.ExternalSource(lambda: gen_transforms(self.batch_size, 10))
          self.warp = ops.WarpAffine(device = device, size=static_size, fill_value = 42, dtype = output_type, inverse_map=inv_map)
        else:
          warp_matrix = (0.1, 0.9, 10, 0.8, -0.2, -20)
          self.warp = ops.WarpAffine(device = device, size=static_size, matrix = warp_matrix, fill_value = 42, dtype = output_type, inverse_map=inv_map)

        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        if self.warp.device == "gpu":
          images = images.gpu()
        if self.cast:
          images = self.cast(images)

        dynamic_size = types.Constant(np.array([240, 320], dtype=np.float32)) if self.use_dynamic_size else None

        if self.use_input:
          transform = self.transform_source()
          outputs = self.warp(images, transform, size = dynamic_size)
        else:
          outputs = self.warp(images, size = dynamic_size)
        return outputs


class CVPipeline(Pipeline):
    def __init__(self, batch_size, output_type, input_type, use_input, num_threads=3, device_id=0, num_gpus=1, inv_map=False):
        super(CVPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.use_input = use_input
        self.name = "cv"
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        if self.use_input:
          self.transform_source = ops.ExternalSource(lambda: gen_transforms(self.batch_size, 10))
          self.warp = ops.PythonFunction(function=CVWarp(output_type, input_type, inv_map=inv_map),
                                         output_layouts="HWC")
        else:
          self.warp = ops.PythonFunction(function=CVWarp(output_type, input_type, [[0.1, 0.9, 10], [0.8, -0.2, -20]], inv_map),
                                         output_layouts="HWC")
        self.iter = 0

    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        if self.use_input:
          self.transform = self.transform_source()
          outputs = self.warp(images, self.transform)
        else:
          outputs = self.warp(images)
        return outputs

def compare(pipe1, pipe2, eps):
  epoch_size = pipe1.epoch_size("Reader")
  batch_size = pipe1.batch_size
  niter = (epoch_size + batch_size - 1) // batch_size
  compare_pipelines(pipe1, pipe2, batch_size, niter, eps);

io_types = [
  (dali.types.UINT8, dali.types.UINT8),
  (dali.types.UINT8, dali.types.FLOAT),
  (dali.types.FLOAT, dali.types.UINT8),
  (dali.types.FLOAT, dali.types.FLOAT)
]


def test_cpu_vs_cv():
  random.seed(1009)
  for batch_size in [1, 4, 19]:
    for use_input in [False, True]:
      for (itype, otype) in io_types:
        inv_map = random.choice([False, True])
        print("Testing cpu vs cv",
              "\nbatch size: ", batch_size,
              " matrix as input: ", use_input,
              " input_type: ", itype,
              " output_type: ", otype,
              " map_inverse:", inv_map)
        cv_pipeline = CVPipeline(batch_size, otype, itype, use_input, inv_map=inv_map);
        cv_pipeline.build();

        cpu_pipeline = WarpPipeline("cpu", batch_size, otype, itype, use_input, inv_map=inv_map);
        cpu_pipeline.build();

        compare(cv_pipeline, cpu_pipeline, 8)

def test_gpu_vs_cv():
  random.seed(1007)
  for batch_size in [1, 4, 19]:
    for use_input in [False, True]:
      for (itype, otype) in io_types:
        inv_map = random.choice([False, True])
        print("Testing gpu vs cv",
              "\nbatch size: ", batch_size,
              " matrix as input: ", use_input,
              " input_type: ", itype,
              " output_type: ", otype,
              " map_inverse:", inv_map)
        cv_pipeline = CVPipeline(batch_size, otype, itype, use_input, inv_map=inv_map);
        cv_pipeline.build();

        gpu_pipeline = WarpPipeline("gpu", batch_size, otype, itype, use_input, inv_map=inv_map);
        gpu_pipeline.build();

        compare(cv_pipeline, gpu_pipeline, 8)

def test_gpu_vs_cpu():
  random.seed(1005)
  for batch_size in [1, 4, 19]:
    for use_input in [False, True]:
      for (itype, otype) in io_types:
        inv_map = random.choice([False, True])
        print("Testing gpu vs cpu",
              "\nbatch size: ", batch_size,
              " matrix as input: ", use_input,
              " input_type: ", itype,
              " output_type: ", otype,
              " map_inverse:", inv_map)
        cpu_pipeline = WarpPipeline("cpu", batch_size, otype, itype, use_input, inv_map=inv_map);
        cpu_pipeline.build();

        gpu_pipeline = WarpPipeline("gpu", batch_size, otype, itype, use_input, inv_map=inv_map);
        gpu_pipeline.build();

        compare(cpu_pipeline, gpu_pipeline, 1)

def _test_extremely_large_data(device):
  in_size = 30000
  out_size = 10
  channels = 3

  def get_data():
    out = np.full([in_size, in_size, channels], 42, dtype=np.uint8)
    for c in range(channels):
      out[in_size-1, in_size-1, c] = c
    return [out]

  pipe = Pipeline(1, 3, 0, prefetch_queue_depth=1)
  input = fn.external_source(source=get_data, device=device)

  rotated = fn.warp_affine(input, matrix=[-1, 0, in_size, 0, -1, in_size],
                           fill_value=255.0, size=[out_size,out_size], interp_type=types.INTERP_NN)
  pipe.set_outputs(rotated)
  pipe.build()

  out = None
  try:
    out, = pipe.run()
  except RuntimeError as e:
    if "bad_alloc" in str(e):
      print("Skipping test due to out-of-memory error:", e)
      return
    raise
  except MemoryError as e:
    print("Skipping test due to out-of-memory error:", e)
    return
  if device == "cpu":
    out = out.at(0)
  else:
    out = out.as_cpu().at(0)
  assert out.shape == (out_size, out_size, channels)
  for c in range(channels):
    assert out[0,0,c] == c

def test_extremely_large_data():
  for device in ["cpu", "gpu"]:
    yield _test_extremely_large_data, device
