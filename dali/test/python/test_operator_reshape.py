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
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
import math
from numpy.testing import assert_array_equal, assert_allclose
from functools import partial
import os
import cv2
from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from nose.tools import assert_raises

test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')


class ReshapePipeline(Pipeline):
    def __init__(self, device, batch_size, relative, use_wildcard, num_threads=3, device_id=0, num_gpus=1):
        super(ReshapePipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=True, exec_pipelined=True)
        self.device = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        W = 320
        H = 224
        self.resize = ops.Resize(device = "cpu", resize_x = W, resize_y = H);
        WC = -1 if use_wildcard else W * 3
        if relative:
            rel_shape = (-1, 3) if use_wildcard else (1, 3)
            self.reshape = ops.Reshape(device = device, rel_shape = rel_shape, layout = "ab");
        else:
            self.reshape = ops.Reshape(device = device, shape = (H, WC), layout = "ab");

    def define_graph(self):
        jpegs, labels = self.input(name = "Reader")
        images = self.resize(self.decode(jpegs))
        if self.device == "gpu":
          images = images.gpu()
        reshaped = self.reshape(images)

        # `images+0` creates a (no-op) arithmetic expression node - this prevents the
        # original `images` node from being marked as pipeline output
        return [images+0, reshaped]

def CollapseChannels(image):
  new_shape = np.array([ image.shape[0], image.shape[1] * image.shape[2] ]).astype(np.int)
  return new_shape

def CollapseChannelsWildcard(image):
  new_shape = np.array([ image.shape[0], -1 ]).astype(np.int)
  return new_shape

class ReshapeWithInput(Pipeline):
    def __init__(self, device, batch_size, use_wildcard, num_threads=3, device_id=0, num_gpus=1):
        super(ReshapeWithInput, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.device = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        fn = CollapseChannelsWildcard if use_wildcard else CollapseChannels
        self.gen_shapes = ops.PythonFunction(function=fn)
        self.reshape = ops.Reshape(device = device, layout = "ab");

    def define_graph(self):
        jpegs, labels = self.input(name = "Reader")
        images_cpu = self.decode(jpegs)
        shapes = self.gen_shapes(images_cpu)
        images = images_cpu.gpu() if self.device == "gpu" else images_cpu
        reshaped = self.reshape(images, shapes)

        return [images, reshaped]

def MakeTallFunc(relative, wildcard):
  def func(image):
    if relative:
        return np.array([ -1 if wildcard else 2, 0.5, 1]).astype(np.float32)
    else:
        h, w, c = image.shape
        return np.array([ -1 if wildcard else 2*h, w/2, c]).astype(np.int)
  return func

class ReshapeWithArgInput(Pipeline):
    def __init__(self, device, batch_size, relative, use_wildcard, num_threads=3, device_id=0, num_gpus=1):
        super(ReshapeWithArgInput, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.device = device
        self.input = ops.readers.Caffe(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.resize = ops.Resize(device = "cpu");
        self.decode = ops.decoders.Image(device = "cpu", output_type = types.RGB)
        self.gen_shapes = ops.PythonFunction(function=MakeTallFunc(relative, use_wildcard))
        self.reshape = ops.Reshape(device = device);
        self.relative = relative

    def define_graph(self):
        jpegs, labels = self.input(name = "Reader")
        images_cpu = self.decode(jpegs)

        rng = ops.random.Uniform(range=[100,128])
        cast = ops.Cast(dtype=types.INT32)
        widths = cast(rng()) * 2.0
        heights = cast(rng()) * 2.0
        images_cpu = self.resize(images_cpu, resize_x = widths, resize_y = heights)

        shapes = self.gen_shapes(images_cpu)
        images = images_cpu.gpu() if self.device == "gpu" else images_cpu
        if self.relative:
            reshaped = self.reshape(images, rel_shape = shapes)
        else:
            reshaped = self.reshape(images, shape = shapes)

        return [images, reshaped]

def verify_tensor_layouts(imgs, reshaped):
  assert imgs.layout() == "HWC"
  assert reshaped.layout() == "ab"
  for i in range(len(imgs)):
    assert imgs[i].layout() == "HWC"
    assert reshaped[i].layout() == "ab"

def verify_flatten(imgs, reshaped, src_shape = None):
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

def verify_make_tall(imgs, reshaped, src_shape = None):
  assert imgs.layout() == "HWC"
  assert reshaped.layout() == "HWC"
  for i in range(len(imgs)):
    if src_shape is not None:
      assert imgs.at(i).shape == src_shape
    img_shape = imgs.at(i).shape
    # collapse width and channels
    ref_shape = (img_shape[0] * 2, img_shape[1] // 2, 3)
    assert reshaped.at(i).shape == ref_shape
    assert_array_equal(imgs.at(i).flatten(), reshaped.at(i).flatten())


def check_reshape(device, batch_size, relative, use_wildcard):
  pipe = ReshapePipeline(device, batch_size, relative, use_wildcard)
  pipe.build()
  for iter in range(10):
    imgs, reshaped = pipe.run()
    if device == "gpu":
      verify_tensor_layouts(imgs, reshaped)
      imgs = imgs.as_cpu()
      reshaped = reshaped.as_cpu()
    verify_flatten(imgs, reshaped, (224, 320, 3))

def check_reshape_with_input(device, batch_size, use_wildcard):
  pipe = ReshapeWithInput(device, batch_size, use_wildcard)
  pipe.build()
  for iter in range(2):
    imgs, reshaped = pipe.run()
    if device == "gpu":
      verify_tensor_layouts(imgs, reshaped)
      imgs = imgs.as_cpu()
      reshaped = reshaped.as_cpu()
    verify_flatten(imgs, reshaped)

def check_reshape_with_arg_input(device, batch_size, relative, use_wildcard):
  pipe = ReshapeWithArgInput(device, batch_size, relative, use_wildcard)
  pipe.build()
  for iter in range(2):
    imgs, reshaped = pipe.run()
    if device == "gpu":
      imgs = imgs.as_cpu()
      reshaped = reshaped.as_cpu()
    verify_make_tall(imgs, reshaped)

def test_reshape_arg():
  for device in ["cpu", "gpu"]:
    for batch_size in [16]:
        for relative in [False, True]:
            for use_wildcard in [False, True]:
                yield check_reshape, device, batch_size, relative, use_wildcard

def test_reshape_input():
  for device in ["cpu", "gpu"]:
    for batch_size in [16]:
      for use_wildcard in [False, True]:
        yield check_reshape_with_input, device, batch_size, use_wildcard

def test_reshape_arg_input():
  for device in ["cpu", "gpu"]:
    for batch_size in [16]:
      for relative in [False, True]:
        for use_wildcard in [False, True]:
           yield check_reshape_with_arg_input, device, batch_size, relative, use_wildcard

class ReinterpretPipelineWithDefaultShape(Pipeline):
    def __init__(self, device, batch_size, num_threads=3, device_id=0, num_gpus=1):
        super(ReinterpretPipelineWithDefaultShape, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=True, exec_pipelined=True)
        self.device = device
        self.ext_src = ops.ExternalSource()
        self.reinterpret = ops.Reinterpret(device = device, dtype = types.INT32)

    def define_graph(self):
        input = self.input = self.ext_src()
        if self.device == "gpu":
          input = input.gpu()
        reinterpreted = self.reinterpret(input)

        # `input+0` creates a (no-op) arithmetic expression node - this prevents the
        # original `input` node from being marked as pipeline output
        return [input, reinterpreted]

    def iter_setup(self):
        data = []
        for i in range(self.batch_size):
            shape = np.random.randint(4, 20, size = [2])
            shape[1] &= -4  # align to 4
            data.append(np.random.randint(0, 255, shape, dtype = np.uint8))
        self.feed_input(self.input, data)


def _test_reinterpret_default_shape(device):
    np.random.seed(31337)
    batch_size = 4
    pipe = ReinterpretPipelineWithDefaultShape(device, batch_size)
    pipe.build()
    pipe_outs = pipe.run()
    in_batch = pipe_outs[0].as_cpu() if device == "gpu" else pipe_outs[0]
    out_batch = pipe_outs[1].as_cpu() if device == "gpu" else pipe_outs[1]
    for i in range(batch_size):
        ref = in_batch.at(i).view(dtype = np.int32)
        out = out_batch.at(i)
        assert_array_equal(ref, out)

def test_reinterpret_default_shape():
    for device in ["cpu", "gpu"]:
        yield _test_reinterpret_default_shape, device


class ReinterpretPipelineWildcardDim(Pipeline):
    def __init__(self, device, batch_size, num_threads=3, device_id=0, num_gpus=1):
        super(ReinterpretPipelineWildcardDim, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=True, exec_pipelined=True)
        self.device = device
        self.ext_src = ops.ExternalSource()
        self.reinterpret = ops.Reinterpret(device = device, shape = (20, 2), dtype = types.INT32)

    def define_graph(self):
        input = self.input = self.ext_src()
        if self.device == "gpu":
          input = input.gpu()
        reinterpreted = self.reinterpret(input)

        # `input+0` creates a (no-op) arithmetic expression node - this prevents the
        # original `input` node from being marked as pipeline output
        return [input, reinterpreted]

    def iter_setup(self):
        data = [np.random.randint(0, 255, [10, 16], dtype = np.uint8) for i in range(self.batch_size)]
        self.feed_input(self.input, data)


def _test_reinterpret_wildcard_shape(device):
    np.random.seed(31337)
    batch_size = 4
    pipe = ReinterpretPipelineWildcardDim(device, batch_size)
    pipe.build()
    pipe_outs = pipe.run()
    in_batch = pipe_outs[0].as_cpu() if device == "gpu" else pipe_outs[0]
    out_batch = pipe_outs[1].as_cpu() if device == "gpu" else pipe_outs[1]
    for i in range(batch_size):
        ref = in_batch.at(i).view(dtype = np.int32).reshape([20, 2])
        out = out_batch.at(i)
        assert_array_equal(ref, out)

def test_reinterpret_wildcard_shape():
    for device in ["cpu", "gpu"]:
        yield _test_reinterpret_wildcard_shape, device

def get_data(shapes):
    return [np.empty(shape, dtype = np.uint8) for shape in shapes]

@pipeline_def
def reshape_pipe(shapes, src_dims=None, rel_shape=None):
    data = fn.external_source(lambda: get_data(shapes), batch=True, device = "cpu")
    return fn.reshape(data, src_dims=src_dims, rel_shape=rel_shape)

def _testimpl_reshape_src_dims_arg(src_dims, rel_shape, shapes, expected_out_shapes):
    batch_size = len(shapes)
    pipe = reshape_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, src_dims=src_dims, rel_shape=rel_shape)
    pipe.build()
    for _ in range(3):
        outs = pipe.run()
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            assert out_arr.shape == expected_out_shapes[i]

def test_reshape_src_dims_arg():
    # src_dims, rel_shape, shapes, expected_out_shapes
    args = [
        ([0, 1], None, [[200, 300, 1], [300, 400, 1]], [(200, 300), (300, 400)]),
        ([1, 2, 0], None, [[10, 20, 30], [30, 20, 10], [2, 1, 3]], [(20, 30, 10), (20, 10, 30), (1, 3, 2)]),
        ([1], None, [[1, 2, 1], [1, 3, 1]], [(2,), (3,)]),
        ([2, -1, 1, 0], None, [[10, 20, 30]], [(30, 1, 20, 10)]),
        ([-1, 2], None, [[1, 1, 30], [1, 1, 70]], [(1, 30), (1, 70)]),
        ([2, 0, 1], [0.5, 0.5, -1], [[200, 300, 100]], [(50, 100, 1200)]),
        ([], None, [[1]], [()]),
    ]
    for src_dims, rel_shape, shapes, expected_out_shapes in args:
        yield _testimpl_reshape_src_dims_arg, src_dims, rel_shape, shapes, expected_out_shapes

def test_reshape_src_dims_throw_error():
    args = [
        ([2, 0], None, [[20, 10, 20]]),
        ([2, 0, 1], [1, -1], [[1, 2, 3]]),
        ([0, 1, 3], None, [1, 2, 3]),
    ]
    for src_dims, rel_shape, shapes in args:
        pipe = reshape_pipe(batch_size=len(shapes), num_threads=1, device_id=0, shapes=shapes, src_dims=src_dims, rel_shape=rel_shape)
        pipe.build()
        with assert_raises(RuntimeError):
          pipe.run()
