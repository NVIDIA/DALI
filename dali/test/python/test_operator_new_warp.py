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

def CVWarp(output_type, input_type):
  def warp_fn(img, matrix):
    size = (320, 240)
    matrix = ToCVMatrix(matrix)
    if output_type == dali.types.FLOAT or input_type == dali.types.FLOAT:
      img = np.float32(img)
    out = cv2.warpAffine(img, matrix, size, borderMode = cv2.BORDER_CONSTANT, borderValue = 0,
      flags = (cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP));
    if output_type == dali.types.UINT8 and input_type == dali.types.FLOAT:
      out = np.uint8(np.clip(out, 0, 255))
    return out
  return warp_fn


class NewWarpPipeline(Pipeline):
    def __init__(self, device, batch_size, output_type, input_type, num_threads=1, device_id=0, num_gpus=1):
        super(NewWarpPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.name = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        if input_type != dali.types.UINT8:
          self.cast = ops.Cast(device = device, dtype = input_type)
        else:
          self.cast = None
        self.warp = ops.NewWarpAffine(device = device, size=(240,320), output_type = output_type)
        self.transform_source = ops.ExternalSource()
        self.iter = 0

    def define_graph(self):
        self.transform = self.transform_source()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        if self.warp.device == "gpu":
          images = images.gpu()
        if self.cast:
          images = self.cast(images)
        outputs = self.warp(images, self.transform)
        return outputs

    def iter_setup(self):
        self.feed_input(self.transform, gen_transforms(self.batch_size, 10))


class CVPipeline(Pipeline):
    def __init__(self, batch_size, output_type, input_type, num_threads=1, device_id=0, num_gpus=1):
        super(CVPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.name = "cv"
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.warp = ops.PythonFunction(function=CVWarp(output_type, input_type))
        self.transform_source = ops.ExternalSource()
        self.iter = 0

    def define_graph(self):
        self.transform = self.transform_source()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        outputs = self.warp(images, self.transform)
        return outputs

    def iter_setup(self):
        self.feed_input(self.transform, gen_transforms(self.batch_size, 10))


def compare(pipe1, pipe2, eps):
  epochSize = pipe1.epoch_size("Reader")
  n = 0
  while n < epochSize:
    out1 = pipe1.run()[0]
    out2 = pipe2.run()[0]
    n += len(out1)
    if hasattr(out1, 'as_cpu'):
      out1 = out1.as_cpu()
    if hasattr(out2, 'as_cpu'):
      out2 = out2.as_cpu()
    for i in range(len(out1)):
      img1 = out1.at(i)
      img2 = out2.at(i)
      dif = cv2.absdiff(img1, img2)
      err = np.amax(dif)
      if (err > eps):
        print("Max. difference ", err, " > ", eps)
        cv2.imwrite("out_%0d_%s.png"%(i, pipe1.name), img1)
        cv2.imwrite("out_%0d_%s.png"%(i, pipe2.name), img2)
        cv2.imwrite("dif_%0d_%s_%s.png"%(i, pipe1.name, pipe2.name), dif*10)
        assert(err <= eps)

io_types = [
  (dali.types.UINT8, dali.types.UINT8),
  (dali.types.UINT8, dali.types.FLOAT),
  (dali.types.FLOAT, dali.types.UINT8),
  (dali.types.FLOAT, dali.types.FLOAT)
]


def test_cpu_vs_cv():
  for (itype, otype) in io_types:
    cv_pipeline = CVPipeline(10, otype, itype);
    cv_pipeline.build();

    cpu_pipeline = NewWarpPipeline("cpu", 10, otype, itype);
    cpu_pipeline.build();

    compare(cv_pipeline, cpu_pipeline, 8)

def test_gpu_vs_cv():
  for (itype, otype) in io_types:
    cv_pipeline = CVPipeline(10, otype, itype);
    cv_pipeline.build();

    gpu_pipeline = NewWarpPipeline("gpu", 10, otype, itype);
    gpu_pipeline.build();

  compare(cv_pipeline, gpu_pipeline, 8)

def test_gpu_vs_cpu():
  for (itype, otype) in io_types:
    cpu_pipeline = NewWarpPipeline("cpu", 10, otype, itype);
    cpu_pipeline.build();

    gpu_pipeline = NewWarpPipeline("gpu", 10, otype, itype);
    gpu_pipeline.build();

    compare(cpu_pipeline, gpu_pipeline, 1)



def main():
  test_cpu_vs_cv()
  test_gpu_vs_cv()
  test_gpu_vs_cpu()

if __name__ == '__main__':
  main()
