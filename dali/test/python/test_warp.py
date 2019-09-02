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

def CVWarpBatch(img, matrix):
  out = []
  for i in range(len(img)):
    out.append(CVWarp(img[i], matrix[i]))
  return out

def CVWarp(img, matrix):
  size = (320, 240)
  matrix = ToCVMatrix(matrix)
  out = cv2.warpAffine(img, matrix, size, borderMode = cv2.BORDER_CONSTANT, borderValue = 0,
    flags = (cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP));
  return out


class NewWarpPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1):

        super(NewWarpPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        dec_device = "cpu" if device == "cpu" else "mixed"
        self.decode = ops.ImageDecoder(device = dec_device, output_type = types.RGB)
        self.warp = ops.NewWarpAffine(device = device, size=(240,320))
        self.transform_source = ops.ExternalSource()
        self.iter = 0

    def define_graph(self):
        self.transform = self.transform_source()
        self.jpegs, self.labels = self.input()
        images = self.decode(self.jpegs)
        outputs = self.warp(images, self.transform)
        return outputs

    def iter_setup(self):
        self.feed_input(self.transform, gen_transforms(self.batch_size, 10))


class PythonOpPipeline(Pipeline):
    def __init__(self, batch_size, function, num_threads=1, device_id=0, num_gpus=1):

        super(PythonOpPipeline, self).__init__(batch_size, num_threads, device_id, seed=7865, exec_async=False, exec_pipelined=False)
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.warp = ops.PythonFunction(function=function)
        self.transform_source = ops.ExternalSource()
        self.iter = 0

    def define_graph(self):
        self.transform = self.transform_source()
        self.jpegs, self.labels = self.input()
        images = self.decode(self.jpegs)
        outputs = self.warp(images, self.transform)
        return outputs

    def iter_setup(self):
        self.feed_input(self.transform, gen_transforms(self.batch_size, 10))


def test(ref_pipe, test_pipe, eps = 8):
  ref_pipeline = PythonOpPipeline(10, CVWarp);
  ref_pipeline.build();

  ref_pipeline = NewWarpPipeline(device, 10);
  cpu_pipeline.build();
  cpu_batch = cpu_pipeline.run()
  ref_batch = ref_pipeline.run()
  for i in range(len(cpu_batch[0])):
    err = np.amax(cv2.absdiff(cpu_batch[0].at(i), ref_batch[0].at(i)))
    assert(err < eps)

    cv2.imwrite("out_%0d_ref.png"%i, ref_batch[0].at(i))
    cv2.imwrite("out_%0d_cpu.png"%i, cpu_batch[0].at(i))


def test_cv_cpu():
  pass



def main():
  ref_pipeline = PythonOpPipeline(10, CVWarp);
  ref_pipeline.build();

  cpu_pipeline = NewWarpPipeline("cpu", 10);
  cpu_pipeline.build();
  cpu_batch = cpu_pipeline.run()
  ref_batch = ref_pipeline.run()
  for i in range(len(cpu_batch[0])):
    err = np.amax(cv2.absdiff(cpu_batch[0].at(i), ref_batch[0].at(i)))
    assert(err < eps)

    cv2.imwrite("out_%0d_ref.png"%i, ref_batch[0].at(i))
    cv2.imwrite("out_%0d_cpu.png"%i, cpu_batch[0].at(i))

if __name__ == '__main__':
  main()
