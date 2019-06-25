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


class RotatePipeline(Pipeline):
    def __init__(self, device, batch_size, angle, num_threads=1, device_id=0, num_gpus=1 ):
        super(RotatePipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        self.rotate = ops.Rotate(device = self.device, angle=angle)


    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        images = self.rotate(images)
        return images


class RotatePythonOpPipeline(Pipeline):
    def __init__(self,  batch_size,function, num_threads=1, device_id=0, num_gpus=1 ):
        super(RotatePythonOpPipeline, self).__init__(batch_size,
                                                   num_threads,
                                                   device_id,
                                                   exec_async=False,
                                                   exec_pipelined=False)
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        self.rotate = ops.PythonFunction(function=function)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        images = self.rotate(images)
        return images

def check_rotate_cpu_vs_gpu(batch_size, angle):
    compare_pipelines(RotatePipeline('cpu', batch_size, angle),
                      RotatePipeline('gpu', batch_size, angle),
                      batch_size=batch_size, N_iterations=10)

def test_rotate_cpu_vs_gpu():
    for batch_size in {1, 32, 100}:
        for angle in range(0,360):
            yield check_rotate_cpu_vs_gpu, batch_size, float(angle)

