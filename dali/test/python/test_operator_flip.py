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
test_data_video = os.path.join(test_data_root, 'db', 'optical_flow', 'sintel_trailer')



class FlipPipeline(Pipeline):
    def __init__(self, device, batch_size, num_threads=1, device_id=0, num_gpus=1, is_vertical=0, is_horizontal=1 ):
        super(FlipPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu",
                                          output_type = types.RGB)
        self.flip = ops.Flip(device = self.device, vertical=is_vertical, horizontal=is_horizontal)
        

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        images = self.flip(images)
        return images

class FlipPythonOpPipeline(Pipeline):
    def __init__(self,  batch_size,function, num_threads=1, device_id=0, num_gpus=1 ):
        super(FlipPythonOpPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id, exec_async=False,exec_pipelined=False)
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu",
                                          output_type = types.RGB)
        self.flip = ops.PythonFunction(function=function)
        

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        images = self.flip(images)
        return images
	
def check_flip_cpu_vs_gpu(batch_size):
    compare_pipelines(FlipPipeline('cpu', batch_size),
                      FlipPipeline('gpu', batch_size),
                      batch_size=batch_size, N_iterations=10)
			
def test_flip_cpu_vs_gpu():
    for batch_size in {1, 32, 100}:
        yield check_flip_cpu_vs_gpu, batch_size			
			

def flip_horizontal(image):
    return np.fliplr(image)

def flip_vertical(image):
    return np.flipud(image)	
			
def flip_rotate(image):
    return np.flipud(np.fliplr(image))

def check_flip_vs_numpy_horizontal(batch_size):
    compare_pipelines(FlipPipeline('cpu', batch_size),
                      FlipPythonOpPipeline(batch_size, flip_horizontal),
                      batch_size=batch_size, N_iterations=10)
    compare_pipelines(FlipPipeline('gpu', batch_size),
                      FlipPythonOpPipeline(batch_size, flip_horizontal),
                      batch_size=batch_size, N_iterations=10)

def test_flip_vs_numpy_horizontal():
    for batch_size in {1, 32, 100}:
        yield check_flip_vs_numpy_horizontal, batch_size	

def check_flip_vs_numpy_vertical(batch_size):
    compare_pipelines(FlipPipeline('cpu', batch_size, is_vertical=1, is_horizontal=0),
                      FlipPythonOpPipeline(batch_size, flip_vertical),
                      batch_size=batch_size, N_iterations=10)
    compare_pipelines(FlipPipeline('gpu', batch_size,is_vertical=1, is_horizontal=0),
                      FlipPythonOpPipeline(batch_size, flip_vertical),
                      batch_size=batch_size, N_iterations=10)

def test_flip_vs_numpy_vertical():
    for batch_size in {1, 32, 100}:
        yield check_flip_vs_numpy_vertical, batch_size

def check_flip_vs_numpy_rotate(batch_size):
    compare_pipelines(FlipPipeline('cpu', batch_size, is_vertical=1, is_horizontal=1),
                      FlipPythonOpPipeline(batch_size, flip_rotate),
                      batch_size=batch_size, N_iterations=10)
    compare_pipelines(FlipPipeline('gpu', batch_size,is_vertical=1, is_horizontal=1),
                      FlipPythonOpPipeline(batch_size, flip_rotate),
                      batch_size=batch_size, N_iterations=10)

def test_flip_vs_numpy_rotate():
    for batch_size in {1, 32, 100}:
        yield check_flip_vs_numpy_rotate, batch_size
