from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
import cv2
from test_utils import check_batch
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from PIL import Image         
test_data_root = os.environ['DALI_EXTRA_PATH']
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')



class WaterPipeline(Pipeline):
    def __init__(self, device, batch_size,  num_threads=1, device_id=0, num_gpus=1 ):
        super(WaterPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        self.water = ops.Water(device = self.device, ampl_x=2., ampl_y=3., phase_x=0.2, phase_y=0.5, freq_x=0.06,freq_y=0.08, interp_type = dali.types.INTERP_LINEAR)
        
    
    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        images = self.water(images)
        return images

def check_water_cpu_vs_gpu(batch_size):
    compare_pipelines(WaterPipeline('cpu', batch_size),
                      WaterPipeline('gpu', batch_size),
                      batch_size=batch_size, N_iterations=10, eps=1)


def python_water(img):
    nh,nw=img.shape[:2]
    phase_y=0.5
    phase_x=0.2
    freq_x=0.06
    freq_y=0.08
    ampl_x=2.0
    ampl_y=3.0
    img_x=np.zeros((nh,nw),np.float32)
    img_y=np.zeros((nh,nw),np.float32)
    x_idx = np.arange(0, nw, 1, np.float32)
    y_idx = np.arange(0, nh, 1, np.float32)
    x_wave = ampl_y * np.cos(freq_y * x_idx + phase_y)
    y_wave = ampl_x * np.sin(freq_x * y_idx + phase_x)
    for x in range(nw):
        img_x[:,x] = y_wave + x - 0.5
        
    for y in range(nh):
        img_y[y,:] = x_wave + y - 0.5
    return cv2.remap(img,img_x,img_y,cv2.INTER_LINEAR)


class WaterPythonPipeline(Pipeline):
    def __init__(self,  batch_size,function,  num_threads=1, device_id=0, num_gpus=1 ):
        super(WaterPythonPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id,
                                           exec_async=False,
                                           exec_pipelined=False)
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)
        self.water = ops.PythonFunction(function=function)


    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        images = self.water(images)
        return images



def test_water_cpu_vs_gpu():
   for batch_size in {1, 32, 100}:
       yield check_water_cpu_vs_gpu, batch_size 

def check_water_vs_cv(device, batch_size):
    python_func = python_water
    compare_pipelines(WaterPipeline(device, batch_size),
                      WaterPythonPipeline( batch_size, python_func),
                      batch_size=batch_size, N_iterations=10,eps=8)

def test_water_vs_cv():
    for device in ['cpu', 'gpu']:
        for batch_size in [1, 32, 100]:
            yield check_water_vs_cv, device,batch_size


