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
import numpy as np
import os
import cv2
import math
from test_utils import compare_pipelines
from test_utils import get_dali_extra_path

test_data_root = get_dali_extra_path()
caffe_db_folder = os.path.join(test_data_root, 'db', 'lmdb')

class WaterPipeline(Pipeline):
    def __init__(self, device, batch_size, phase_y, phase_x, freq_x, freq_y, ampl_x, ampl_y, num_threads=1, device_id=0, num_gpus=1):
        super(WaterPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id)
        self.device = device
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.water = ops.Water(device = self.device, ampl_x=ampl_x, ampl_y=ampl_y,
                               phase_x=phase_x, phase_y=phase_y, freq_x=freq_x, freq_y=freq_y,
                               interp_type = dali.types.INTERP_LINEAR)

    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        if self.device == 'gpu':
            images = images.gpu()
        images = self.water(images)
        return images

def python_water(img, phase_y, phase_x, freq_x, freq_y, ampl_x, ampl_y):
    nh,nw=img.shape[:2]
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

    return cv2.remap(img, img_x, img_y, cv2.INTER_LINEAR)

class WaterPythonPipeline(Pipeline):
    def __init__(self,  batch_size,function,  num_threads=1, device_id=0, num_gpus=1 ):
        super(WaterPythonPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id,
                                           exec_async=False,
                                           exec_pipelined=False)
        self.input = ops.CaffeReader(path = caffe_db_folder, shard_id = device_id, num_shards = num_gpus)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.water = ops.PythonFunction(function=function)


    def define_graph(self):
        inputs, labels = self.input(name="Reader")

        images = self.decode(inputs)
        images = self.water(images)
        return images

def check_water_cpu_vs_gpu(batch_size):
    phase_y=0.5
    phase_x=0.2
    freq_x=0.06
    freq_y=0.08
    ampl_x=2.0
    ampl_y=3.0

    pipe = WaterPipeline('cpu', batch_size, ampl_x=ampl_x, ampl_y=ampl_y,
                                    phase_x=phase_x, phase_y=phase_y, freq_x=freq_x, freq_y=freq_y)
    pipe.build()
    data_set_size = pipe.epoch_size("Reader")
    N_iterations = int(math.ceil(data_set_size/float(batch_size)))
    compare_pipelines(WaterPipeline('cpu', batch_size, ampl_x=ampl_x, ampl_y=ampl_y,
                                    phase_x=phase_x, phase_y=phase_y, freq_x=freq_x, freq_y=freq_y),
                      WaterPipeline('gpu', batch_size, ampl_x=ampl_x, ampl_y=ampl_y,
                                    phase_x=phase_x, phase_y=phase_y, freq_x=freq_x, freq_y=freq_y),
                      batch_size=batch_size, N_iterations=N_iterations, eps=1)

def test_water_cpu_vs_gpu():
   for batch_size in {1, 32, 100}:
       yield check_water_cpu_vs_gpu, batch_size

def check_water_vs_cv(device, batch_size):
    phase_y=0.5
    phase_x=0.2
    freq_x=0.06
    freq_y=0.08
    ampl_x=2.0
    ampl_y=3.0

    pipe = WaterPipeline('cpu', batch_size, ampl_x=ampl_x, ampl_y=ampl_y,
                                    phase_x=phase_x, phase_y=phase_y, freq_x=freq_x, freq_y=freq_y)
    pipe.build()
    data_set_size = pipe.epoch_size("Reader")
    N_iterations = int(math.ceil(data_set_size/float(batch_size)))

    python_func = lambda img: python_water(img, phase_y, phase_x, freq_x, freq_y, ampl_x, ampl_y)
    compare_pipelines(WaterPipeline(device, batch_size, ampl_x=ampl_x, ampl_y=ampl_y,
                                    phase_x=phase_x, phase_y=phase_y, freq_x=freq_x, freq_y=freq_y),
                      WaterPythonPipeline(batch_size, python_func),
                                          batch_size=batch_size, N_iterations=N_iterations,eps=8)

def test_water_vs_cv():
    for device in ['cpu', 'gpu']:
        for batch_size in [1, 32, 100]:
            yield check_water_vs_cv, device,batch_size
