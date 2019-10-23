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

from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os
import random
import torch
import nvidia.dali.plugin.pytorch as dalitorch
import numpy
from test_utils import get_dali_extra_path


test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')


def random_seed():
    return int(random.random() * (1 << 32))


DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 64
SEED = random_seed()
NUM_WORKERS = 6


class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, _seed, image_dir):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id, seed=_seed,
                                             exec_async=False, exec_pipelined=False)
        self.input = ops.FileReader(file_root=image_dir)
        self.decode = ops.ImageDecoder(device='cpu', output_type=types.RGB)

    def load(self):
        jpegs, labels = self.input()
        decoded = self.decode(jpegs)
        return decoded, labels


class BasicPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super(BasicPipeline, self).__init__(batch_size, num_threads, device_id, seed, image_dir)

    def define_graph(self):
        images, labels = self.load()
        return images


class TorchPythonFunctionPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function):
        super(TorchPythonFunctionPipeline, self).__init__(batch_size, num_threads,
                                                          device_id, seed, image_dir)
        self.torch_function = dalitorch.TorchPythonFunction(function=function, num_outputs=2)

    def define_graph(self):
        images, labels = self.load()
        return self.torch_function(images)


def torch_operation(tensor):
    tensor_n = tensor.double() / 255
    return tensor_n.sin(), tensor_n.cos()


def test_torch_operator():
    pipe = BasicPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pt_pipe = TorchPythonFunctionPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID,
                                          SEED, images_dir, torch_operation)
    pipe.build()
    pt_pipe.build()
    for it in range(ITERS):
        preprocessed_output, = pipe.run()
        output1, output2 = pt_pipe.run()
        for i in range(len(output1)):
            res1 = output1.at(i)
            res2 = output2.at(i)
            exp1_t, exp2_t = torch_operation(torch.from_numpy(preprocessed_output.at(i)))
            assert numpy.array_equal(res1, exp1_t.numpy())
            assert numpy.array_equal(res2, exp2_t.numpy())
