# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

import cupy
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import test_utils
import random
import os


def random_seed():
    return int(random.random() * (1 << 32))


test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')


DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 32
SEED = random_seed()
NUM_WORKERS = 6


class PythonFunctionPipeline(Pipeline):
    def __init__(self, function, device, num_outputs=1):
        super(PythonFunctionPipeline, self).__init__(BATCH_SIZE, NUM_WORKERS, DEVICE_ID,
                                                     seed=SEED)
        self.device = device
        self.reader = ops.readers.File(file_root=images_dir)
        self.decode = ops.decoders.Image(device='cpu',
                                         output_type=types.RGB)
        self.norm = ops.CropMirrorNormalize(std=255., mean=0., device=device, output_layout="HWC")
        self.func = ops.PythonFunction(device=device, function=function, num_outputs=num_outputs)

    def define_graph(self):
        jpegs, labels = self.reader()
        decoded = self.decode(jpegs)
        images = decoded if self.device == 'cpu' else decoded.gpu()
        normalized = self.norm(images)
        return self.func(normalized, normalized)


def validate_cpu_vs_gpu(gpu_fun, cpu_fun, num_outputs=1):
    gpu_pipe = PythonFunctionPipeline(gpu_fun, 'gpu', num_outputs)
    cpu_pipe = PythonFunctionPipeline(cpu_fun, 'cpu', num_outputs)
    test_utils.compare_pipelines(gpu_pipe, cpu_pipe, BATCH_SIZE, ITERS)


def arrays_arithmetic(in1, in2):
    return in1 + in2, in1 - in2 / 2.


def test_simple_arithm():
    validate_cpu_vs_gpu(arrays_arithmetic, arrays_arithmetic, num_outputs=2)


square_diff_kernel = cupy.ElementwiseKernel(
    'T x, T y',
    'T z',
    'z = x*x - y*y',
    'square_diff'
)


def square_diff(in1, in2):
    return in1 * in1 - in2 * in2


def test_cupy_kernel():
    validate_cpu_vs_gpu(square_diff_kernel, square_diff)


def test_builtin_func():
    validate_cpu_vs_gpu(cupy.logaddexp, np.logaddexp)

gray_scale_kernel = cupy.RawKernel(r'''
    extern "C" __global__
    void gray_scale(float *output, const unsigned char *input, long long height, long long width) {
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        int tidy = blockIdx.y * blockDim.y + threadIdx.y;
        if (tidx < width && tidy < height) {
            int idx = (tidy * width + tidx) * 3;
            float r = input[idx];
            float g = input[idx + 1];
            float b = input[idx + 2];
            output[tidy * width + tidx] = 0.299 * r + 0.59 * g + 0.11 * b;
        }
    }
    ''', 'gray_scale')

def gray_scale_call(input):
    height = input.shape[0]
    width = input.shape[1]
    output = cupy.ndarray((height, width), dtype=cupy.float32)
    gray_scale_kernel(grid=((height + 31) // 32, (width + 31) // 32),
                      block=(32, 32),
                      args=(output, input, height, width))
    return output

def cupy_kernel_gray_scale(in1, in2):
    print('STOP 2')
    # out1 = [gray_scale_call(arr) for arr in in1]
    out1 = gray_scale_call(in1)
    out2 = gray_scale_call(in2)
    return out1, out2

def gray_scale_cpu(inp1, inp2):
    out1 = inp1[:, :, 0] * 0.299 + inp1[:, :, 1] * 0.59 + inp1[:, :, 2] * 0.11
    out2 = inp2[:, :, 0] * 0.299 + inp2[:, :, 1] * 0.59 + inp2[:, :, 2] * 0.11
    return out1, out2

def test_raw_kernel():
    validate_cpu_vs_gpu(cupy_kernel_gray_scale, gray_scale_cpu, num_outputs=2)