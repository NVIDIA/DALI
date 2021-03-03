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

import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import os
import random
import numpy
from functools import partial

test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')

def setup_pytorch():
    global torch_dlpack
    global torch
    import torch.utils.dlpack as torch_dlpack
    import torch as torch
    global torch_stream
    torch_stream = torch.cuda.Stream()

def setup_mxnet():
    global mxnd
    from mxnet import ndarray as mxnd

def setup_cupy():
    global cupy
    global cupy_stream
    global square_diff_kernel
    global mix_channels_kernel
    global gray_scale_kernel
    import cupy as cupy
    cupy_stream = cupy.cuda.Stream()
    square_diff_kernel = cupy.ElementwiseKernel(
        'T x, T y',
        'T z',
        'z = x*x - y*y',
        'square_diff'
    )

    mix_channels_kernel = cupy.ElementwiseKernel(
        'uint8 x, uint8 y',
        'uint8 z',
        'z = (i % 3) ? x : y',
        'mix_channels'
    )

    gray_scale_kernel = cupy.RawKernel(r'''
    extern "C" __global__
    void gray_scale(float *output, const unsigned char *input, long long height, long long width) {
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        int tidy = blockIdx.y * blockDim.y + threadIdx.y;
        if (tidx < width && tidy < height) {
            float r = input[tidy * width + tidx] / 255.;
            float g = input[tidy * width + tidx + 1] / 255.;
            float b = input[tidy * width + tidx + 2] / 255.;
            output[tidy * width + tidx] = 0.299 * r + 0.59 * g + 0.11 * b;
        }
    }
    ''', 'gray_scale')

def random_seed():
    return int(random.random() * (1 << 32))


DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 128
SEED = random_seed()
NUM_WORKERS = 6


class CommonPipeline(Pipeline):
    def __init__(self, device):
        super(CommonPipeline, self).__init__(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, seed=SEED,
                                             exec_async=False, exec_pipelined=False)
        self.input = ops.readers.File(file_root=images_dir)
        self.decode = ops.decoders.Image(device='mixed' if device == 'gpu' else 'cpu',
                                         output_type=types.RGB)
        self.resize = ops.Resize(resize_x=400, resize_y=400, device=device)
        self.flip = ops.Flip(device=device)

    def load(self):
        jpegs, labels = self.input()
        decoded = self.decode(jpegs)
        return self.resize(decoded)


class LoadingPipeline(CommonPipeline):
    def __init__(self, device):
        super(LoadingPipeline, self).__init__(device)

    def define_graph(self):
        im = self.load()
        im2 = self.load()
        return im, self.flip(im2)


class DLTensorOpPipeline(CommonPipeline):
    def __init__(self, function, device, synchronize=True):
        super(DLTensorOpPipeline, self).__init__(device)
        self.op = ops.DLTensorPythonFunction(function=function, device=device, num_outputs=2,
                                             synchronize_stream=synchronize)

    def define_graph(self):
        im = self.load()
        im2 = self.load()
        return self.op(im, self.flip(im2))


def pytorch_adapter(fun, in1, in2):
    with torch.cuda.stream(torch_stream):
        tin1 = [torch_dlpack.from_dlpack(dltensor) for dltensor in in1]
        tin2 = [torch_dlpack.from_dlpack(dltensor) for dltensor in in2]
        tout1, tout2 = fun(tin1, tin2)
        out1, out2 = [torch_dlpack.to_dlpack(tout) for tout in tout1], \
                     [torch_dlpack.to_dlpack(tout) for tout in tout2]
    torch_stream.synchronize()
    return out1, out2


def pytorch_wrapper(fun):
    return lambda in1, in2: pytorch_adapter(fun, in1, in2)


def common_case(wrapped_fun, device, compare, synchronize=True):
    load_pipe = LoadingPipeline(device)
    op_pipe = DLTensorOpPipeline(wrapped_fun, device, synchronize)

    load_pipe.build()
    op_pipe.build()

    for iter in range(ITERS):
        pre1, pre2 = load_pipe.run()
        post1, post2 = op_pipe.run()

        if device == 'gpu':
            pre1 = pre1.as_cpu()
            pre2 = pre2.as_cpu()
            post1 = post1.as_cpu()
            post2 = post2.as_cpu()

        compare(pre1, pre2, post1, post2)


def pytorch_compare(fun, pre1, pre2, post1, post2):
    torch_pre1 = [torch.from_numpy(pre1.at(i)) for i in range(BATCH_SIZE)]
    torch_pre2 = [torch.from_numpy(pre2.at(i)) for i in range(BATCH_SIZE)]
    torch_post1, torch_post2 = fun(torch_pre1, torch_pre2)
    for i in range(BATCH_SIZE):
        assert numpy.array_equal(post1.at(i), torch_post1[i].numpy())
        assert numpy.array_equal(post2.at(i), torch_post2[i].numpy())


def pytorch_case(fun, device):
    common_case(pytorch_wrapper(fun), device, partial(pytorch_compare, fun))


def simple_pytorch_op(in1, in2):
    fin1 = [t.to(dtype=torch.float) for t in in1]
    fin2 = [t.to(dtype=torch.float) for t in in2]
    return [fin1[i] + fin2[i] for i in range(len(fin1))], \
           [fin1[i] - fin2[i] for i in range(len(fin1))]


def pytorch_red_channel_op(in1, in2):
    return [t.narrow(2, 0, 1).squeeze() for t in in1], [t.narrow(2, 0, 1).squeeze() for t in in2]


def test_pytorch():
    setup_pytorch()
    for testcase in [simple_pytorch_op, pytorch_red_channel_op]:
        for device in ['cpu', 'gpu']:
            yield pytorch_case, testcase, device


def mxnet_adapter(fun, in1, in2):
    tin1 = [mxnd.from_dlpack(dltensor) for dltensor in in1]
    tin2 = [mxnd.from_dlpack(dltensor) for dltensor in in2]
    tout1, tout2 = fun(tin1, tin2)
    return [mxnd.to_dlpack_for_read(tout) for tout in tout1], \
           [mxnd.to_dlpack_for_read(tout) for tout in tout2]


def mxnet_wrapper(fun):
    return lambda in1, in2: mxnet_adapter(fun, in1, in2)


def mxnet_compare(fun, pre1, pre2, post1, post2):
    mxnet_pre1 = [mxnd.array(pre1.at(i)) for i in range(BATCH_SIZE)]
    mxnet_pre2 = [mxnd.array(pre2.at(i)) for i in range(BATCH_SIZE)]
    mxnet_post1, mxnet_post2 = fun(mxnet_pre1, mxnet_pre2)
    for i in range(BATCH_SIZE):
        assert numpy.array_equal(post1.at(i), mxnet_post1[i].asnumpy())
        assert numpy.array_equal(post2.at(i), mxnet_post2[i].asnumpy())


def mxnet_case(fun, device):
    setup_mxnet()
    common_case(mxnet_wrapper(fun), device, partial(mxnet_compare, fun))




def mxnet_flatten(in1, in2):
    return [mxnd.flatten(t) for t in in1], [mxnd.flatten(t) for t in in2]


def mxnet_slice(in1, in2):
    return [t[:, :, 1] for t in in1], [t[:, :, 2] for t in in2]


def mxnet_cast(in1, in2):
    return [mxnd.cast(t, dtype='float32') for t in in1], [mxnd.cast(t, dtype='int64') for t in in2]


def test_mxnet():
    for testcase in [mxnet_flatten, mxnet_slice, mxnet_cast]:
        for device in ['cpu', 'gpu']:
            yield mxnet_case, testcase, device



def cupy_adapter_sync(fun, in1, in2):
    with cupy_stream:
        tin1 = [cupy.fromDlpack(dltensor) for dltensor in in1]
        tin2 = [cupy.fromDlpack(dltensor) for dltensor in in2]
        tout1, tout2 = fun(tin1, tin2)
        out1, out2 = [tout.toDlpack() for tout in tout1], \
                     [tout.toDlpack() for tout in tout2]
    cupy_stream.synchronize()
    return out1, out2


def cupy_adapter(fun, in1, in2):
    tin1 = [cupy.fromDlpack(dltensor) for dltensor in in1]
    tin2 = [cupy.fromDlpack(dltensor) for dltensor in in2]
    tout1, tout2 = fun(tin1, tin2)
    return [tout.toDlpack() for tout in tout1], \
           [tout.toDlpack() for tout in tout2]


def cupy_wrapper(fun, synchronize):
    if synchronize:
        return lambda in1, in2: cupy_adapter_sync(fun, in1, in2)
    else:
        return lambda in1, in2: cupy_adapter(fun, in1, in2)


def cupy_compare(fun, pre1, pre2, post1, post2):
    cupy_pre1 = [cupy.asarray(pre1.at(i)) for i in range(BATCH_SIZE)]
    cupy_pre2 = [cupy.asarray(pre2.at(i)) for i in range(BATCH_SIZE)]
    cupy_post1, cupy_post2 = fun(cupy_pre1, cupy_pre2)
    for i in range(BATCH_SIZE):
        assert post1.at(i).shape == cupy_post1[i].shape
        assert post2.at(i).shape == cupy_post2[i].shape
        assert numpy.array_equal(post1.at(i), cupy.asnumpy(cupy_post1[i]))
        assert numpy.array_equal(post2.at(i), cupy.asnumpy(cupy_post2[i]))


def cupy_case(fun, synchronize=True):
    common_case(cupy_wrapper(fun, synchronize), 'gpu', partial(cupy_compare, fun), synchronize)


def cupy_simple(in1, in2):
    fin1 = [arr.astype(cupy.float32) for arr in in1]
    fin2 = [arr.astype(cupy.float32) for arr in in2]
    return [cupy.sin(fin1[i]*fin2[i]).astype(cupy.float32) for i in range(BATCH_SIZE)], \
           [cupy.cos(fin1[i]*fin2[i]).astype(cupy.float32) for i in range(BATCH_SIZE)]


def gray_scale_call(input):
    height = input.shape[0]
    width = input.shape[1]
    output = cupy.ndarray((height, width), dtype=cupy.float32)
    gray_scale_kernel(grid=((height + 31) // 32, (width + 31) // 32),
                      block=(32, 32),
                      stream=ops.PythonFunction.current_stream(),
                      args=(output, input, height, width))
    return output


def cupy_kernel_square_diff(in1, in2):
    fin1 = [arr.astype(cupy.float32) for arr in in1]
    fin2 = [arr.astype(cupy.float32) for arr in in2]
    out1, out2 = [square_diff_kernel(fin1[i], fin2[i]) for i in range(BATCH_SIZE)], in2
    return out1, out2


def cupy_kernel_mix_channels(in1, in2):
    return [mix_channels_kernel(in1[i], in2[i]) for i in range(BATCH_SIZE)], in2


def cupy_kernel_gray_scale(in1, in2):
    out1 = [gray_scale_call(arr) for arr in in1]
    out2 = [gray_scale_call(arr) for arr in in2]
    return out1, out2


def test_cupy():
    setup_cupy()
    print(cupy)
    for testcase in [cupy_simple,  cupy_kernel_square_diff, cupy_kernel_mix_channels]:
        yield cupy_case, testcase


def test_cupy_kernel_gray_scale():
    setup_cupy()
    cupy_case(cupy_kernel_gray_scale, synchronize=False)
