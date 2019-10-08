import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import torch.utils.dlpack as torch_dlpack
import torch
import os
import random
import numpy
from mxnet import ndarray as mxnd
import cupy
import functools
import ctypes

test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')


def random_seed():
    return int(random.random() * (1 << 32))


DEVICE_ID = 0
BATCH_SIZE = 4
ITERS = 64
SEED = random_seed()
NUM_WORKERS = 6


class CommonPipeline(Pipeline):
    def __init__(self, device):
        super(CommonPipeline, self).__init__(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, seed=SEED,
                                             exec_async=False, exec_pipelined=False)
        self.input = ops.FileReader(file_root=images_dir)
        self.decode = ops.ImageDecoder(device='mixed' if device == 'gpu' else 'cpu',
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
    def __init__(self, function, device):
        super(DLTensorOpPipeline, self).__init__(device)
        self.op = ops.DLTensorPythonFunction(function=function, device=device, num_outputs=2)

    def define_graph(self):
        im = self.load()
        im2 = self.load()
        return self.op(im, self.flip(im2))


torch_stream = torch.cuda.Stream()


def torch_adapter(fun, in1, in2):
    with torch.cuda.stream(torch_stream):
        tin1 = [torch_dlpack.from_dlpack(dltensor) for dltensor in in1]
        tin2 = [torch_dlpack.from_dlpack(dltensor) for dltensor in in2]
        tout1, tout2 = fun(tin1, tin2)
        out1, out2 = [torch_dlpack.to_dlpack(tout) for tout in tout1], \
                     [torch_dlpack.to_dlpack(tout) for tout in tout2]
    torch_stream.synchronize()
    return out1, out2


def torch_wrapper(fun):
    return lambda in1, in2: torch_adapter(fun, in1, in2)


def torch_case(fun, device):
    load_pipe = LoadingPipeline(device)
    op_pipe = DLTensorOpPipeline(torch_wrapper(fun), device)

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

        torch_pre1 = [torch.from_numpy(pre1.at(i)) for i in range(BATCH_SIZE)]
        torch_pre2 = [torch.from_numpy(pre2.at(i)) for i in range(BATCH_SIZE)]
        torch_post1, torch_post2 = fun(torch_pre1, torch_pre2)
        for i in range(BATCH_SIZE):
            assert numpy.array_equal(post1.at(i), torch_post1[i].numpy())
            assert numpy.array_equal(post2.at(i), torch_post2[i].numpy())


def simple_torch_op(in1, in2):
    fin1 = [t.to(dtype=torch.float) for t in in1]
    fin2 = [t.to(dtype=torch.float) for t in in2]
    return [fin1[i] + fin2[i] for i in range(len(fin1))], \
           [fin1[i] - fin2[i] for i in range(len(fin1))]


def torch_red_channel_op(in1, in2):
    return [t.narrow(2, 0, 1).squeeze() for t in in1], [t.narrow(2, 0, 1).squeeze() for t in in2]


def test_torch_simple_cpu():
    torch_case(simple_torch_op, 'cpu')


def test_torch_red_channel_cpu():
    torch_case(torch_red_channel_op, 'cpu')


def test_torch_simple_gpu():
    torch_case(simple_torch_op, 'gpu')


def test_torch_red_channel_gpu():
    torch_case(torch_red_channel_op, 'gpu')


def mxnet_adapter(fun, in1, in2):
    tin1 = [mxnd.from_dlpack(dltensor) for dltensor in in1]
    tin2 = [mxnd.from_dlpack(dltensor) for dltensor in in2]
    tout1, tout2 = fun(tin1, tin2)
    return [mxnd.to_dlpack_for_read(tout) for tout in tout1], \
           [mxnd.to_dlpack_for_read(tout) for tout in tout2]


def mxnet_wrapper(fun):
    return lambda in1, in2: mxnet_adapter(fun, in1, in2)


def mxnet_case(fun, device):
    load_pipe = LoadingPipeline(device)
    op_pipe = DLTensorOpPipeline(mxnet_wrapper(fun), device)

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

        mxnet_pre1 = [mxnd.array(pre1.at(i)) for i in range(BATCH_SIZE)]
        mxnet_pre2 = [mxnd.array(pre2.at(i)) for i in range(BATCH_SIZE)]
        mxnet_post1, mxnet_post2 = fun(mxnet_pre1, mxnet_pre2)
        for i in range(BATCH_SIZE):
            assert numpy.array_equal(post1.at(i), mxnet_post1[i].asnumpy())
            assert numpy.array_equal(post2.at(i), mxnet_post2[i].asnumpy())


def mxnet_flatten(in1, in2):
    return [mxnd.flatten(t) for t in in1], [mxnd.flatten(t) for t in in2]


def mxnet_slice(in1, in2):
    return [t[:, :, 1] for t in in1], [t[:, :, 2] for t in in2]


def mxnet_cast(in1, in2):
    return [mxnd.cast(t, dtype='float32') for t in in1], [mxnd.cast(t, dtype='int64') for t in in2]


def test_mxnet_flatten_cpu():
    mxnet_case(mxnet_flatten, device='cpu')


def test_mxnet_flatten_gpu():
    mxnet_case(mxnet_flatten, device='gpu')


def test_mxnet_slice_cpu():
    mxnet_case(mxnet_slice, device='cpu')


def test_mxnet_slice_gpu():
    mxnet_case(mxnet_slice, device='gpu')


def test_mxnet_cast():
    mxnet_case(mxnet_cast, device='cpu')


cupy_stream = cupy.cuda.Stream()


def cupy_adapter(fun, in1, in2):
    with cupy_stream:
        tin1 = [cupy.fromDlpack(dltensor) for dltensor in in1]
        tin2 = [cupy.fromDlpack(dltensor) for dltensor in in2]
        tout1, tout2 = fun(tin1, tin2)
        out1, out2 = [tout.toDlpack() for tout in tout1], \
                     [tout.toDlpack() for tout in tout2]
    cupy_stream.synchronize()
    return out1, out2


def cupy_wrapper(fun):
    return lambda in1, in2: cupy_adapter(fun, in1, in2)


def cupy_case(fun):
    load_pipe = LoadingPipeline('gpu')
    op_pipe = DLTensorOpPipeline(cupy_wrapper(fun), 'gpu')

    load_pipe.build()
    op_pipe.build()

    for iter in range(ITERS):
        pre1, pre2 = load_pipe.run()
        post1, post2 = op_pipe.run()
        pre1 = pre1.as_cpu()
        pre2 = pre2.as_cpu()
        post1 = post1.as_cpu()
        post2 = post2.as_cpu()
        cupy_pre1 = [cupy.asarray(pre1.at(i)) for i in range(BATCH_SIZE)]
        cupy_pre2 = [cupy.asarray(pre2.at(i)) for i in range(BATCH_SIZE)]
        cupy_post1, cupy_post2 = fun(cupy_pre1, cupy_pre2)
        print("iter: " + str(iter))
        for i in range(BATCH_SIZE):
            print("i: " + str(i))
            print("pre1 : ")
            print(pre1.at(i))
            print("pre2 : ")
            print(pre2.at(i))
            print("post 1 data: ")
            print(post1.at(i))
            print("cupy 1 data: ")
            print(cupy.asnumpy(cupy_post1[i]))
            print("post 2 data: ")
            print(post2.at(i))
            print("cupy 2 data: ")
            print(cupy.asnumpy(cupy_post2[i]))
            assert post1.at(i).shape == cupy_post1[i].shape
            assert post2.at(i).shape == cupy_post2[i].shape
            assert numpy.allclose(post1.at(i), cupy.asnumpy(cupy_post1[i]), atol=0.0001)
            assert numpy.allclose(post2.at(i), cupy.asnumpy(cupy_post2[i]), atol=0.0001)


def cupy_simple(in1, in2):
    fin1 = [arr.astype(cupy.float32) for arr in in1]
    fin2 = [arr.astype(cupy.float32) for arr in in2]
    return [cupy.sin(fin1[i]*fin2[i]).astype(cupy.float32) for i in range(BATCH_SIZE)], \
           [cupy.cos(fin1[i]*fin2[i]).astype(cupy.float32) for i in range(BATCH_SIZE)]


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


class StreamWrapper:
    def __init__(self, ptr):
        self.ptr = ptr


def gray_scale_call(input):
    height = input.shape[0]
    width = input.shape[1]
    output = cupy.ndarray((height, width), dtype=cupy.float)
    gray_scale_kernel(grid=((height + 31) // 32, (width + 31) // 32),
                      block=(32, 32),
                      stream=StreamWrapper(ops.current_dali_stream()),
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


# def test_cupy_simple():
#     cupy_case(cupy_simple)
#
#
# def test_cupy_kernel_square_diff():
#     cupy_case(cupy_kernel_square_diff)
#
#
# def test_cupy_kernel_mix_channels():
#     cupy_case(cupy_kernel_mix_channels)
#

def test_cupy_kernel_gray_scale():
    cupy_case(cupy_kernel_gray_scale)
