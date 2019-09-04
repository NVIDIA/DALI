import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import torch.utils.dlpack as torch_dlpack
import torch
import os
import random
import numpy
from mxnet import ndarray as mxnd

test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')


def random_seed():
    return int(random.random() * (1 << 32))


DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 32
SEED = random_seed()
NUM_WORKERS = 6


class CommonPipeline(Pipeline):
    def __init__(self):
        super(CommonPipeline, self).__init__(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, seed=SEED,
                                             exec_async=False, exec_pipelined=False)
        self.input = ops.FileReader(file_root=images_dir)
        self.decode = ops.ImageDecoder(device='cpu', output_type=types.RGB)
        self.resize = ops.Resize(resize_x=3, resize_y=3)

    def load(self):
        jpegs, labels = self.input()
        decoded = self.decode(jpegs)
        return self.resize(decoded)


class LoadingPipeline(CommonPipeline):
    def __init__(self):
        super(LoadingPipeline, self).__init__()

    def define_graph(self):
        im = self.load()
        im2 = self.load()
        return im, im2


class DLTensorOpPipeline(CommonPipeline):
    def __init__(self, function):
        super(DLTensorOpPipeline, self).__init__()
        self.op = ops.DLTensorPythonFunction(function=function, num_outputs=2)

    def define_graph(self):
        im = self.load()
        im2 = self.load()
        return self.op(im, im2)


def torch_adapter(fun, in1, in2):
    tin1 = [torch_dlpack.from_dlpack(dltensor) for dltensor in in1]
    tin2 = [torch_dlpack.from_dlpack(dltensor) for dltensor in in2]
    tout1, tout2 = fun(tin1, tin2)
    return [torch_dlpack.to_dlpack(tout) for tout in tout1], \
           [torch_dlpack.to_dlpack(tout) for tout in tout2]


def torch_wrapper(fun):
    return lambda in1, in2: torch_adapter(fun, in1, in2)


def torch_case(fun):
    load_pipe = LoadingPipeline()
    op_pipe = DLTensorOpPipeline(function=torch_wrapper(fun))

    load_pipe.build()
    op_pipe.build()

    for iter in range(ITERS):
        pre1, pre2 = load_pipe.run()
        post1, post2 = op_pipe.run()

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


def test_torch_simple():
    torch_case(simple_torch_op)


def test_torch_red_channel():
    torch_case(torch_red_channel_op)


def mxnet_adapter(fun, in1, in2):
    tin1 = [mxnd.from_dlpack(dltensor) for dltensor in in1]
    tin2 = [mxnd.from_dlpack(dltensor) for dltensor in in2]
    tout1, tout2 = fun(tin1, tin2)
    return [mxnd.to_dlpack_for_read(tout) for tout in tout1], \
           [mxnd.to_dlpack_for_read(tout) for tout in tout2]


def mxnet_wrapper(fun):
    return lambda in1, in2: mxnet_adapter(fun, in1, in2)


def mxnet_case(fun):
    load_pipe = LoadingPipeline()
    op_pipe = DLTensorOpPipeline(function=mxnet_wrapper(fun))

    load_pipe.build()
    op_pipe.build()

    for iter in range(ITERS):
        pre1, pre2 = load_pipe.run()
        post1, post2 = op_pipe.run()

        mxnet_pre1 = [mxnd.array(pre1.at(i)) for i in range(BATCH_SIZE)]
        mxnet_pre2 = [mxnd.array(pre2.at(i)) for i in range(BATCH_SIZE)]
        mxnet_post1, mxnet_post2 = fun(mxnet_pre1, mxnet_pre2)
        for i in range(BATCH_SIZE):
            print(post1.at(i))
            print(mxnet_post1[i].asnumpy())
            assert numpy.array_equal(post1.at(i), mxnet_post1[i].asnumpy())
            assert numpy.array_equal(post2.at(i), mxnet_post2[i].asnumpy())


def mxnet_flatten(in1, in2):
    return [mxnd.flatten(t) for t in in1], [mxnd.flatten(t) for t in in2]


def mxnet_slice(in1, in2):
    return [t[:, :, 1] for t in in1], [t[:, :, 2] for t in in2]


def mxnet_cast(in1, in2):
    return [mxnd.cast(t, dtype='float32') for t in in1], [mxnd.cast(t, dtype='int64') for t in in2]


def test_mxnet_flatten():
    mxnet_case(mxnet_flatten)


def test_mxnet_slice():
    mxnet_case(mxnet_slice)


def test_mxnet_cast():
    mxnet_case(mxnet_cast)
