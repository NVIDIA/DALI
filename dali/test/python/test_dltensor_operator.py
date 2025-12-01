# Copyright (c) 2019, 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os
import random
from functools import partial
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import fn, pipeline_def
from nvidia.dali.python_function_plugin import current_dali_stream

test_data_root = os.environ["DALI_EXTRA_PATH"]
images_dir = os.path.join(test_data_root, "db", "single", "jpeg")


def setup_pytorch():
    global torch_dlpack
    global torch
    import torch
    import torch.utils.dlpack as torch_dlpack

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
    square_diff_kernel = cupy.ElementwiseKernel("T x, T y", "T z", "z = x*x - y*y", "square_diff")

    mix_channels_kernel = cupy.ElementwiseKernel(
        "uint8 x, uint8 y", "uint8 z", "z = (i % 3) ? x : y", "mix_channels"
    )

    gray_scale_kernel = cupy.RawKernel(
        r"""
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
    """,
        "gray_scale",
    )


def random_seed():
    return int(random.random() * (1 << 32))


DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 32
SEED = random_seed()
NUM_WORKERS = 6


class CommonPipeline(Pipeline):
    def __init__(self, device):
        super().__init__(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, seed=SEED, prefetch_queue_depth=2)
        self.input = ops.readers.File(file_root=images_dir)
        self.decode = ops.decoders.Image(
            device="mixed" if device == "gpu" else "cpu", output_type=types.RGB, hw_decoder_load=0
        )
        self.resize = ops.Resize(resize_x=400, resize_y=400, device=device)
        self.flip = ops.Flip(device=device)

    def load(self):
        jpegs, labels = self.input()
        decoded = self.decode(jpegs)
        return self.resize(decoded)


class LoadingPipeline(CommonPipeline):
    def __init__(self, device):
        super().__init__(device)

    def define_graph(self):
        im = self.load()
        im2 = self.load()
        return im, self.flip(im2)


class DLTensorOpPipeline(CommonPipeline):
    def __init__(self, function, device, synchronize=True):
        super(DLTensorOpPipeline, self).__init__(device)
        self.op = ops.DLTensorPythonFunction(
            function=function, device=device, num_outputs=2, synchronize_stream=synchronize
        )

    def define_graph(self):
        im = self.load()
        im2 = self.load()
        return self.op(im, self.flip(im2))


def pytorch_adapter(fun, in1, in2):
    with torch.cuda.stream(torch_stream):
        tin1 = [torch_dlpack.from_dlpack(dltensor) for dltensor in in1]
        tin2 = [torch_dlpack.from_dlpack(dltensor) for dltensor in in2]
        tout1, tout2 = fun(tin1, tin2)
        out1, out2 = [torch_dlpack.to_dlpack(tout) for tout in tout1], [
            torch_dlpack.to_dlpack(tout) for tout in tout2
        ]
    torch_stream.synchronize()
    return out1, out2


def pytorch_wrapper(fun):
    return lambda in1, in2: pytorch_adapter(fun, in1, in2)


def common_case(wrapped_fun, device, compare, synchronize=True):
    load_pipe = LoadingPipeline(device)
    op_pipe = DLTensorOpPipeline(wrapped_fun, device, synchronize)

    for iter in range(ITERS):
        pre1, pre2 = load_pipe.run()
        post1, post2 = op_pipe.run()

        if device == "gpu":
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
    return [fin1[i] + fin2[i] for i in range(len(fin1))], [
        fin1[i] - fin2[i] for i in range(len(fin1))
    ]


def pytorch_red_channel_op(in1, in2):
    return [t.narrow(2, 0, 1).squeeze() for t in in1], [t.narrow(2, 0, 1).squeeze() for t in in2]


def test_pytorch():
    setup_pytorch()
    for testcase in [simple_pytorch_op, pytorch_red_channel_op]:
        for device in ["cpu", "gpu"]:
            yield pytorch_case, testcase, device

    yield from _gpu_sliced_torch_suite()
    yield from _gpu_permuted_extents_torch_suite()


def mxnet_adapter(fun, in1, in2):
    tin1 = [mxnd.from_dlpack(dltensor) for dltensor in in1]
    tin2 = [mxnd.from_dlpack(dltensor) for dltensor in in2]
    tout1, tout2 = fun(tin1, tin2)
    return [mxnd.to_dlpack_for_read(tout) for tout in tout1], [
        mxnd.to_dlpack_for_read(tout) for tout in tout2
    ]


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
    return [mxnd.cast(t, dtype="float32") for t in in1], [mxnd.cast(t, dtype="int64") for t in in2]


def test_mxnet():
    for testcase in [mxnet_flatten, mxnet_slice, mxnet_cast]:
        for device in ["cpu", "gpu"]:
            yield mxnet_case, testcase, device


def cupy_adapter_sync(fun, in1, in2):
    with cupy_stream:
        tin1 = [cupy.fromDlpack(dltensor) for dltensor in in1]
        tin2 = [cupy.fromDlpack(dltensor) for dltensor in in2]
        tout1, tout2 = fun(tin1, tin2)
        out1, out2 = [tout.toDlpack() for tout in tout1], [tout.toDlpack() for tout in tout2]
    cupy_stream.synchronize()
    return out1, out2


def cupy_adapter(fun, in1, in2):
    tin1 = [cupy.fromDlpack(dltensor) for dltensor in in1]
    tin2 = [cupy.fromDlpack(dltensor) for dltensor in in2]
    tout1, tout2 = fun(tin1, tin2)
    return [tout.toDlpack() for tout in tout1], [tout.toDlpack() for tout in tout2]


def cupy_wrapper(fun, synchronize):
    if synchronize:
        return lambda in1, in2: cupy_adapter_sync(fun, in1, in2)
    else:
        return lambda in1, in2: cupy_adapter(fun, in1, in2)


def cupy_compare(fun, synchronize, pre1, pre2, post1, post2):
    cupy_pre1 = [cupy.asarray(pre1.at(i)) for i in range(BATCH_SIZE)]
    cupy_pre2 = [cupy.asarray(pre2.at(i)) for i in range(BATCH_SIZE)]
    if synchronize:
        cupy_post1, cupy_post2 = fun(cupy_pre1, cupy_pre2)
    else:
        stream = cupy.cuda.Stream()
        cupy_post1, cupy_post2 = fun(cupy_pre1, cupy_pre2, stream=stream)
        stream.synchronize()
    for i in range(BATCH_SIZE):
        assert post1.at(i).shape == cupy_post1[i].shape
        assert post2.at(i).shape == cupy_post2[i].shape
        assert numpy.array_equal(post1.at(i), cupy.asnumpy(cupy_post1[i]))
        assert numpy.array_equal(post2.at(i), cupy.asnumpy(cupy_post2[i]))


def cupy_case(fun, synchronize=True):
    common_case(
        cupy_wrapper(fun, synchronize), "gpu", partial(cupy_compare, fun, synchronize), synchronize
    )


def cupy_simple(in1, in2):
    fin1 = [arr.astype(cupy.float32) for arr in in1]
    fin2 = [arr.astype(cupy.float32) for arr in in2]
    return [cupy.sin(fin1[i] * fin2[i]).astype(cupy.float32) for i in range(BATCH_SIZE)], [
        cupy.cos(fin1[i] * fin2[i]).astype(cupy.float32) for i in range(BATCH_SIZE)
    ]


def gray_scale_call(input):
    height = input.shape[0]
    width = input.shape[1]
    output = cupy.ndarray((height, width), dtype=cupy.float32)
    gray_scale_kernel(
        grid=((height + 31) // 32, (width + 31) // 32),
        block=(32, 32),
        stream=cupy.cuda.get_current_stream(),
        args=(output, input, height, width),
    )
    return output


def cupy_kernel_square_diff(in1, in2):
    fin1 = [arr.astype(cupy.float32) for arr in in1]
    fin2 = [arr.astype(cupy.float32) for arr in in2]
    out1, out2 = [square_diff_kernel(fin1[i], fin2[i]) for i in range(BATCH_SIZE)], in2
    return out1, out2


def cupy_kernel_mix_channels(in1, in2):
    return [mix_channels_kernel(in1[i], in2[i]) for i in range(BATCH_SIZE)], in2


def cupy_kernel_gray_scale(in1, in2, stream=None):
    if stream is None:
        stream = ops.PythonFunction.current_stream()
    s = cupy.cuda.Stream()
    s.ptr = stream.ptr
    with s:
        out1 = [gray_scale_call(arr) for arr in in1]
        out2 = [gray_scale_call(arr) for arr in in2]
    s.ptr = 0
    return out1, out2


def test_cupy():
    setup_cupy()
    print(cupy)
    for testcase in [cupy_simple, cupy_kernel_square_diff, cupy_kernel_mix_channels]:
        yield cupy_case, testcase
    yield from _cupy_flip_with_negative_strides_suite()


def test_cupy_kernel_gray_scale():
    setup_cupy()
    cupy_case(cupy_kernel_gray_scale, synchronize=False)


# ---------------- test strided copy kernel with strided tensors -----------------


def get_random_torch_batch(g, shapes, dtype):
    is_fp = torch.is_floating_point(torch.tensor([], dtype=dtype))
    if is_fp:
        return [torch.rand((shape), generator=g, dtype=dtype) for shape in shapes]
    else:
        iinfo = torch.iinfo(dtype)
        dtype_min, dtype_max = iinfo.min, iinfo.max
        return [
            torch.randint(dtype_min, dtype_max, shape, generator=g, dtype=dtype) for shape in shapes
        ]


def get_sliced_torch_case(case_name):
    # [(extents of the original shape), (slice of the corresponding extent)]
    # the original extents and slice shapes are purposely all prime numbers
    # to test handling of unaligned tensors
    prime_images = [
        ((107, 181, 3), (slice(1, 102), slice(179), slice(None))),
        ((1097, 227, 5), (slice(None), slice(None), slice(1, 4))),
        ((107, 167, 1), (slice(1, 14), slice(None), slice(None))),
        ((107, 23, 3), (slice(103), slice(None), slice(None))),
        ((173, 23, 5), (slice(None), slice(None), slice(1, 1))),
        ((401, 167, 5), (slice(4, 167), slice(None), slice(0, 3))),
        ((181, 401, 5), (slice(2, None), slice(397), slice(None))),
        ((181, 107, 1), (slice(179), slice(103), slice(1))),
        ((373, 181, 5), (slice(None), slice(None), slice(None, None, 2))),
        ((199, 401, 3), (slice(None), slice(None), slice(None))),
        ((167, 1097, 1), (slice(8, None, 7), slice(24, None, 23), slice(None))),
        ((181, 61, 1), (slice(179), slice(58, None), slice(None))),
        ((401, 61, 1), (slice(397), slice(None), slice(None))),
        ((373, 173, 1), (slice(None), slice(167), slice(None))),
        ((173, 199, 3), (slice(None), slice(None), slice(2, 3))),
        ((181, 1097, 1), (slice(2, None, None), slice(1093), slice(None))),
    ]

    prime_grey_images = [
        ((199, 23), (slice(None, 173, None), slice(None, 19, None))),
        ((373, 373), (slice(None, 331, None), slice(42, None, None))),
        ((1097, 181), (slice(114, None, None), slice(None, 157, None))),
        ((61, 227), (slice(None, 53, None), slice(28, None, None))),
        ((1097, 61), (slice(114, None, None), slice(None, 53, None))),
        ((181, 199), (slice(None, 157, None), slice(None, 173, None))),
        ((1097, 1097), (slice(114, None, None), slice(None, 983, None))),
        ((373, 227), (slice(42, None, None), slice(None, 199, None))),
        ((227, 173), (slice(None, 199, None), slice(None, 151, None))),
        ((227, 173), (slice(None, 199, None), slice(22, None, None))),
        ((401, 173), (slice(42, None, None), slice(None, 151, None))),
        ((107, 23), (slice(18, None, None), slice(None, 19, None))),
        ((23, 199), (slice(4, None, None), slice(26, None, None))),
        ((199, 23), (slice(26, None, None), slice(4, None, None))),
        ((227, 23), (slice(None, 199, None), slice(None, 19, None))),
        ((23, 23), (slice(4, None, None), slice(4, None, None))),
        ((167, 181), (slice(18, None, None), slice(24, None, None))),
        ((167, 181), (slice(18, None, None), slice(24, None, None))),
        ((181, 227), (slice(None, 157, None), slice(None, 199, None))),
        ((401, 199), (slice(None, 359, None), slice(None, 173, None))),
        ((107, 181), (slice(None, 89, None), slice(None, 157, None))),
        ((173, 61), (slice(None, 151, None), slice(8, None, None))),
        ((227, 167), (slice(None, 199, None), slice(18, None, None))),
        ((173, 401), (slice(22, None, None), slice(None, 359, None))),
        ((23, 227), (slice(4, None, None), slice(28, None, None))),
        ((227, 23), (slice(28, None, None), slice(4, None, None))),
        ((373, 373), (slice(42, None, None), slice(None, 331, None))),
        ((61, 107), (slice(None, 53, None), slice(18, None, None))),
        ((181, 61), (slice(24, None, None), slice(None, 53, None))),
        ((107, 181), (slice(None, 89, None), slice(24, None, None))),
        ((401, 23), (slice(42, None, None), slice(4, None, None))),
        ((373, 401), (slice(None, 331, None), slice(42, None, None))),
    ]

    vid = [((17,) + shape, (slice(None),) + sl) for shape, sl in prime_images]

    ndim_11 = [
        (tuple(3 if i == j else 1 for j in range(11)) + shape, ((slice(None),) * 11) + sl)
        for i, (shape, sl) in enumerate(prime_images)
    ]

    cases = {
        "slice_images": prime_images,
        "slice_grey_images": prime_grey_images,
        "slice_vid": vid,
        "slice_ndim_11": ndim_11,
    }
    shape_slices = cases[case_name]
    shapes, slices = tuple(zip(*shape_slices))
    assert len(shapes) == len(slices) == len(shape_slices)
    return shapes, slices


def _gpu_sliced_torch_case(case_name, dtype, g):
    shapes, slices = get_sliced_torch_case(case_name)
    input_batch = get_random_torch_batch(g, shapes, dtype)
    assert len(input_batch) == len(shapes)

    # returns sliced view of the input tensors
    def sliced_tensor(batch):
        stream = current_dali_stream()
        torch_stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(torch_stream):
            tensors = [torch_dlpack.from_dlpack(t) for t in batch]
            assert len(tensors) == len(slices)
            tensor_views = [t[sl] for t, sl in zip(tensors, slices)]
            out = [torch_dlpack.to_dlpack(t) for t in tensor_views]
            return out

    @pipeline_def(batch_size=len(input_batch), num_threads=4, device_id=0)
    def pipeline():
        data = fn.external_source(lambda: input_batch)
        data = fn.dl_tensor_python_function(
            data.gpu(), batch_processing=True, function=sliced_tensor, synchronize_stream=False
        )
        return data

    p = pipeline()
    (out,) = p.run()

    out = [numpy.array(sample) for sample in out.as_cpu()]
    ref = [numpy.array(sample)[sl] for sample, sl in zip(input_batch, slices)]

    numpy.testing.assert_equal(out, ref)


def _gpu_sliced_torch_suite():
    g = torch.Generator()
    g.manual_seed(42)

    for case_name in ("slice_images", "slice_grey_images", "slice_vid", "slice_ndim_11"):
        for dtype in (torch.uint8, torch.int16, torch.float32, torch.float64):
            yield _gpu_sliced_torch_case, case_name, dtype, g


def get_permute_extents_case(case_name):
    rng = random.Random(44)

    def permuted(it):
        copy = list(it)
        rng.shuffle(copy)
        return tuple(copy)

    def permuted_extents(ndim):
        extents = list(range(ndim))
        rng.shuffle(extents)
        return tuple(extents)

    # the original extents are purposely all prime numbers
    # to test handling of unaligned tensors
    prime_images = [
        (199, 181, 3),
        (1097, 61, 5),
        (373, 373, 1),
        (107, 23, 3),
        (173, 23, 5),
        (401, 167, 5),
        (181, 401, 5),
        (181, 107, 1),
        (373, 181, 5),
        (199, 401, 3),
        (1097, 1097, 1),
        (181, 61, 1),
        (401, 61, 1),
        (373, 173, 1),
        (227, 199, 3),
        (181, 1097, 1),
    ]

    if case_name == "transpose_channels_image":
        prime_images_transposed_channel = list(zip(prime_images, [(2, 0, 1)] * len(prime_images)))
        assert len(prime_images_transposed_channel) == len(prime_images)
        return prime_images_transposed_channel

    if case_name == "transpose_hw_image":
        prime_images_transposed_hw = list(zip(prime_images, [(1, 0, 2)] * len(prime_images)))
        assert len(prime_images_transposed_hw) == len(prime_images)
        return prime_images_transposed_hw

    if case_name == "image_random_permutation":
        prime_images_rnd_permuted = list(
            zip(prime_images, [permuted_extents(3) for _ in range(len(prime_images))])
        )
        assert len(prime_images_rnd_permuted) == len(prime_images)
        return prime_images_rnd_permuted

    if case_name == "transpose_channels_video":
        prime_vid_like = [
            (13, 199, 181, 3),
            (3, 1097, 61, 5),
            (17, 373, 373, 1),
            (5, 107, 23, 3),
            (11, 173, 23, 5),
            (11, 401, 167, 5),
            (7, 181, 401, 5),
            (5, 181, 107, 1),
            (3, 373, 181, 5),
            (23, 199, 401, 3),
            (3, 1097, 1097, 1),
            (31, 181, 61, 1),
            (17, 401, 61, 1),
            (5, 373, 173, 1),
            (3, 227, 199, 3),
            (7, 181, 1097, 1),
        ]

        prime_vid_like_transposed_channel = list(
            zip(prime_vid_like, [(3, 0, 1, 2)] * len(prime_vid_like))
        )
        assert len(prime_vid_like_transposed_channel) == len(prime_vid_like)
        return prime_vid_like_transposed_channel

    if case_name == "ndim_6_permute_outermost_3":
        # optimization to early stop translation of flat output index to flat input index
        # should kick in, test if that's fine
        ndim_6_transpose_outermost = [
            (permuted([3, 5, 7, 11, 13, 17]), permuted_extents(3) + (3, 4, 5)) for _ in range(5)
        ]
        assert len(ndim_6_transpose_outermost) == 5
        return ndim_6_transpose_outermost

    if case_name == "ndim_6_permute_all":
        ndim_6_rnd_permuted = [
            (permuted([3, 5, 7, 11, 13, 17]), permuted_extents(6)) for _ in range(32)
        ]
        assert len(ndim_6_rnd_permuted) == 32
        return ndim_6_rnd_permuted

    if case_name == "ndim_15_permute_all":
        # max ndim supported
        ndim_15_rnd_permuted = [
            (permuted([3, 5, 7, 11, 13, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1]), permuted_extents(15))
            for _ in range(32)
        ]
        assert len(ndim_15_rnd_permuted) == 32
        return ndim_15_rnd_permuted


def _gpu_permuted_extents_torch_case(case_name, dtype, g):
    shapes_perms = get_permute_extents_case(case_name)
    shapes, perms = tuple(zip(*shapes_perms))
    assert len(shapes) == len(perms) == len(shapes_perms)
    input_batch = get_random_torch_batch(g, shapes, dtype)
    assert len(input_batch) == len(shapes)

    # returns permuted view of the input tensors
    def permuted_tensors(batch):
        stream = current_dali_stream()
        torch_stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(torch_stream):
            tensors = [torch_dlpack.from_dlpack(t) for t in batch]
            assert len(tensors) == len(perms)
            tensor_views = [t.permute(perm) for t, perm in zip(tensors, perms)]
            out = [torch_dlpack.to_dlpack(t) for t in tensor_views]
            return out

    @pipeline_def(batch_size=len(input_batch), num_threads=4, device_id=0)
    def pipeline():
        data = fn.external_source(lambda: input_batch)
        data = fn.dl_tensor_python_function(
            data.gpu(), batch_processing=True, function=permuted_tensors, synchronize_stream=False
        )
        return data

    p = pipeline()
    (out,) = p.run()

    out = [numpy.array(sample) for sample in out.as_cpu()]
    ref = [numpy.array(sample).transpose(perm) for sample, perm in zip(input_batch, perms)]

    numpy.testing.assert_equal(out, ref)


def _gpu_permuted_extents_torch_suite():
    g = torch.Generator()
    g.manual_seed(44)

    for case_name in (
        "transpose_channels_image",
        "transpose_hw_image",
        "image_random_permutation",
        "transpose_channels_video",
        "ndim_6_permute_outermost_3",
        "ndim_6_permute_all",
        "ndim_15_permute_all",
    ):
        for dtype in (torch.uint8, torch.int16, torch.int32, torch.float64):
            yield _gpu_permuted_extents_torch_case, case_name, dtype, g


def _cupy_negative_strides_case(dtype, batch_size, steps):
    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0, seed=42)
    def baseline_pipeline():
        img, _ = fn.readers.file(name="Reader", file_root=images_dir, random_shuffle=True, seed=42)
        img = fn.decoders.image(img, device="mixed")
        img = fn.cast(img, dtype=dtype)
        img = img[tuple(slice(None, None, step) for step in steps)]
        return img

    def flip_cupy(dlps):
        stream = current_dali_stream()
        cp_stream = cupy.cuda.ExternalStream(stream, device_id=0)
        with cp_stream:
            imgs = [cupy.from_dlpack(dlp) for dlp in dlps]
            imgs = [img[tuple(slice(None, None, step) for step in steps)] for img in imgs]
            imgs = [img.toDlpack() for img in imgs]
        return imgs

    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0, seed=42)
    def pipeline():
        img, _ = fn.readers.file(name="Reader", file_root=images_dir, random_shuffle=True, seed=42)
        img = fn.decoders.image(img, device="mixed")
        img = fn.cast(img, dtype=dtype)
        img = fn.dl_tensor_python_function(
            img, batch_processing=True, function=flip_cupy, synchronize_stream=False
        )
        return img

    p = pipeline()
    baseline = baseline_pipeline()

    for _ in range(5):
        (batch,) = p.run()
        (baseline_batch,) = baseline.run()
        batch = [numpy.array(sample) for sample in batch.as_cpu()]
        baseline_batch = [numpy.array(sample) for sample in baseline_batch.as_cpu()]
        assert len(batch) == len(baseline_batch) == batch_size
        for sample, baseline_sample in zip(batch, baseline_batch):
            numpy.testing.assert_equal(sample, baseline_sample)


def _cupy_flip_with_negative_strides_suite():
    for dtype, batch_size, steps in [
        (types.DALIDataType.UINT8, 4, (-1, -1, None)),
        (types.DALIDataType.UINT8, 16, (-1, None, None)),
        (types.DALIDataType.UINT8, 2, (None, None, -1)),
        (types.DALIDataType.UINT8, 5, (-1, -1, -1)),
        (types.DALIDataType.UINT8, 16, (-2, -2, None)),
        (types.DALIDataType.UINT16, 11, (None, -1, None)),
        (types.DALIDataType.FLOAT, 16, (2, -2, None)),
        (types.DALIDataType.INT32, 12, (-2, None, None)),
        (types.DALIDataType.FLOAT64, 11, (-2, 4, -1)),
    ]:
        yield _cupy_negative_strides_case, dtype, batch_size, steps


def verify_pipeline(pipeline, input):
    assert pipeline is Pipeline.current()
    return input


def test_current_pipeline():
    pipe1 = Pipeline(13, 4, 0)
    with pipe1:
        dummy = types.Constant(numpy.ones((1)))
        output = fn.dl_tensor_python_function(
            dummy, function=lambda inp: verify_pipeline(pipe1, inp)
        )
        pipe1.set_outputs(output)

    pipe2 = Pipeline(6, 2, 0)
    with pipe2:
        dummy = types.Constant(numpy.ones((1)))
        output = fn.dl_tensor_python_function(
            dummy, function=lambda inp: verify_pipeline(pipe2, inp)
        )
        pipe2.set_outputs(output)

    pipe1.run()
    pipe2.run()
