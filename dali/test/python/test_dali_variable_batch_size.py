# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as dmath
import nvidia.dali.types as types
import os
import random
import re
from functools import partial
from nose_utils import SkipTest, attr, nottest
from nvidia.dali.pipeline import Pipeline, pipeline_def
from nvidia.dali.pipeline.experimental import pipeline_def as experimental_pipeline_def

import test_utils
from segmentation_test_utils import make_batch_select_masks
from test_dali_cpu_only_utils import setup_test_numpy_reader_cpu
from test_detection_pipeline import coco_anchors
from test_utils import (
    module_functions,
    has_operator,
    restrict_platform,
    check_numba_compatibility_cpu,
    check_numba_compatibility_gpu,
    is_of_supported,
)

"""
How to test variable (iter-to-iter) batch size for a given op?
-------------------------------------------------------------------------------
The idea is to create a Pipeline that assumes i2i variability, run 2 iterations
and compare them with ad-hoc created Pipelines for given (constant) batch sizes.
This can be easily done using `check_batch` function below.

On top of that, there are some utility functions and routines to help with some
common cases:
1. If the operator is typically processing image-like data (i.e. 3-dim, uint8,
   0-255, with shape like [640, 480, 3]) and you want to test default arguments
   only, please add a record to the `ops_image_default_args` list
2. If the operator is typically processing image-like data (i.e. 3-dim, uint8,
   0-255, with shape like [640, 480, 3]) and you want to specify any number of
   its arguments, please add a record to the `ops_image_custom_args` list
3. If the operator is typically processing audio-like data (i.e. 1-dim, float,
   0.-1.) please add a record to the `float_array_ops` list
4. If the operator supports sequences, please add a record to the
   `sequence_ops` list
5. If your operator case doesn't fit any of the above, please create a nosetest
   function, in which you can define a function, that returns not yet built
   pipeline, and pass it to the `check_batch` function.
6. If your operator performs random operation, this approach won't provide
   a comparable result. In this case, the best thing you can do is to check
   whether the operator works, without qualitative comparison. Use `run_pipeline`
   instead of `check_pipeline`.
"""


def generate_data(max_batch_size, n_iter, sample_shape, lo=0.0, hi=1.0, dtype=np.float32):
    """
    Generates an epoch of data, that will be used for variable batch size verification.

    :param max_batch_size: Actual sizes of every batch in the epoch will be less or equal
                           to max_batch_size
    :param n_iter: Number of iterations in the epoch
    :param sample_shape: If sample_shape is callable, shape of every sample will be determined by
                         calling sample_shape. In this case, every call to sample_shape has to
                         return a tuple of integers. If sample_shape is a tuple, this will be a
                         shape of every sample.
    :param lo: Begin of the random range
    :param hi: End of the random range
    :param dtype: Numpy data type
    :return: An epoch of data
    """
    batch_sizes = np.array([max_batch_size // 2, max_batch_size // 4, max_batch_size])

    if isinstance(sample_shape, tuple):

        def sample_shape_wrapper():
            return sample_shape

        size_fn = sample_shape_wrapper
    elif inspect.isfunction(sample_shape):
        size_fn = sample_shape
    else:
        raise RuntimeError(
            "`sample_shape` shall be either a tuple or a callable. "
            "Provide `(val,)` tuple for 1D shape"
        )

    if np.issubdtype(dtype, np.integer):
        return [
            np.random.randint(lo, hi, size=(bs,) + size_fn(), dtype=dtype) for bs in batch_sizes
        ]
    elif np.issubdtype(dtype, np.float32):
        ret = (np.random.random_sample(size=(bs,) + size_fn()) for bs in batch_sizes)
        ret = map(lambda batch: (hi - lo) * batch + lo, ret)
        ret = map(lambda batch: batch.astype(dtype), ret)
        return list(ret)
    elif np.issubdtype(dtype, bool):
        assert isinstance(lo, bool)
        assert isinstance(hi, bool)
        return [np.random.choice(a=[lo, hi], size=(bs,) + size_fn()) for bs in batch_sizes]
    else:
        raise RuntimeError(f"Invalid type argument: {dtype}")


def single_op_pipeline(
    max_batch_size,
    input_data,
    device,
    *,
    input_layout=None,
    operator_fn=None,
    needs_input=True,
    **opfn_args,
):
    pipe = Pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    with pipe:
        input = fn.external_source(
            source=input_data, cycle=False, device=device, layout=input_layout
        )
        if operator_fn is None:
            output = input
        else:
            if needs_input:
                output = operator_fn(input, **opfn_args)
            else:
                output = operator_fn(**opfn_args)
        if needs_input:
            pipe.set_outputs(output)
        else:
            # set input as an output to make sure it is not pruned from the graph
            pipe.set_outputs(output, input)
    return pipe


def get_batch_size(batch):
    """
    Returns the batch size in samples

    :param batch: List of input batches, if there is one input a batch can be either
                  a numpy array or a list, for multiple inputs it can be tuple of lists or
                  numpy arrays.
    """
    if isinstance(batch, tuple):
        return get_batch_size(batch[0])
    else:
        if isinstance(batch, list):
            return len(batch)
        else:
            return batch.shape[0]


def run_pipeline(input_epoch, pipeline_fn, *, devices: list = ["cpu", "gpu"], **pipeline_fn_args):
    """
    Verifies, if given pipeline supports iter-to-iter variable batch size

    This function verifies only if given pipeline runs without crashing.
    There is no qualitative verification. Use this for checking pipelines
    based on random operators (as they can't be verifies against one another).

    :param input_epoch: List of input batches, if there is one input a batch can be either
                        a numpy array or a list, for multiple inputs it can be tuple of lists or
                        numpy arrays.
    :param pipeline_fn: Function, that returns created (but not built) pipeline.
                        Its signature should be (at least):
                        pipeline_fn(max_batch_size, input_data, device, ...)
    :param devices: Devices to run the check on
    :param pipeline_fn_args: Additional args to pipeline_fn
    """
    for device in devices:
        n_iter = len(input_epoch)
        max_bs = max(get_batch_size(batch) for batch in input_epoch)
        var_pipe = pipeline_fn(max_bs, input_epoch, device, **pipeline_fn_args)
        for _ in range(n_iter):
            var_pipe.run()


def check_pipeline(
    input_epoch, pipeline_fn, *, devices: list = ["cpu", "gpu"], eps=1e-7, **pipeline_fn_args
):
    """
    Verifies, if given pipeline supports iter-to-iter variable batch size

    This function conducts qualitative verification. It compares the result of
    running multiple iterations of the same pipeline (with possible varying batch sizes,
    according to `input_epoch`) with results of the ad-hoc created pipelines per iteration

    :param input_epoch: List of input batches, if there is one input a batch can be either
                        a numpy array or a list, for multiple inputs it can be tuple of lists or
                        numpy arrays.
    :param pipeline_fn: Function, that returns created (but not built) pipeline.
                        Its signature should be (at least):
                        pipeline_fn(max_batch_size, input_data, device, ...)
    :param devices: Devices to run the check on
    :param eps: Epsilon for mean error
    :param pipeline_fn_args: Additional args to pipeline_fn
    """
    for device in devices:
        n_iter = len(input_epoch)
        max_bs = max(get_batch_size(batch) for batch in input_epoch)
        var_pipe = pipeline_fn(max_bs, input_epoch, device, **pipeline_fn_args)

        for iter_idx in range(n_iter):
            iter_input = input_epoch[iter_idx]
            batch_size = get_batch_size(iter_input)

            const_pipe = pipeline_fn(batch_size, [iter_input], device, **pipeline_fn_args)

            test_utils.compare_pipelines(
                var_pipe, const_pipe, batch_size=batch_size, N_iterations=1, eps=eps
            )


def image_like_shape_generator():
    return random.randint(160, 161), random.randint(80, 81), 3


def array_1d_shape_generator():
    return (random.randint(300, 400),)  # The coma is important


def custom_shape_generator(*args):
    """
    Fully configurable shape generator.
    Returns a callable which serves as a non-uniform & random shape generator to generate_epoch

    Usage:
    custom_shape_generator(dim1_lo, dim1_hi, dim2_lo, dim2_hi, etc...)
    """
    assert len(args) % 2 == 0, "Incorrect number of arguments"
    ndims = len(args) // 2
    gen_conf = [[args[2 * i], args[2 * i + 1]] for i in range(ndims)]
    return lambda: tuple([random.randint(lohi[0], lohi[1]) for lohi in gen_conf])


def image_data_helper(operator_fn, opfn_args={}):
    data = generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8)
    check_pipeline(
        data,
        pipeline_fn=single_op_pipeline,
        input_layout="HWC",
        operator_fn=operator_fn,
        **opfn_args,
    )


def float_array_helper(operator_fn, opfn_args={}):
    data = generate_data(31, 13, array_1d_shape_generator)
    check_pipeline(data, pipeline_fn=single_op_pipeline, operator_fn=operator_fn, **opfn_args)


def sequence_op_helper(operator_fn, opfn_args={}):
    data = generate_data(
        31, 13, custom_shape_generator(3, 7, 160, 200, 80, 100, 3, 3), lo=0, hi=255, dtype=np.uint8
    )
    check_pipeline(
        data,
        pipeline_fn=single_op_pipeline,
        input_layout="FHWC",
        operator_fn=operator_fn,
        **opfn_args,
    )


def random_op_helper(operator_fn, opfn_args={}):
    run_pipeline(
        generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8),
        pipeline_fn=single_op_pipeline,
        operator_fn=operator_fn,
        **opfn_args,
    )


def test_external_source():
    check_pipeline(generate_data(31, 13, custom_shape_generator(2, 4, 2, 4)), single_op_pipeline)


ops_image_default_args = [
    fn.brightness,
    fn.brightness_contrast,
    fn.cat,
    fn.color_twist,
    fn.contrast,
    fn.copy,
    fn.crop_mirror_normalize,
    fn.dump_image,
    fn.hsv,
    fn.hue,
    fn.jpeg_compression_distortion,
    fn.reductions.mean,
    fn.reductions.mean_square,
    fn.reductions.rms,
    fn.reductions.min,
    fn.reductions.max,
    fn.reductions.sum,
    fn.saturation,
    fn.shapes,
    fn.sphere,
    fn.stack,
    fn.water,
]


def test_ops_image_default_args():
    for op in ops_image_default_args:
        yield image_data_helper, op, {}


def numba_set_all_values_to_255_batch(out0, in0):
    out0[0][:] = 255


def numba_setup_out_shape(out_shape, in_shape):
    out_shape[0] = in_shape[0]


ops_image_custom_args = [
    (fn.cast, {"dtype": types.INT32}),
    (fn.color_space_conversion, {"image_type": types.BGR, "output_type": types.RGB}),
    (fn.coord_transform, {"M": 0.5, "T": 2}),
    (fn.coord_transform, {"T": 2}),
    (fn.coord_transform, {"M": 0.5}),
    (fn.crop, {"crop": (5, 5)}),
    (fn.experimental.equalize, {"devices": ["gpu"]}),
    (
        fn.erase,
        {
            "anchor": [0.3],
            "axis_names": "H",
            "normalized_anchor": True,
            "shape": [0.1],
            "normalized_shape": True,
        },
    ),
    (fn.fast_resize_crop_mirror, {"crop": [5, 5], "resize_shorter": 10, "devices": ["cpu"]}),
    (fn.flip, {"horizontal": True}),
    (fn.gaussian_blur, {"window_size": 5}),
    (fn.get_property, {"key": "layout"}),
    (fn.laplacian, {"window_size": 3}),
    (fn.laplacian, {"window_size": 3, "smoothing_size": 1}),
    (fn.laplacian, {"window_size": 3, "normalized_kernel": True}),
    (fn.normalize, {"batch": True}),
    (fn.pad, {"fill_value": -1, "axes": (0,), "shape": (10,)}),
    (fn.pad, {"fill_value": -1, "axes": (0,), "align": 16}),
    (fn.paste, {"fill_value": 69, "ratio": 1, "devices": ["gpu", "cpu"]}),
    (fn.per_frame, {"replace": True, "devices": ["cpu"]}),
    (fn.resize, {"resize_x": 50, "resize_y": 50}),
    (fn.resize_crop_mirror, {"crop": [5, 5], "resize_shorter": 10, "devices": ["cpu"]}),
    (fn.experimental.tensor_resize, {"sizes": [50, 50], "axes": [0, 1]}),
    (fn.rotate, {"angle": 25}),
    (fn.transpose, {"perm": [2, 0, 1]}),
    (fn.warp_affine, {"matrix": (0.1, 0.9, 10, 0.8, -0.2, -20)}),
    (fn.expand_dims, {"axes": 1, "new_axis_names": "Z"}),
    (fn.grid_mask, {"angle": 2.6810782, "ratio": 0.38158387, "tile": 51}),
    (fn.multi_paste, {"in_ids": np.zeros([31], dtype=np.int32), "output_size": [300, 300, 3]}),
    (fn.experimental.median_blur, {"devices": ["gpu"]}),
    (fn.experimental.dilate, {"devices": ["gpu"]}),
    (fn.experimental.erode, {"devices": ["gpu"]}),
    (fn.experimental.warp_perspective, {"matrix": np.eye(3), "devices": ["gpu", "cpu"]}),
    (fn.experimental.resize, {"resize_x": 50, "resize_y": 50, "devices": ["gpu"]}),
    (fn.zeros_like, {"devices": ["cpu"]}),
    (fn.ones_like, {"devices": ["cpu"]}),
]

numba_compatible_devices = []

if check_numba_compatibility_gpu(False):
    numba_compatible_devices.append("gpu")

if check_numba_compatibility_cpu(False):
    numba_compatible_devices.append("cpu")

if len(numba_compatible_devices) > 0 and not os.environ.get("DALI_ENABLE_SANITIZERS", None):
    from nvidia.dali.plugin.numba.fn.experimental import numba_function

    ops_image_custom_args.append(
        (
            numba_function,
            {
                "batch_processing": False,
                "devices": numba_compatible_devices,
                "in_types": [types.UINT8],
                "ins_ndim": [3],
                "out_types": [types.UINT8],
                "outs_ndim": [3],
                "blocks": [32, 32, 1],
                "threads_per_block": [32, 16, 1],
                "run_fn": numba_set_all_values_to_255_batch,
                "setup_fn": numba_setup_out_shape,
            },
        )
    )


def test_ops_image_custom_args():
    for op, args in ops_image_custom_args:
        yield image_data_helper, op, args


float_array_ops = [
    (fn.power_spectrum, {"devices": ["cpu"]}),
    (fn.preemphasis_filter, {}),
    (fn.spectrogram, {"nfft": 60, "window_length": 50, "window_step": 25}),
    (fn.to_decibels, {}),
    (fn.audio_resample, {"devices": ["cpu"], "scale": 1.2}),
]


def test_float_array_ops():
    for op, args in float_array_ops:
        yield float_array_helper, op, args


random_ops = [
    (fn.jitter, {"devices": ["gpu"]}),
    (fn.random_resized_crop, {"size": 69}),
    (fn.noise.gaussian, {}),
    (fn.noise.shot, {}),
    (fn.noise.salt_and_pepper, {}),
    (fn.segmentation.random_mask_pixel, {"devices": ["cpu"]}),
    (
        fn.roi_random_crop,
        {
            "devices": ["cpu"],
            "crop_shape": [10, 15, 3],
            "roi_start": [25, 20, 0],
            "roi_shape": [40, 30, 3],
        },
    ),
]


def test_random_ops():
    for op, args in random_ops:
        yield random_op_helper, op, args


sequence_ops = [
    (fn.cast, {"dtype": types.INT32}),
    (fn.copy, {}),
    (fn.crop, {"crop": (5, 5)}),
    (fn.crop_mirror_normalize, {"mirror": 1, "output_layout": "FCHW"}),
    (
        fn.erase,
        {
            "anchor": [0.3],
            "axis_names": "H",
            "normalized_anchor": True,
            "shape": [0.1],
            "normalized_shape": True,
        },
    ),
    (fn.flip, {"horizontal": True}),
    (fn.gaussian_blur, {"window_size": 5}),
    (fn.normalize, {"batch": True}),
    (fn.per_frame, {"devices": ["cpu"]}),
    (fn.resize, {"resize_x": 50, "resize_y": 50}),
]


def test_sequence_ops():
    for op, args in sequence_ops:
        yield sequence_op_helper, op, args


def test_batch_permute():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        perm = fn.batch_permutation(seed=420)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.permute_batch(data, indices=perm)
        pipe.set_outputs(processed)
        return pipe

    run_pipeline(generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe)


def test_coin_flip():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        depthwise = fn.random.coin_flip()
        horizontal = fn.random.coin_flip()
        vertical = fn.random.coin_flip()
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.flip(data, depthwise=depthwise, horizontal=horizontal, vertical=vertical)
        pipe.set_outputs(processed)
        return pipe

    run_pipeline(
        generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe, devices=["cpu"]
    )


def test_uniform():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        dist = fn.random.uniform()
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = data * dist
        pipe.set_outputs(processed)
        return pipe

    run_pipeline(generate_data(31, 13, array_1d_shape_generator), pipeline_fn=pipe)


def test_random_choice():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        dist = fn.random.choice(data)
        pipe.set_outputs(dist)
        return pipe

    run_pipeline(generate_data(31, 13, array_1d_shape_generator), pipeline_fn=pipe, devices=["cpu"])


def test_random_normal():
    def pipe_input(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        dist = fn.random.normal(data)
        pipe.set_outputs(dist)
        return pipe

    def pipe_no_input(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        dist = data + fn.random.normal()
        pipe.set_outputs(dist)
        return pipe

    run_pipeline(generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe_input)
    run_pipeline(generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe_no_input)


def test_random_beta():
    def pipe_input(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        dist = fn.random.beta(data)
        pipe.set_outputs(dist)
        return pipe

    def pipe_no_input(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        dist = data + fn.random.beta()
        pipe.set_outputs(dist)
        return pipe

    run_pipeline(
        generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe_input, devices=["cpu"]
    )
    run_pipeline(
        generate_data(31, 13, image_like_shape_generator),
        pipeline_fn=pipe_no_input,
        devices=["cpu"],
    )


def no_input_op_helper(operator_fn, opfn_args={}):
    data = generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8)
    check_pipeline(
        data,
        pipeline_fn=single_op_pipeline,
        input_layout="HWC",
        operator_fn=operator_fn,
        needs_input=False,
        **opfn_args,
    )


no_input_ops = [
    (fn.constant, {"fdata": 3.1415, "shape": (10, 10)}),
    (fn.transforms.translation, {"offset": (2, 3), "devices": ["cpu"]}),
    (fn.transforms.scale, {"scale": (2, 3), "devices": ["cpu"]}),
    (fn.transforms.rotation, {"angle": 30.0, "devices": ["cpu"]}),
    (fn.transforms.shear, {"shear": (2.0, 1.0), "devices": ["cpu"]}),
    (
        fn.transforms.crop,
        {
            "from_start": (0.0, 1.0),
            "from_end": (1.0, 1.0),
            "to_start": (0.2, 0.3),
            "to_end": (0.8, 0.5),
            "devices": ["cpu"],
        },
    ),
]


def test_no_input_ops():
    for op, args in no_input_ops:
        yield no_input_op_helper, op, args


def test_combine_transforms():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            # just to drive the variable batch size.
            batch_size_setter = fn.external_source(source=input_data, cycle=False, device=device)
            t = fn.transforms.translation(offset=(1, 2))
            r = fn.transforms.rotation(angle=30.0)
            s = fn.transforms.scale(scale=(2, 3))
            out = fn.transforms.combine(t, r, s)
        pipe.set_outputs(out, batch_size_setter)
        return pipe

    check_pipeline(
        generate_data(31, 13, custom_shape_generator(2, 4), lo=1, hi=255, dtype=np.uint8),
        pipeline_fn=pipe,
        devices=["cpu"],
    )


@attr("pytorch")
def test_dl_tensor_python_function():
    import torch.utils.dlpack as torch_dlpack

    def dl_tensor_operation(tensor):
        tensor = torch_dlpack.from_dlpack(tensor)
        tensor_n = tensor.double() / 255
        ret = tensor_n.sin()
        ret = torch_dlpack.to_dlpack(ret)
        return ret

    def batch_dl_tensor_operation(tensors):
        out = [dl_tensor_operation(t) for t in tensors]
        return out

    def pipe(max_batch_size, input_data, device, input_layout=None):
        pipe = Pipeline(
            batch_size=max_batch_size,
            num_threads=4,
            device_id=0,
            exec_async=False,
            exec_pipelined=False,
        )
        with pipe:
            input = fn.external_source(
                source=input_data, cycle=False, device=device, layout=input_layout
            )
            output_batch = fn.dl_tensor_python_function(
                input, function=batch_dl_tensor_operation, batch_processing=True
            )
            output_sample = fn.dl_tensor_python_function(
                input, function=dl_tensor_operation, batch_processing=False
            )
            pipe.set_outputs(output_batch, output_sample, input)
        return pipe

    check_pipeline(
        generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8),
        pipeline_fn=pipe,
        devices=["cpu"],
    )


def test_random_object_bbox():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            # just to drive the variable batch size.
            data = fn.external_source(source=input_data, batch=False, cycle="quiet", device=device)
            out = fn.segmentation.random_object_bbox(data)
        pipe.set_outputs(*out)
        return pipe

    get_data = [
        np.int32([[1, 0, 0, 0], [1, 2, 2, 1], [1, 1, 2, 0], [2, 0, 0, 1]]),
        np.int32([[0, 3, 3, 0], [1, 0, 1, 2], [0, 1, 1, 0], [0, 2, 0, 1], [0, 2, 2, 1]]),
    ]
    run_pipeline(get_data, pipeline_fn=pipe, devices=["cpu"])


def test_math_ops():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            # just to drive the variable batch size.
            data, data2 = fn.external_source(
                source=input_data, cycle=False, device=device, num_outputs=2
            )
            processed = [
                -data,
                +data,
                data * data2,
                data + data2,
                data - data2,
                data / data2,
                data // data2,
                data**data2,
                #  compare_pipelines doesn't work well with bool so promote to int by *
                (data == data2) * 1,
                (data != data2) * 1,
                (data < data2) * 1,
                (data <= data2) * 1,
                (data > data2) * 1,
                (data >= data2) * 1,
                data & data,
                data | data,
                data ^ data,
                dmath.abs(data),
                dmath.fabs(data),
                dmath.floor(data),
                dmath.ceil(data),
                dmath.pow(data, 2),
                dmath.fpow(data, 1.5),
                dmath.min(data, 2),
                dmath.max(data, 50),
                dmath.clamp(data, 10, 50),
                dmath.sqrt(data),
                dmath.rsqrt(data),
                dmath.cbrt(data),
                dmath.exp(data),
                dmath.log(data),
                dmath.log2(data),
                dmath.log10(data),
                dmath.sin(data),
                dmath.cos(data),
                dmath.tan(data),
                dmath.asin(data),
                dmath.acos(data),
                dmath.atan(data),
                dmath.atan2(data, 3),
                dmath.sinh(data),
                dmath.cosh(data),
                dmath.tanh(data),
                dmath.asinh(data),
                dmath.acosh(data),
                dmath.atanh(data),
            ]
        pipe.set_outputs(*processed)
        return pipe

    def get_data(batch_size):
        test_data_shape = [random.randint(5, 21), random.randint(5, 21), 3]
        data1 = [
            np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
            for _ in range(batch_size)
        ]
        data2 = [
            np.random.randint(1, 4, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)
        ]
        return (data1, data2)

    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe)


def test_squeeze_op():
    def pipe(max_batch_size, input_data, device, input_layout=None):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            # just to drive the variable batch size.
            data = fn.external_source(
                source=input_data, cycle=False, device=device, layout=input_layout
            )
            out = fn.expand_dims(data, axes=[0, 2], new_axis_names="YZ")
            out = fn.squeeze(out, axis_names="Z")
        pipe.set_outputs(out)
        return pipe

    check_pipeline(
        generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8),
        pipeline_fn=pipe,
        input_layout="HWC",
    )


def test_box_encoder_op():
    def pipe(max_batch_size, input_data, device, input_layout=None):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            boxes, lables = fn.external_source(device=device, source=input_data, num_outputs=2)
            processed, _ = fn.box_encoder(boxes, lables, anchors=coco_anchors())
        pipe.set_outputs(processed)
        return pipe

    def get_data(batch_size):
        obj_num = random.randint(1, 20)
        test_box_shape = [obj_num, 4]
        test_lables_shape = [obj_num, 1]
        bboxes = [
            np.random.random(size=test_box_shape).astype(dtype=np.float32)
            for _ in range(batch_size)
        ]
        labels = [
            np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32)
            for _ in range(batch_size)
        ]
        return (bboxes, labels)

    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe, devices=["cpu"])


def test_remap():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, device_id=0, num_threads=4)
        with pipe:
            input, mapx, mapy = fn.external_source(device=device, source=input_data, num_outputs=3)
            out = fn.experimental.remap(input, mapx, mapy)
        pipe.set_outputs(out)
        return pipe

    def get_data(batch_size):
        input_shape = [480, 640, 3]
        mapx_shape = mapy_shape = [480, 640]
        input = [
            np.random.randint(0, 255, size=input_shape, dtype=np.uint8) for _ in range(batch_size)
        ]
        mapx = [
            640 * np.random.random(size=mapx_shape).astype(np.float32)  # [0, 640) interval
            for _ in range(batch_size)
        ]
        mapy = [
            480 * np.random.random(size=mapy_shape).astype(np.float32)  # [0, 480) interval
            for _ in range(batch_size)
        ]
        return input, mapx, mapy

    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe, devices=["gpu"])


def test_random_bbox_crop_op():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            boxes, lables = fn.external_source(device=device, source=input_data, num_outputs=2)
            processed = fn.random_bbox_crop(
                boxes,
                lables,
                aspect_ratio=[0.5, 2.0],
                thresholds=[0.1, 0.3, 0.5],
                scaling=[0.8, 1.0],
                bbox_layout="xyXY",
            )
        pipe.set_outputs(*processed)
        return pipe

    def get_data(batch_size):
        obj_num = random.randint(1, 20)
        test_box_shape = [obj_num, 4]
        test_lables_shape = [obj_num, 1]
        bboxes = [
            np.random.random(size=test_box_shape).astype(dtype=np.float32)
            for _ in range(batch_size)
        ]
        labels = [
            np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32)
            for _ in range(batch_size)
        ]
        return (bboxes, labels)

    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    run_pipeline(input_data, pipeline_fn=pipe, devices=["cpu"])


def test_ssd_random_crop_op():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            data, boxes, lables = fn.external_source(
                device=device, source=input_data, num_outputs=3
            )
            processed = fn.ssd_random_crop(data, boxes, lables)
        pipe.set_outputs(*processed)
        return pipe

    def get_data(batch_size):
        obj_num = random.randint(1, 20)
        test_data_shape = [50, 20, 3]
        test_box_shape = [obj_num, 4]
        test_lables_shape = [obj_num]
        data = [
            np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
            for _ in range(batch_size)
        ]
        bboxes = [
            np.random.random(size=test_box_shape).astype(dtype=np.float32)
            for _ in range(batch_size)
        ]
        labels = [
            np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32)
            for _ in range(batch_size)
        ]
        return (data, bboxes, labels)

    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    run_pipeline(input_data, pipeline_fn=pipe, devices=["cpu"])


def test_reshape():
    data = generate_data(31, 13, (160, 80, 3), lo=0, hi=255, dtype=np.uint8)
    check_pipeline(
        data, pipeline_fn=single_op_pipeline, operator_fn=fn.reshape, shape=(160 / 2, 80 * 2, 3)
    )


def test_slice():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.slice(data, 0.1, 0.5, axes=0, device=device)
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(
        generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8),
        pipeline_fn=pipe,
    )


def test_bb_flip():
    check_pipeline(
        generate_data(31, 13, custom_shape_generator(150, 250, 4, 4)),
        single_op_pipeline,
        operator_fn=fn.bb_flip,
    )


def test_1_hot():
    data = generate_data(31, 13, array_1d_shape_generator, lo=0, hi=255, dtype=np.uint8)
    check_pipeline(data, single_op_pipeline, operator_fn=fn.one_hot)


def test_bbox_paste():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        paste_posx = fn.random.uniform(range=(0, 1))
        paste_posy = fn.random.uniform(range=(0, 1))
        paste_ratio = fn.random.uniform(range=(1, 2))
        processed = fn.bbox_paste(data, paste_x=paste_posx, paste_y=paste_posy, ratio=paste_ratio)
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(
        generate_data(31, 13, custom_shape_generator(150, 250, 4, 4)),
        pipe,
        eps=0.5,
        devices=["cpu"],
    )


def test_coord_flip():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.coord_flip(data)
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, custom_shape_generator(150, 250, 2, 2)), pipe)


def test_lookup_table():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.lookup_table(data, keys=[1, 3], values=[10, 50])
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(
        generate_data(31, 13, array_1d_shape_generator, lo=0, hi=5, dtype=np.uint8), pipe
    )
    # TODO sequence


def test_reduce():
    reduce_fns = [fn.reductions.std_dev, fn.reductions.variance]

    def pipe(max_batch_size, input_data, device, reduce_fn):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        mean = fn.reductions.mean(data)
        reduced = reduce_fn(data, mean)
        pipe.set_outputs(reduced)
        return pipe

    for rf in reduce_fns:
        check_pipeline(generate_data(31, 13, image_like_shape_generator), pipe, reduce_fn=rf)


def test_sequence_rearrange():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device, layout="FHWC")
        processed = fn.sequence_rearrange(data, new_order=[0, 4, 1, 3, 2])
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (5, 10, 20, 3), lo=0, hi=255, dtype=np.uint8), pipe)


def test_element_extract():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device, layout="FHWC")
        processed, _ = fn.element_extract(data, element_map=[0, 3])
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, (5, 10, 20, 3), lo=0, hi=255, dtype=np.uint8), pipe)


def test_nonsilent_region():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed, _ = fn.nonsilent_region(data)
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(
        generate_data(31, 13, array_1d_shape_generator, lo=0, hi=255, dtype=np.uint8),
        pipe,
        devices=["cpu"],
    )


def test_mel_filter_bank():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            data = fn.external_source(source=input_data, cycle=False, device=device)
            spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
            processed = fn.mel_filter_bank(spectrum)
            pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, array_1d_shape_generator), pipe)


def test_mfcc():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
        mel = fn.mel_filter_bank(spectrum)
        dec = fn.to_decibels(mel)
        processed = fn.mfcc(dec)
        pipe.set_outputs(processed)

        return pipe

    check_pipeline(generate_data(31, 13, array_1d_shape_generator), pipe)


@nottest
def generate_decoders_data(data_dir, data_extension, exclude_subdirs=[]):
    # File reader won't work, so I need to load audio files into external_source manually
    fnames = test_utils.filter_files(data_dir, data_extension, exclude_subdirs=exclude_subdirs)

    nfiles = len(fnames)
    # TODO(janton): Workaround for audio data (not enough samples)
    #               To be removed when more audio samples are added
    for i in range(len(fnames), 10):  # At least 10 elements
        fnames.append(fnames[-1])
    nfiles = len(fnames)
    _input_epoch = [
        list(map(lambda fname: test_utils.read_file_bin(fname), fnames[: nfiles // 3])),
        list(map(lambda fname: test_utils.read_file_bin(fname), fnames[nfiles // 3 : nfiles // 2])),
        list(map(lambda fname: test_utils.read_file_bin(fname), fnames[nfiles // 2 :])),
    ]

    # Since we pack buffers into ndarray, we need to pad samples with 0.
    input_epoch = []
    for inp in _input_epoch:
        max_len = max(sample.shape[0] for sample in inp)
        inp = map(lambda sample: np.pad(sample, (0, max_len - sample.shape[0])), inp)
        input_epoch.append(np.stack(list(inp)))
    input_epoch = list(map(lambda batch: np.reshape(batch, batch.shape), input_epoch))

    return input_epoch


@nottest
def test_decoders_check(pipeline_fn, data_dir, data_extension, devices=["cpu"], exclude_subdirs=[]):
    data = generate_decoders_data(
        data_dir=data_dir, data_extension=data_extension, exclude_subdirs=exclude_subdirs
    )
    check_pipeline(data, pipeline_fn=pipeline_fn, devices=devices)


@nottest
def test_decoders_run(pipeline_fn, data_dir, data_extension, devices=["cpu"], exclude_subdirs=[]):
    data = generate_decoders_data(
        data_dir=data_dir, data_extension=data_extension, exclude_subdirs=exclude_subdirs
    )
    run_pipeline(data, pipeline_fn=pipeline_fn, devices=devices)


def test_audio_decoders():
    def audio_decoder_pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device="cpu")
        decoded, _ = fn.decoders.audio(encoded, downmix=True, sample_rate=12345, device=device)
        pipe.set_outputs(decoded)
        return pipe

    audio_path = os.path.join(test_utils.get_dali_extra_path(), "db", "audio")
    yield test_decoders_check, audio_decoder_pipe, audio_path, ".wav"


def test_image_decoders():
    extra_args = {
        "hw_decoder_load": 0.0,  # for deterministic results regardless of batch size
    }

    def image_decoder_pipe(module, max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device="cpu")
        decoded = module.image(encoded, device=device, **extra_args)
        pipe.set_outputs(decoded)
        return pipe

    def image_decoder_crop_pipe(module, max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device="cpu")
        decoded = module.image_crop(encoded, device=device, **extra_args)
        pipe.set_outputs(decoded)
        return pipe

    def image_decoder_slice_pipe(module, max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device="cpu")
        decoded = module.image_slice(encoded, 0.1, 0.4, axes=0, device=device, **extra_args)
        pipe.set_outputs(decoded)
        return pipe

    def image_decoder_rcrop_pipe(module, max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device="cpu")
        decoded = module.image_random_crop(encoded, device=device, **extra_args)
        pipe.set_outputs(decoded)
        return pipe

    def peek_image_shape_pipe(module, max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device="cpu")
        shape = module.peek_image_shape(encoded, device=device)
        pipe.set_outputs(shape)
        return pipe

    image_decoder_extensions = [".jpg", ".bmp", ".png", ".pnm", ".jp2"]
    image_decoder_pipes = [
        image_decoder_pipe,
        image_decoder_crop_pipe,
        image_decoder_slice_pipe,
    ]
    data_path = os.path.join(test_utils.get_dali_extra_path(), "db", "single")
    # excluding paths that contain images that are not widely supported (by legacy and new decoders)
    exclude_subdirs = ["jpeg_lossless"]
    for ext in image_decoder_extensions:
        for pipe_template in image_decoder_pipes:
            pipe = partial(pipe_template, fn.decoders)
            yield test_decoders_check, pipe, data_path, ext, ["cpu", "mixed"], exclude_subdirs
            pipe = partial(pipe_template, fn.experimental.decoders)
            yield test_decoders_check, pipe, data_path, ext, ["cpu", "mixed"], exclude_subdirs
        pipe = partial(image_decoder_rcrop_pipe, fn.decoders)
        yield test_decoders_run, pipe, data_path, ext, ["cpu", "mixed"], exclude_subdirs
        pipe = partial(image_decoder_rcrop_pipe, fn.experimental.decoders)
        yield test_decoders_run, pipe, data_path, ext, ["cpu", "mixed"], exclude_subdirs

    pipe = partial(peek_image_shape_pipe, fn)
    yield test_decoders_check, pipe, data_path, ".jpg", ["cpu"], exclude_subdirs
    pipe = partial(peek_image_shape_pipe, fn.experimental)
    yield test_decoders_check, pipe, data_path, ".jpg", ["cpu"], exclude_subdirs


def test_numpy_decoder():
    def numpy_decoder_pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device="cpu")
        decoded = fn.decoders.numpy(encoded)
        if device == "gpu":
            decoded = decoded.gpu()
        pipe.set_outputs(decoded)
        return pipe

    with setup_test_numpy_reader_cpu() as numpy_path:
        test_decoders_check(numpy_decoder_pipe, numpy_path, ".npy")


def test_python_function():
    def resize(data):
        data += 13
        return data

    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(
            batch_size=max_batch_size,
            num_threads=4,
            device_id=0,
            exec_async=False,
            exec_pipelined=False,
        )
        with pipe:
            data = fn.external_source(source=input_data, cycle=False, device=device)
            processed = fn.python_function(data, function=resize, num_outputs=1)
            pipe.set_outputs(processed)
        return pipe

    check_pipeline(generate_data(31, 13, image_like_shape_generator), pipe, devices=["cpu"])


def test_reinterpret():
    def pipe(max_batch_size, input_data, device, input_layout):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(
            source=input_data, cycle=False, device=device, layout=input_layout
        )
        processed = fn.reinterpret(data, rel_shape=[0.5, 1, -1])
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(
        generate_data(31, 13, (160, 80, 3), lo=0, hi=255, dtype=np.uint8),
        pipeline_fn=pipe,
        input_layout="HWC",
    )
    check_pipeline(
        generate_data(31, 13, (5, 160, 80, 3), lo=0, hi=255, dtype=np.uint8),
        pipeline_fn=pipe,
        input_layout="FHWC",
    )


def test_segmentation_select_masks():
    def get_data_source(*args, **kwargs):
        return make_batch_select_masks(*args, **kwargs)

    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=None, seed=1234)
        with pipe:
            polygons, vertices, selected_masks = fn.external_source(
                num_outputs=3, device=device, source=input_data
            )
            out_polygons, out_vertices = fn.segmentation.select_masks(
                selected_masks, polygons, vertices, reindex_masks=False
            )
        pipe.set_outputs(polygons, vertices, selected_masks, out_polygons, out_vertices)
        return pipe

    input_data = [
        get_data_source(
            random.randint(5, 31), vertex_ndim=2, npolygons_range=(1, 5), nvertices_range=(3, 10)
        )
        for _ in range(13)
    ]
    check_pipeline(input_data, pipeline_fn=pipe, devices=["cpu"])


def test_optical_flow():
    if not is_of_supported():
        raise SkipTest("Optical Flow is not supported on this platform")

    def pipe(max_batch_size, input_data, device, input_layout=None):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            data = fn.external_source(
                device=device, source=input_data, cycle=False, layout=input_layout
            )
            processed = fn.optical_flow(data, device=device, output_grid=4)
        pipe.set_outputs(processed)
        return pipe

    input_data = generate_data(5, 2, (10, 480, 640, 3), lo=0, hi=255, dtype=np.uint8)
    check_pipeline(input_data, pipeline_fn=pipe, devices=["gpu"], input_layout="FHWC")


def test_tensor_subscript():
    def pipe(max_batch_size, input_data, device, input_layout):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(
            source=input_data, cycle=False, device=device, layout=input_layout
        )
        processed = data[2, :-2:, 1]
        pipe.set_outputs(processed)
        return pipe

    check_pipeline(
        generate_data(31, 13, (160, 80, 3), lo=0, hi=255, dtype=np.uint8),
        pipeline_fn=pipe,
        input_layout="HWC",
    )
    check_pipeline(
        generate_data(31, 13, (5, 160, 80, 3), lo=0, hi=255, dtype=np.uint8),
        pipeline_fn=pipe,
        input_layout="FHWC",
    )


def test_subscript_dim_check():
    data = generate_data(31, 13, array_1d_shape_generator, lo=0, hi=255, dtype=np.uint8)
    check_pipeline(data, single_op_pipeline, operator_fn=fn.subscript_dim_check, num_subscripts=1)


def test_crop_argument_from_external_source():
    """
    Tests, if the fn.crop operator works correctly, when its actual batch size is lower
    than max batch size.
    """

    @pipeline_def(batch_size=32, num_threads=4, device_id=0)
    def pipeline():
        images = fn.external_source(device="cpu", name="IMAGE", no_copy=False)
        crop_x = fn.external_source(device="cpu", name="CROP_X", no_copy=False)
        images = fn.decoders.image(images, device="mixed")

        images = fn.crop(images, crop_pos_x=crop_x, crop_pos_y=0.05, crop_w=113, crop_h=149)
        return images

    pipe = pipeline()

    image_data = np.fromfile(
        os.path.join(
            test_utils.get_dali_extra_path(), "db", "single", "jpeg", "100", "swan-3584559_640.jpg"
        ),
        dtype=np.uint8,
    )
    pipe.feed_input("IMAGE", [image_data])
    pipe.feed_input("CROP_X", [np.float32(0.5)])
    pipe.feed_input("IMAGE", [image_data])
    pipe.feed_input("CROP_X", [np.float32(0.4)])

    pipe.run()


def test_video_decoder():
    def video_decoder_pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device="cpu")
        decoded = fn.experimental.decoders.video(encoded, device=device)
        pipe.set_outputs(decoded)
        return pipe

    file_path = os.path.join(test_utils.get_dali_extra_path(), "db", "video", "cfr", "test_1.mp4")
    video_file = np.fromfile(file_path, dtype=np.uint8)
    batches = [[video_file] * 2, [video_file] * 5, [video_file] * 3]
    check_pipeline(batches, video_decoder_pipe, devices=["cpu", "mixed"])


@has_operator("experimental.inflate")
@restrict_platform(min_compute_cap=6.0)
def test_inflate():
    import lz4.block

    def sample_to_lz4(sample):
        deflated_buf = lz4.block.compress(sample, store_size=False)
        return np.frombuffer(deflated_buf, dtype=np.uint8)

    def inflate_pipline(max_batch_size, inputs, device):
        input_data = [[sample_to_lz4(sample) for sample in batch] for batch in inputs]
        input_shape = [
            [np.array(sample.shape, dtype=np.int32) for sample in batch] for batch in inputs
        ]

        @pipeline_def
        def piepline():
            defalted = fn.external_source(source=input_data)
            shape = fn.external_source(source=input_shape)
            return fn.experimental.inflate(defalted.gpu(), shape=shape)

        return piepline(batch_size=max_batch_size, num_threads=4, device_id=0)

    def sample_gen():
        j = 42
        while True:
            yield np.full((13, 7), j)
            j += 1

    sample = sample_gen()
    batches = [
        [next(sample) for _ in range(5)],
        [next(sample) for _ in range(13)],
        [next(sample) for _ in range(2)],
    ]

    check_pipeline(batches, inflate_pipline, devices=["gpu"])


def test_debayer():
    from debayer_test_utils import rgb2bayer, bayer_patterns, blue_position

    def debayer_pipline(max_batch_size, inputs, device):
        batches = [list(zip(*batch)) for batch in inputs]
        img_batches = [list(imgs) for imgs, _ in batches]
        blue_positions = [list(positions) for _, positions in batches]

        @pipeline_def
        def piepline():
            bayered = fn.external_source(source=img_batches)
            positions = fn.external_source(source=blue_positions)
            if device == "gpu":
                bayered = bayered.gpu()
            return fn.experimental.debayer(bayered, blue_position=positions)

        return piepline(batch_size=max_batch_size, num_threads=4, device_id=0)

    def sample_gen():
        rng = np.random.default_rng(seed=101)
        j = 0
        while True:
            pattern = bayer_patterns[j % len(bayer_patterns)]
            h, w = 2 * np.int32(rng.uniform(2, 3, 2))
            r, g, b = np.full((h, w), j), np.full((h, w), j + 1), np.full((h, w), j + 2)
            rgb = np.uint8(np.stack([r, g, b], axis=2))
            yield rgb2bayer(rgb, pattern), np.array(blue_position(pattern), dtype=np.int32)
            j += 1

    sample = sample_gen()
    batches = [
        [next(sample) for _ in range(5)],
        [next(sample) for _ in range(13)],
        [next(sample) for _ in range(2)],
    ]

    check_pipeline(batches, debayer_pipline, devices=["gpu", "cpu"])


def test_filter():
    def filter_pipeline(max_batch_size, inputs, device):
        batches = [list(zip(*batch)) for batch in inputs]
        sample_batches = [list(inp_batch) for inp_batch, _, _ in batches]
        filter_batches = [list(filt_batch) for _, filt_batch, _ in batches]
        fill_value_bacthes = [list(fvs) for _, _, fvs in batches]

        @pipeline_def
        def pipeline():
            samples = fn.external_source(source=sample_batches, layout="HWC")
            filters = fn.external_source(source=filter_batches)
            fill_values = fn.external_source(source=fill_value_bacthes)
            return fn.experimental.filter(samples.gpu(), filters, fill_values, border="constant")

        return pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)

    def sample_gen():
        rng = np.random.default_rng(seed=101)
        sample_shapes = [(300, 600, 3), (100, 100, 3), (500, 1024, 1), (40, 40, 20)]
        filter_shapes = [(5, 7), (3, 3), (60, 2)]
        j = 0
        while True:
            sample_shape = sample_shapes[j % len(sample_shapes)]
            filter_shape = filter_shapes[j % len(filter_shapes)]
            sample = np.uint8(rng.uniform(0, 255, sample_shape))
            filter = np.float32(rng.uniform(0, 255, filter_shape))
            yield sample, filter, np.array([rng.uniform(0, 255)], dtype=np.uint8)
            j += 1

    sample = sample_gen()
    batches = [
        [next(sample) for _ in range(5)],
        [next(sample) for _ in range(13)],
        [next(sample) for _ in range(2)],
    ]

    check_pipeline(batches, filter_pipeline, devices=["gpu"])


def test_cast_like():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data, data2 = fn.external_source(
            source=input_data, cycle=False, device=device, num_outputs=2
        )
        out = fn.cast_like(data, data2)
        pipe.set_outputs(out)
        return pipe

    def get_data(batch_size):
        test_data_shape = [random.randint(5, 21), random.randint(5, 21), 3]
        data1 = [
            np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8)
            for _ in range(batch_size)
        ]
        data2 = [
            np.random.randint(1, 4, size=test_data_shape, dtype=np.int32) for _ in range(batch_size)
        ]
        return (data1, data2)

    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe)


def test_conditional():
    def conditional_wrapper(max_batch_size, input_data, device):
        @experimental_pipeline_def(
            enable_conditionals=True, batch_size=max_batch_size, num_threads=4, device_id=0
        )
        def actual_pipe():
            variable_condition = fn.external_source(source=input_data, cycle=False, device=device)
            variable_data = variable_condition + 42.0
            if variable_condition:
                other_variable_data = variable_condition + 100
                output = variable_data + other_variable_data
            else:
                output = types.Constant(np.array(42.0), device="cpu")
            logical_expr = variable_condition or not variable_condition
            logical_expr2 = not variable_condition and variable_condition
            return output, variable_condition, variable_data, logical_expr, logical_expr2

        return actual_pipe()

    check_pipeline(
        generate_data(31, 13, custom_shape_generator(), lo=False, hi=True, dtype=np.bool_),
        pipeline_fn=conditional_wrapper,
        devices=["cpu"],
    )

    def split_merge_wrapper(max_batch_size, input_data, device):
        @experimental_pipeline_def(
            enable_conditionals=True, batch_size=max_batch_size, num_threads=4, device_id=0
        )
        def actual_pipe():
            variable_pred = fn.external_source(source=input_data, cycle=False, device=device)
            variable_data = variable_pred + 42.0
            true, false = fn._conditional.split(variable_data, predicate=variable_pred)
            true = true + 10.0
            merged = fn._conditional.merge(true, false, predicate=variable_pred)
            return merged, variable_pred

        return actual_pipe()

    check_pipeline(
        generate_data(31, 13, custom_shape_generator(), lo=False, hi=True, dtype=np.bool_),
        pipeline_fn=split_merge_wrapper,
        devices=["cpu"],
    )

    def not_validate_wrapper(max_batch_size, input_data, device):
        @experimental_pipeline_def(
            enable_conditionals=True, batch_size=max_batch_size, num_threads=4, device_id=0
        )
        def actual_pipe():
            variable_pred = fn.external_source(source=input_data, cycle=False, device=device)
            negated = fn._conditional.not_(variable_pred)
            validated = fn._conditional.validate_logical(
                variable_pred, expression_name="or", expression_side="right"
            )
            return negated, validated, variable_pred

        return actual_pipe()

    check_pipeline(
        generate_data(31, 13, custom_shape_generator(), lo=False, hi=True, dtype=np.bool_),
        pipeline_fn=not_validate_wrapper,
        devices=["cpu"],
    )


def test_random_crop_gen():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        anchor, shape = fn.random_crop_generator(data)
        pipe.set_outputs(anchor, shape)
        return pipe

    sh = (2,)
    run_pipeline(generate_data(31, 13, sh, dtype=np.int64), pipeline_fn=pipe, devices=["cpu"])


def test_zeros():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        dist = fn.zeros(shape=())
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = data * dist
        pipe.set_outputs(processed)
        return pipe

    run_pipeline(generate_data(31, 13, array_1d_shape_generator), pipeline_fn=pipe, devices=["cpu"])


def test_ones():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        dist = fn.ones(shape=())
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = data * dist
        pipe.set_outputs(processed)
        return pipe

    run_pipeline(generate_data(31, 13, array_1d_shape_generator), pipeline_fn=pipe, devices=["cpu"])


def test_full():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        dist = fn.full(np.array([1]), shape=(1,))
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = data * dist
        pipe.set_outputs(processed)
        return pipe

    run_pipeline(generate_data(31, 13, array_1d_shape_generator), pipeline_fn=pipe, devices=["cpu"])


def test_full_like():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.full_like(np.array([1]), data)
        pipe.set_outputs(processed)
        return pipe

    run_pipeline(generate_data(31, 13, array_1d_shape_generator), pipeline_fn=pipe, devices=["cpu"])


def test_io_file_read():
    def pipe(max_batch_size, input_data, device):
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device, num_outputs=1)
        out = fn.io.file.read(data)
        pipe.set_outputs(out)
        return pipe

    def get_data(batch_size):
        rel_fpaths = [
            "db/single/png/0/cat-1046544_640.png",
            "db/single/png/0/cat-111793_640.png",
            "db/single/multichannel/with_alpha/cat-111793_640-alpha.jp2",
            "db/single/jpeg2k/2/tiled-cat-300572_640.jp2",
        ]
        path_strs = [
            os.path.join(test_utils.get_dali_extra_path(), rel_fpath) for rel_fpath in rel_fpaths
        ]
        data = []
        for i in range(batch_size):
            data.append(np.frombuffer(path_strs[i % len(rel_fpaths)].encode(), dtype=np.int8))
        return (data,)

    input_data = [get_data(random.randint(3, 9)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe, devices=["cpu"])


tested_methods = [
    "_conditional.merge",
    "_conditional.split",
    "_conditional.not_",
    "_conditional.validate_logical",
    "arithmetic_generic_op",
    "audio_decoder",
    "audio_resample",
    "batch_permutation",
    "bb_flip",
    "bbox_paste",
    "box_encoder",
    "brightness",
    "brightness_contrast",
    "cast",
    "cast_like",
    "cat",
    "coin_flip",
    "color_space_conversion",
    "color_twist",
    "constant",
    "contrast",
    "coord_flip",
    "coord_transform",
    "copy",
    "crop",
    "crop_mirror_normalize",
    "decoders.audio",
    "decoders.image",
    "decoders.image_crop",
    "decoders.image_random_crop",
    "decoders.image_slice",
    "decoders.numpy",
    "dl_tensor_python_function",
    "dump_image",
    "experimental.equalize",
    "element_extract",
    "erase",
    "expand_dims",
    "experimental.debayer",
    "experimental.decoders.image",
    "experimental.decoders.image_crop",
    "experimental.decoders.image_slice",
    "experimental.decoders.image_random_crop",
    "experimental.decoders.video",
    "experimental.dilate",
    "experimental.erode",
    "experimental.filter",
    "experimental.inflate",
    "experimental.median_blur",
    "experimental.peek_image_shape",
    "experimental.remap",
    "experimental.resize",
    "experimental.warp_perspective",
    "external_source",
    "fast_resize_crop_mirror",
    "flip",
    "gaussian_blur",
    "get_property",
    "grid_mask",
    "hsv",
    "hue",
    "image_decoder",
    "image_decoder_crop",
    "image_decoder_random_crop",
    "image_decoder_slice",
    "jitter",
    "jpeg_compression_distortion",
    "laplacian",
    "lookup_table",
    "math.abs",
    "math.acos",
    "math.acosh",
    "math.asin",
    "math.asinh",
    "math.atan",
    "math.atan2",
    "math.atanh",
    "math.cbrt",
    "math.ceil",
    "math.clamp",
    "math.cos",
    "math.cosh",
    "math.exp",
    "math.fabs",
    "math.floor",
    "math.fpow",
    "math.log",
    "math.log10",
    "math.log2",
    "math.max",
    "math.min",
    "math.pow",
    "math.rsqrt",
    "math.sin",
    "math.sinh",
    "math.sqrt",
    "math.tan",
    "math.tanh",
    "mel_filter_bank",
    "mfcc",
    "noise.gaussian",
    "noise.salt_and_pepper",
    "noise.shot",
    "nonsilent_region",
    "normal_distribution",
    "normalize",
    "numba.fn.experimental.numba_function",
    "one_hot",
    "optical_flow",
    "pad",
    "paste",
    "peek_image_shape",
    "per_frame",
    "permute_batch",
    "power_spectrum",
    "preemphasis_filter",
    "python_function",
    "random.choice",
    "random.coin_flip",
    "random.normal",
    "random.uniform",
    "random.beta",
    "random_bbox_crop",
    "random_crop_generator",
    "random_resized_crop",
    "reductions.max",
    "reductions.mean",
    "reductions.mean_square",
    "reductions.min",
    "reductions.rms",
    "reductions.std_dev",
    "reductions.sum",
    "reductions.variance",
    "reinterpret",
    "reshape",
    "resize",
    "resize_crop_mirror",
    "experimental.tensor_resize",
    "roi_random_crop",
    "rotate",
    "saturation",
    "segmentation.random_mask_pixel",
    "segmentation.random_object_bbox",
    "segmentation.select_masks",
    "sequence_rearrange",
    "shapes",
    "slice",
    "spectrogram",
    "sphere",
    "squeeze",
    "ssd_random_crop",
    "stack",
    "subscript_dim_check",
    "tensor_subscript",
    "to_decibels",
    "transform_translation",
    "transforms.combine",
    "transforms.crop",
    "transforms.rotation",
    "transforms.scale",
    "transforms.shear",
    "transforms.translation",
    "transpose",
    "uniform",
    "warp_affine",
    "water",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "io.file.read",
]

excluded_methods = [
    "hidden.*",
    "_conditional.hidden.*",
    "multi_paste",  # ToDo - crashes
    "coco_reader",  # readers do not support variable batch size yet
    "sequence_reader",  # readers do not support variable batch size yet
    "numpy_reader",  # readers do not support variable batch size yet
    "file_reader",  # readers do not support variable batch size yet
    "caffe_reader",  # readers do not support variable batch size yet
    "caffe2_reader",  # readers do not support variable batch size yet
    "mxnet_reader",  # readers do not support variable batch size yet
    "tfrecord_reader",  # readers do not support variable batch size yet
    "nemo_asr_reader",  # readers do not support variable batch size yet
    "video_reader",  # readers do not support variable batch size yet
    "video_reader_resize",  # readers do not support variable batch size yet
    "readers.coco",  # readers do not support variable batch size yet
    "readers.sequence",  # readers do not support variable batch size yet
    "readers.numpy",  # readers do not support variable batch size yet
    "readers.file",  # readers do not support variable batch size yet
    "readers.caffe",  # readers do not support variable batch size yet
    "readers.caffe2",  # readers do not support variable batch size yet
    "readers.mxnet",  # readers do not support variable batch size yet
    "readers.tfrecord",  # readers do not support variable batch size yet
    "readers.nemo_asr",  # readers do not support variable batch size yet
    "readers.video",  # readers do not support variable batch size yet
    "readers.video_resize",  # readers do not support variable batch size yet
    "readers.webdataset",  # readers do not support variable batch size yet
    "experimental.inputs.video",  # Input batch_size of inputs.video is always 1 and output
    # batch_size varies and is tested in this operator's test.
    "experimental.readers.video",  # readers do not support variable batch size yet
    "experimental.audio_resample",  # Alias of audio_resample (already tested)
    "experimental.readers.fits",  # readers do not support variable batch size yet
    "plugin.video.decoder",  # plugin not yet tested
]


def test_coverage():
    methods = module_functions(
        fn, remove_prefix="nvidia.dali.fn", allowed_private_modules=["_conditional"]
    )
    methods += module_functions(dmath, remove_prefix="nvidia.dali")
    exclude = "|".join(
        [
            "(^" + x.replace(".", "\\.").replace("*", ".*").replace("?", ".") + "$)"
            for x in excluded_methods
        ]
    )
    exclude = re.compile(exclude)
    methods = [x for x in methods if not exclude.match(x)]
    # we are fine with covering more we can easily list, like numba
    assert set(methods).difference(set(tested_methods)) == set(), "Test doesn't cover:\n {}".format(
        set(methods) - set(tested_methods)
    )
