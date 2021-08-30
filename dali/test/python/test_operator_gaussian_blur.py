# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.types as types
import nvidia.dali.fn as fn

import numpy as np
import cv2
from scipy.ndimage import convolve1d
import os
from nose.tools import raises
from nose.plugins.attrib import attr

from test_utils import get_dali_extra_path, check_batch, compare_pipelines, RandomlyShapedDataIterator, dali_type

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

test_iters = 4

shape_layout_axes_cases = [((20, 20, 30, 3), "DHWC", 3), ((20, 20, 30), "", 3),
                           ((20, 30, 3), "HWC", 2), ((20, 30), "HW", 2),
                           ((3, 30, 20), "CWH", 2), ((5, 20, 30, 3), "FHWC", 2),
                           ((5, 10, 10, 7, 3), "FDHWC", 3), ((5, 3, 20, 30), "FCHW", 2),
                           ((3, 5, 10, 10, 7), "CFDHW", 3)]

def to_batch(tl, batch_size):
    return [np.array(tl[i]) for i in range(batch_size)]


def to_cv_sigma(sigma, axes=2):
    if sigma is None:
        return (0,) * axes
    elif isinstance(sigma, (int, float)):
        return (sigma,) * axes
    elif (isinstance(sigma, np.ndarray) and len(sigma.shape) == 0):
        return (float(sigma),) * axes
    elif len(sigma) == 1:
        return (sigma[0],) * axes
    return tuple(reversed(sigma))


def to_cv_win_size(window_size, axes=2, sigma=None):
    if window_size is None:
        # when using cv2.getGaussianKernel we need to always provide window size
        if sigma is not None:
            sigma = to_cv_sigma(sigma, axes)
            return tuple([int(3 * s + 0.5) * 2 + 1 for s in sigma])
        return (0,) * axes
    elif isinstance(window_size, int):
        return (int(window_size),) * axes
    elif (isinstance(window_size, np.ndarray) and len(window_size.shape) == 0):
        return (int(window_size),) * axes
    elif len(window_size) == 1:
        return (int(window_size[0]),) * axes
    # OpenCV shape is the other way round: (width, height)
    return tuple(int(x) for x in reversed(window_size))


def gaussian_cv(image, sigma, window_size):
    sigma_x, sigma_y = to_cv_sigma(sigma)
    window_size_cv = to_cv_win_size(window_size)
    # compute on floats and round like a sane person (in mathematically complicit way)
    blurred = cv2.GaussianBlur(np.float32(image), window_size_cv, sigmaX=sigma_x, sigmaY=sigma_y)
    return np.uint8(blurred + 0.5)


def gaussian_baseline(image, sigma, window_size, axes=2, skip_axes=0, dtype=np.uint8):
    sigma_xyz = to_cv_sigma(sigma, axes)
    win_xyz = to_cv_win_size(window_size, axes, sigma)
    filters = [cv2.getGaussianKernel(win_xyz[i], sigma_xyz[i]) for i in range(axes)]
    filters = [np.float32(f).squeeze() for f in filters]
    filters.reverse()
    for i in reversed(range(axes)):
        axis = i + skip_axes
        if image.shape[axis] == 1:
            mode = "nearest"
        else:
            mode = "mirror"
        image = convolve1d(np.float32(image), filters[i], axis, mode=mode)
    if dtype == np.float32:
        return image
    else:
        return dtype(image + 0.5)


def get_gaussian_pipe(batch_size, sigma, window_size, op_type):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    with pipe:
        input, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
        decoded = fn.decoders.image(input, device="cpu", output_type=types.RGB)
        if op_type == "gpu":
            decoded = decoded.gpu()
        blurred = fn.gaussian_blur(decoded, device=op_type, sigma=sigma, window_size=window_size)
        pipe.set_outputs(blurred, decoded)
    return pipe


def check_gaussian_blur(batch_size, sigma, window_size, op_type="cpu"):
    pipe = get_gaussian_pipe(batch_size, sigma, window_size, op_type)
    pipe.build()
    for _ in range(test_iters):
        result, input = pipe.run()
        if op_type == "gpu":
            result = result.as_cpu()
            input = input.as_cpu()
        input = to_batch(input, batch_size)
        baseline_cv = [gaussian_cv(img, sigma, window_size) for img in input]
        check_batch(result, baseline_cv, batch_size, max_allowed_error=1, expected_layout="HWC")


def test_image_gaussian_blur():
    for dev in ["cpu", "gpu"]:
        for sigma in [1.0]:
            for window_size in [3, 5, None]:
                if sigma is None and window_size is None:
                    continue
                yield check_gaussian_blur, 10, sigma, window_size, dev
        # OpenCv uses fixed values for small windows that are different that Gaussian funcion
        yield check_gaussian_blur, 10, None, 11, dev


@attr('slow')
def test_image_gaussian_blur_slow():
    for dev in ["cpu", "gpu"]:
        for sigma in [1.0, [1.0, 2.0]]:
            for window_size in [3, 5, [7, 5], [5, 9], None]:
                if sigma is None and window_size is None:
                    continue
                yield check_gaussian_blur, 10, sigma, window_size, dev
        # OpenCv uses fixed values for small windows that are different that Gaussian funcion
        for window_size in [15, [17, 31]]:
            yield check_gaussian_blur, 10, None, window_size, dev


def check_gaussian_blur_cpu_gpu(batch_size, sigma, window_size):
    cpu_pipe = get_gaussian_pipe(batch_size, sigma, window_size, "cpu")
    gpu_pipe = get_gaussian_pipe(batch_size, sigma, window_size, "gpu")
    compare_pipelines(cpu_pipe, gpu_pipe, batch_size, 16, max_allowed_error=1)


def test_gaussian_blur_cpu_gpu():
    for window_size in [5, [7, 13]]:
        yield check_gaussian_blur_cpu_gpu, 10, None, window_size

@attr('slow')
def test_gaussian_blur_cpu_gpu_slow():
    for sigma in [1.0, [1.0, 2.0], None]:
        for window_size in [3, 5, [7, 5], [5, 9], 11, 15, 31, None]:
            if sigma is None and window_size is None:
                continue
            yield check_gaussian_blur_cpu_gpu, 10, sigma, window_size


def count_skip_axes(layout):
    if layout.startswith("FC") or layout.startswith("CF"):
        return 2
    elif layout.startswith("F") or layout.startswith("C"):
        return 1
    else:
        return 0


def check_generic_gaussian_blur(
        batch_size, sigma, window_size, shape, layout, axes, op_type="cpu", in_dtype=np.uint8,
        out_dtype=types.NO_TYPE, random_shape=True):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    min_shape = None if random_shape else shape
    data = RandomlyShapedDataIterator(batch_size, min_shape=min_shape, max_shape=shape, dtype=in_dtype)
    # Extract the numpy type from DALI, we can have float32 or the same as input
    if out_dtype == types.NO_TYPE:
        result_type = in_dtype
    elif dali_type(in_dtype) == out_dtype:
        result_type = in_dtype
    else:
        result_type = np.float32
    with pipe:
        input = fn.external_source(data, layout=layout)
        if op_type == "gpu":
            input = input.gpu()
        blurred = fn.gaussian_blur(input, device=op_type, sigma=sigma,
                                   window_size=window_size, dtype=out_dtype)
        pipe.set_outputs(blurred, input)
    pipe.build()

    for _ in range(test_iters):
        result, input = pipe.run()
        if op_type == "gpu":
            result = result.as_cpu()
            input = input.as_cpu()
        input = to_batch(input, batch_size)
        skip_axes = count_skip_axes(layout)
        baseline = [
            gaussian_baseline(img, sigma, window_size, axes, skip_axes, dtype=result_type)
            for img in input]
        max_error = 1 if result_type != np.float32 else 1e-04
        check_batch(result, baseline, batch_size, max_allowed_error=max_error, expected_layout=layout)


# Generate tests for single or per-axis sigma and window_size arguments
def generate_generic_cases(dev, t_in, t_out):
    for shape, layout, axes in shape_layout_axes_cases:
        for sigma in [1.0, [1.0, 2.0, 3.0]]:
            for window_size in [3, 5, [7, 5, 9], [3, 5, 9], None]:
                if isinstance(sigma, list):
                    sigma = sigma[0:axes]
                if isinstance(window_size, list):
                    window_size = window_size[0:axes]
                yield check_generic_gaussian_blur, 10, sigma, window_size, shape, layout, axes, dev, t_in, t_out
    for window_size in [11, 15]:
        yield check_generic_gaussian_blur, 10, None, window_size, shape, layout, axes, dev, t_in, t_out


def test_generic_gaussian_blur():
    for dev in ["cpu", "gpu"]:
        for (t_in, t_out) in [(np.uint8, types.NO_TYPE), (np.float32, types.FLOAT), (np.uint8, types.FLOAT)]:
            yield from generate_generic_cases(dev, t_in, t_out)

def test_one_sized_extent():
    for dev in ["cpu", "gpu"]:
        for shape, layout in [((1, 10, 6), "DHW"), ((10, 1, 3), "HWC"), ((1, 10, 3), "HWC"),
                              ((1, 10), "HW"), ((10, 1), "HW")]:
            axes = len(layout) - ("C" in layout)
            yield check_generic_gaussian_blur, 10, 2.0, 5, shape, layout, axes, dev, np.float32, types.FLOAT, False


@attr('slow')
def test_generic_gaussian_blur_slow():
    for dev in ["cpu", "gpu"]:
        for t_in in [np.uint8, np.int32, np.float32]:
            for t_out in [types.NO_TYPE, types.FLOAT, dali_type(t_in)]:
                yield from generate_generic_cases(dev, t_in, t_out)


def check_per_sample_gaussian_blur(
        batch_size, sigma_dim, window_size_dim, shape, layout, axes, op_type="cpu"):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    data = RandomlyShapedDataIterator(batch_size, max_shape=shape)
    with pipe:
        if sigma_dim is not None:
            sigma = fn.random.uniform(range=[0.5, 3], shape=[sigma_dim])
            sigma_arg = sigma
        else:
            # placeholder, so we can return something
            sigma = fn.random.coin_flip(probability=0)
            sigma_arg = None

        if window_size_dim is not None:
            window_radius = fn.random.uniform(range=[5, 10], shape=[window_size_dim])
            window_size = fn.cast(window_radius, dtype=types.INT32) * 2 + 1
            window_arg = window_size
        else:
            window_size = fn.random.coin_flip(probability=0)
            window_arg = None

        input = fn.external_source(data, layout=layout)
        if op_type == "gpu":
            input = input.gpu()
        blurred = fn.gaussian_blur(input, device=op_type, sigma=sigma_arg, window_size=window_arg)
        pipe.set_outputs(blurred, input, sigma, window_size)
    pipe.build()

    for _ in range(test_iters):
        result, input, sigma, window_size = pipe.run()
        if op_type == "gpu":
            result = result.as_cpu()
            input = input.as_cpu()
        input = to_batch(input, batch_size)
        sigma = to_batch(sigma, batch_size)
        window_size = to_batch(window_size, batch_size)
        baseline = []
        for i in range(batch_size):
            sigma_arg = sigma[i] if sigma is not None else None
            window_arg = window_size[i] if window_size_dim is not None else None
            skip_axes = count_skip_axes(layout)
            baseline.append(gaussian_baseline(input[i], sigma_arg, window_arg, axes, skip_axes))
        check_batch(result, baseline, batch_size, max_allowed_error=1, expected_layout=layout)


# TODO(klecki): consider checking mixed ArgumentInput/Scalar value cases
def test_per_sample_gaussian_blur():
    for dev in ["cpu", "gpu"]:
        for shape, layout, axes in shape_layout_axes_cases:
            for sigma_dim in [None, 1, axes]:
                for window_size_dim in [None, 1, axes]:
                    if sigma_dim is None and window_size_dim is None:
                        continue
                    yield check_per_sample_gaussian_blur, 10, sigma_dim, window_size_dim, shape, layout, axes, dev


@raises(RuntimeError)
def check_fail_gaussian_blur(batch_size, sigma, window_size, shape, layout, axes, op_type, in_dtype=np.uint8, out_dtype=types.NO_TYPE):
    check_generic_gaussian_blur(batch_size, sigma, window_size, shape, layout, axes, op_type, in_dtype, out_dtype)


def test_fail_gaussian_blur():
    for dev in ["cpu", "gpu"]:
        # Check layout and channel placement errors
        for shape, layout, axes in [((20, 20, 30, 3), "DHCW", 3), ((5, 20, 30, 3), "HFWC", 2),
                                    ((5, 10, 10, 10, 7, 3), "FWXYZC", 4),
                                    ((5, 3, 20, 3, 30), "FCHCW", 2),
                                    ((5, 3, 20, 3, 30), "FCCHW", 2)]:
            yield check_fail_gaussian_blur, 10, 1.0, 11, shape, layout, axes, dev
        # Negative, disallowed or both unspecified values of sigma and window size
        yield check_fail_gaussian_blur, 10, 0.0, 0, (100, 20, 3), "HWC", 3, dev
        yield check_fail_gaussian_blur, 10, -1.0, 0, (100, 20, 3), "HWC", 3, dev
        yield check_fail_gaussian_blur, 10, 0.0, -11, (100, 20, 3), "HWC", 3, dev
        yield check_fail_gaussian_blur, 10, 0.0, 2, (100, 20, 3), "HWC", 3, dev
