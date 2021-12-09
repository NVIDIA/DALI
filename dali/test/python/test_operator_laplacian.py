# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

import numpy as np
import cv2
from scipy.ndimage import convolve1d, filters as sp_filters
import os
from nose_utils import assert_raises

from test_utils import get_dali_extra_path, check_batch, RandomlyShapedDataIterator


data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')

test_iters = 4
min_window_size = 3
max_window_size = 23

shape_layout_axes_cases = [((20, 20, 30, 3), "DHWC", 3), ((20, 20, 30), "", 3),
                           ((20, 30, 3), "HWC", 2), ((20, 30), "HW", 2),
                           ((3, 30, 20), "CWH", 2), ((5, 20, 30, 3), "FHWC", 2),
                           ((5, 10, 10, 7, 3), "FDHWC", 3),
                           ((5, 3, 20, 30), "FCHW", 2),
                           ((3, 5, 10, 10, 7), "CFDHW", 3)]


def to_batch(tl, batch_size):
    return [np.array(tl[i]) for i in range(batch_size)]


# Simple sanity check if laplacian of a square matrix that has zeros everywhere
# execpt the middle cell, gives sum of separable convolution kernels used
# to compute the partial derivatives, i.e. sum of the products of 1D convolution windows.
def _test_kernels(num_dims, smoothing, normalize):
    batch_size = (max_window_size + 2 - min_window_size) // 2

    def get_inputs():
        ones = []
        window_sizes = []
        scales = []
        padding = 2
        for k in range(min_window_size, max_window_size + 2, 2):
            a_size = k + padding
            a = np.full((a_size,) * num_dims, 0, dtype=np.float32)
            a[(a_size // 2,) * num_dims] = 1
            ones.append(a)
            if smoothing:
                window_sizes.append(np.array(k, dtype=np.int32))
                total_win_size = num_dims * k
            else:
                window_sizes.append(
                    np.array([k, *[1] * (num_dims - 1)], dtype=np.int32))
                total_win_size = k + num_dims - 1
            exponent = total_win_size - 2 - num_dims
            scales.append(np.array(2.**(-exponent), dtype=np.float32))
        return ones, window_sizes, scales

    @pipeline_def
    def pipeline():
        ones, window_sizes, scales = fn.external_source(
            get_inputs, num_outputs=3)
        kernels = fn.laplacian(
            ones, window_size=window_sizes, dtype=types.FLOAT, normalize=normalize)
        return kernels, scales

    def outer(*vs):
        acc = np.array([1.])
        for v in vs:
            acc = np.outer(acc, v)
        return acc.reshape(tuple(len(v) for v in vs))

    def get_cv2_kernel(k, smoothing):
        d, s = cv2.getDerivKernels(2, 0, k)
        if not smoothing:
            s = np.zeros(k)
            s[k // 2] = 1.
        windows = [[d if i == j else s for j in range(
            num_dims)] for i in range(num_dims)]
        return sum(outer(*ws) for ws in windows)

    pipe = pipeline(num_threads=4, batch_size=batch_size,
                    device_id=types.CPU_ONLY_DEVICE_ID)
    pipe.build()
    (kernels, scales) = pipe.run()
    kernels = [np.array(ker)[(slice(1, -1),) * num_dims] for ker in kernels]
    scales = [np.array(sf).item() for sf in scales]
    rng = range(min_window_size, max_window_size + 2, 2)
    assert (len(kernels) == len(rng) == len(scales))
    for scale_factor, kernel, k in zip(scales, kernels, rng):
        baseline_kernel = get_cv2_kernel(k, smoothing)
        if normalize:
            baseline_kernel *= scale_factor
        np.testing.assert_allclose(kernel, baseline_kernel, rtol=1e-5)


def test_kernels():
    for num_dims in [1, 2, 3]:
        for normalize in [True, False]:
            for smoothing in [True, False]:
                yield _test_kernels, num_dims, smoothing, normalize


@pipeline_def
def laplacian_pipe(window_size, in_type, out_type, normalize, grayscale):
    # use OpenCV convention - window size 1 implies deriv kernel of size 3 and no smoothing
    window_size = window_size if window_size > 1 else [3, 1]
    if out_type is None:
        dtype_arg = {}
    else:
        dtype_arg = {'dtype': out_type}
    imgs, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    output_type = types.GRAY if grayscale else types.RGB
    imgs = fn.decoders.image(imgs, device="cpu", output_type=output_type)
    if in_type != types.UINT8:
        imgs = fn.cast(imgs, dtype=in_type)
    edges = fn.laplacian(imgs, window_size=window_size,
                         normalize=normalize, **dtype_arg)
    return edges, imgs


def laplacian_cv(imgs, window_size, in_type, out_type, scale, grayscale):
    if out_type == types.UINT8 or (out_type is None and in_type == types.UINT8):
        ddepth = cv2.CV_8U
    else:
        ddepth = cv2.CV_32F
    imgs = [
        cv2.Laplacian(img, ddepth=ddepth, ksize=window_size,
                      borderType=cv2.BORDER_REFLECT_101, scale=scale)
        for img in imgs]
    if grayscale:
        imgs = [np.expand_dims(img, axis=2) for img in imgs]
    return imgs


def normalization_factor(window_size):
    exponent = 0 if window_size == 1 else 2 * window_size - 4
    return 2.**(-exponent)


def _test_vs_open_cv(batch_size, window_size, in_type, out_type, normalize, grayscale):
    pipe = laplacian_pipe(
        device_id=types.CPU_ONLY_DEVICE_ID, num_threads=4, batch_size=batch_size,
        window_size=window_size, in_type=in_type, out_type=out_type,
        normalize=normalize, grayscale=grayscale)
    pipe.build()
    norm_factor = normalization_factor(window_size)
    scale = 1 if not normalize else norm_factor
    for _ in range(test_iters):
        edges, imgs = pipe.run()
        imgs = to_batch(imgs, batch_size)
        baseline_cv = laplacian_cv(
            imgs, window_size, in_type, out_type, scale, grayscale)
        edges = to_batch(edges, batch_size)
        actual_out_type = out_type if out_type is not None else in_type
        assert(len(edges) == len(baseline_cv))
        if actual_out_type == types.FLOAT:
            max_error = 1e-7 if window_size <= 11 else 1e-4
        else:
            max_error = 1
        # values in the array raise exponentially with the window_size, so without normalization
        # the absolute error will also be big - normalize the values before the comparison
        if not normalize:
            edges = [a * norm_factor for a in edges]
            baseline_cv = [a * norm_factor for a in baseline_cv]
        check_batch(edges, baseline_cv, batch_size,
                    max_allowed_error=max_error, expected_layout="HWC")


def test_vs_open_cv():
    batch_size = 10
    for normalize in [True, False]:
        for grayscale in [True, False]:
            for in_type in [types.UINT8, types.FLOAT]:
                output_types = [None] if in_type == types.FLOAT else [
                    None, types.FLOAT]
                for out_type in output_types:
                    # For bigger windows and uint8 mode, cv2 seems to use some integral type that
                    # saturates too early (in any case the resulting picture is mostly black and
                    # different from the result of running cv2.laplacian with floats and then
                    # clamping the results)
                    if out_type is None and in_type == types.UINT8:
                        rng = range(1, 13, 2)
                    else:
                        rng = range(1, max_window_size + 2, 2)
                    for window_size in rng:
                        yield _test_vs_open_cv, batch_size, window_size, in_type, out_type, normalize, grayscale


def laplacian_sp(input, out_type):
    output = [sp_filters.laplace(
        sample, output=out_type, mode='mirror') for sample in input]
    return output


def _test_vs_scipy(batch_size, num_dims, in_type, out_type):
    shape = (30,) * num_dims
    # scipy supports only windows of size 3
    window_size = [3, *([1] * (num_dims - 1))]
    data = RandomlyShapedDataIterator(
        batch_size, max_shape=shape, dtype=in_type)

    @pipeline_def
    def pipeline():
        if out_type == np.float32:
            dtype_args = {'dtype': types.FLOAT}
        else:
            dtype_args = {}
        input = fn.external_source(data)
        edges = fn.laplacian(input, window_size=window_size, **dtype_args)
        return edges, input

    pipe = pipeline(
        device_id=types.CPU_ONLY_DEVICE_ID, num_threads=4, batch_size=batch_size)
    pipe.build()

    for _ in range(test_iters):
        edges, input = pipe.run()
        edges = to_batch(edges, batch_size)
        input = to_batch(input, batch_size)
        baseline = laplacian_sp(input, out_type)
        max_error = 1e-6 if in_type != np.float64 else 1e-4
        check_batch(edges, baseline, batch_size, max_allowed_error=max_error)


def test_vs_scipy():
    batch_size = 10
    for num_dims in [1, 2, 3]:
        # scipy simply wraps integers instead of saturating them, so uint8 inputs won't match
        for in_type in [np.int16, np.int32, np.int64, np.float32, np.float64]:
            output_types = [None] if in_type in [
                np.float32, np.float64] else [None, np.float32]
            for out_type in output_types:
                yield _test_vs_scipy, batch_size, num_dims, in_type, out_type


def conver_sat(img, out_type):
    iinfo = np.iinfo(out_type)
    min_v, max_v = iinfo.min, iinfo.max
    img = np.clip(img, min_v, max_v)
    return img.astype(out_type)


def get_windows(window_sizes):
    axes = len(window_sizes)
    d_windows = {window_sizes[i][i]: None for i in range(axes)}
    s_windows = {window_sizes[i][j]: None for i in range(
        axes) for j in range(axes) if i != j}
    for window_size in d_windows:
        d, s = cv2.getDerivKernels(2, 0, ksize=window_size)
        d_windows[window_size] = d.reshape(-1)
        if window_size > 1 and window_size in s_windows and s_windows[window_size] is None:
            s_windows[window_size] = s.reshape(-1)
    for window_size in s_windows:
        if s_windows[window_size] is None:
            if window_size == 1:
                s_windows[window_size] = np.array([1.], dtype=np.float32)
            else:
                _, s = cv2.getDerivKernels(2, 0, ksize=window_size)
                s_windows[window_size] = s.reshape(-1)
    return [[
        (d_windows if i == j else s_windows)[window_sizes[i][j]]
        for j in range(axes)] for i in range(axes)]


def get_window_sizes(window_sizes, axes):
    if window_sizes is None:
        return [[3 for _ in range(axes)] for _ in range(axes)]
    if len(window_sizes.shape) == 0:
        return [[window_sizes for _ in range(axes)] for _ in range(axes)]
    window_sizes = window_sizes.reshape(-1)
    if len(window_sizes) == 1:
        return [[window_sizes[0] for _ in range(axes)] for _ in range(axes)]
    if len(window_sizes) == axes:
        return [[window_sizes[(axes - j + i) % axes] for i in range(axes)] for j in range(axes)]
    if len(window_sizes) == axes * axes:
        window_sizes = window_sizes.reshape(axes, axes)
        return [[window_sizes[j][i] for i in range(axes)] for j in range(axes)]
    assert(False)


def get_scales(scales, normalize, window_sizes, axes):
    if normalize:
        assert(scales is None)
        sums = [sum(ws) - axes - 2 for ws in window_sizes]
        return [2.**(-s) for s in sums]
    if scales is None:
        return [1. for _ in range(axes)]
    if len(scales.shape) == 0:
        return [scales for _ in range(axes)]
    if len(scales) == 1:
        return [scales[0] for _ in range(axes)]
    if len(scales) == axes:
        return [scales[i] for i in range(axes)]
    assert(False)


def laplacian_baseline(img, out_type, window_sizes, scales, normalize, axes, skip_axes=0):
    window_sizes = get_window_sizes(window_sizes, axes)
    window_sizes = [[v if not isinstance(
        v, np.ndarray) else v.item() for v in ws] for ws in window_sizes]
    scales = get_scales(scales, normalize, window_sizes, axes)
    all_windows = get_windows(window_sizes)
    acc = np.zeros(img.shape, dtype=np.float32)
    img = np.float32(img)
    assert(len(all_windows) == len(scales) == axes)
    for windows, scale in zip(all_windows, scales):
        partial = img
        for i in reversed(range(axes)):
            axis = i + skip_axes
            if img.shape[axis] == 1:
                mode = "nearest"
            else:
                mode = "mirror"
            partial = convolve1d(partial, windows[i], axis, mode=mode)
        acc += scale * partial
    if out_type == np.float32:
        return acc
    else:
        return conver_sat(acc, out_type)


def count_skip_axes(layout):
    if layout.startswith("FC") or layout.startswith("CF"):
        return 2
    if layout.startswith("F") or layout.startswith("C"):
        return 1
    return 0


def random_window_size(window_dim, max_window_size, axes):
    rand_window_size_end = max_window_size // 2 + 1
    rng = np.random.default_rng(seed=42)

    def set_deriv_sizes(win_sizes, start, end):
        if window_dim == 1:
            win_sizes[0] = 2 * rng.integers(start, end, dtype=np.int32) + 1
        else:
            deriv_sizes = 2 * rng.integers(start, end, size=axes, dtype=np.int32) + 1
            for i in range(axes):
                win_sizes[i][i] = deriv_sizes[i]

    def inner():
        if window_dim == 0:
            return np.int32(2 * rng.integers(1, rand_window_size_end) + 1)
        no_smoothing = rng.choice(a=[True, False], p=[0.25, 0.75])
        window_shape = [axes for _ in range(window_dim)]
        if no_smoothing:
            win_sizes = np.ones(window_shape, dtype=np.int32)
            set_deriv_sizes(win_sizes, 1, rand_window_size_end)
            return win_sizes
        win_sizes = 2 * rng.integers(0, rand_window_size_end, size=window_shape, dtype=np.int32) + 1
        set_deriv_sizes(win_sizes, 1, rand_window_size_end)
        return win_sizes
    return inner


@pipeline_def
def laplacian_per_sample_pipeline(iterator, layout, window_dim, max_window_size, axes, use_scale,
                                  spread_scale, normalize, out_type, win_size_1d):
    data = fn.external_source(iterator, layout=layout)
    scale_ndim = 0
    if window_dim is None:
        window_size = -1
        max_scale = 2.**(-2 * axes + 2)
        scale = fn.random.uniform(range=[max_scale / 32, max_scale], shape=[])
        window_arg = None
    else:
        assert(window_dim in [0, 1, 2])
        window_size = fn.external_source(random_window_size(
            window_dim, max_window_size, axes), batch=False)
        window_arg = window_size
        if window_dim == 2:
            if win_size_1d:
                window_arg = fn.reshape(window_arg, shape=[axes * axes])
            max_scale = fn.reductions.sum(window_size, axes=1)
            max_scale = 2.**(-max_scale + axes + 2)
            scale_ndim = 1
        elif window_dim == 1:
            win_sum = fn.reductions.sum(window_size, axes=0)
            max_scale = 2.**(-win_sum + axes + 2)
        else:  # window_dim == 0
            max_scale = 2.**(-(window_size - 1) * axes + 2)
        scale_shape = fn.shapes(max_scale, dtype=types.INT32)
        randomize_scale = fn.random.uniform(
            range=[1 / 32, 1], shape=scale_shape)
        scale = max_scale * randomize_scale
    if not use_scale:
        scale_arg = None
        scales = -1
    else:
        if spread_scale:
            assert(scale_ndim == 0)
            scales = fn.stack(*(scale,) * axes)
        else:
            scales = scale
        scale_arg = scales
    if out_type != np.float32:
        dtype_arg = {}
    else:
        dtype_arg = {"dtype": types.FLOAT}
    edges = fn.laplacian(data, window_size=window_arg,
                         scale=scale_arg, normalize=normalize, **dtype_arg)
    return edges, data, window_size, scales


def check_per_sample_laplacian(
        batch_size, window_dim, use_scale, spread_scale, normalize, shape, layout,
        axes, in_type, out_type, win_size_1d):

    iterator = RandomlyShapedDataIterator(
        batch_size, max_shape=shape, dtype=in_type)

    pipe = laplacian_per_sample_pipeline(
        device_id=types.CPU_ONLY_DEVICE_ID, num_threads=4, batch_size=batch_size, seed=42,
        iterator=iterator, layout=layout, window_dim=window_dim, max_window_size=max_window_size,
        axes=axes, use_scale=use_scale, spread_scale=spread_scale, normalize=normalize,
        out_type=out_type, win_size_1d=win_size_1d)
    pipe.build()

    for _ in range(test_iters):
        outputs = pipe.run()
        edges, data, window_sizes, scales = [
            to_batch(out, batch_size) for out in outputs]
        baseline = []
        for i in range(batch_size):
            skip_axes = count_skip_axes(layout)
            sample_window_size = window_sizes[i] if window_dim is not None else None
            sample_scales = scales[i] if use_scale else None
            baseline.append(laplacian_baseline(
                data[i], out_type or in_type, sample_window_size, sample_scales, normalize, axes, skip_axes))
        if out_type == np.float32:
            # Normalized abs values are up to 255 so it still gives over 5 decimal digits of precision
            max_error = 1e-3
        else:
            max_error = 1
        check_batch(edges, baseline, batch_size,
                    max_allowed_error=max_error, expected_layout=layout)


def test_per_sample_laplacian():
    batch_size = 10
    for in_type in [np.uint8, np.int16, np.int32, np.float32]:
        for out_type in [None, np.float32]:
            for shape, layout, axes in shape_layout_axes_cases:
                for window_dim in [None, 0, 1, 2]:
                    for use_scale in [True, False]:
                        normalize = not use_scale
                        for spread_scale in [True, False] if (window_dim and window_dim < 2) else [False]:
                            for win_size_1d in [True, False] if window_dim == 2 else [True]:
                                yield check_per_sample_laplacian, batch_size, window_dim, use_scale, spread_scale,\
                                    normalize, shape, layout, axes, in_type, out_type, win_size_1d


def check_fixed_param_laplacian(batch_size, in_type, out_type, shape, layout, axes, window_sizes, scales, normalize):

    iterator = RandomlyShapedDataIterator(
        batch_size, max_shape=shape, dtype=in_type)

    @pipeline_def
    def pipeline():
        data = fn.external_source(iterator, layout=layout)
        if out_type != np.float32:
            dtype_arg = {}
        else:
            dtype_arg = {"dtype": types.FLOAT}
        edges = fn.laplacian(data, window_size=window_sizes,
                             scale=scales, normalize=normalize, **dtype_arg)
        return edges, data

    pipe = pipeline(
        device_id=types.CPU_ONLY_DEVICE_ID, num_threads=4, batch_size=batch_size, seed=42)
    pipe.build()

    for _ in range(test_iters):
        outputs = pipe.run()
        edges, data = [
            to_batch(out, batch_size) for out in outputs]
        baseline = []
        for i in range(batch_size):
            skip_axes = count_skip_axes(layout)
            window_sizes = np.array(window_sizes, dtype=np.int32)
            scales = None if scales is None else np.array(
                scales, dtype=np.float32)
            sample = laplacian_baseline(
                data[i], out_type or in_type, window_sizes, scales, normalize, axes, skip_axes)
            baseline.append(sample)
        if out_type == np.float32:
            max_error = 1e-4
        else:
            max_error = 1
        check_batch(edges, baseline, batch_size,
                    max_allowed_error=max_error, expected_layout=layout)


def test_fixed_params_laplacian():
    batch_size = 10
    window_size_cases = {
        1: [3, 5, 7, 9, 11, 21],
        2: [[3, 1], [3, 3], 11, [9, 1], [9, 3], [9, 5, 3, 11]],
        3: [[3, 1, 1], [3, 3, 3], 11, [9, 3, 3], [9, 3, 5, 5, 11, 3, 3, 5, 7]],
    }

    def window_scales(window_sizes, axes):
        win_sizes = get_window_sizes(
            np.array(window_sizes, dtype=np.int32), axes)
        scale = get_scales(None, True, win_sizes, axes)
        cases = [scale]
        if all(scale[0] == s for s in scale):
            cases.append([scale[0]])
        return [[v * factor for v in case] for case in cases for factor in [1 / 16, 4.]]

    for in_type in [np.uint8, np.int32, np.int64, np.float32]:
        for out_type in [None, np.float32]:
            for shape, layout, axes in shape_layout_axes_cases:
                for window_sizes in window_size_cases[axes]:
                    for normalize in [True, False]:
                        for scales in [None] if normalize else window_scales(window_sizes, axes):
                            yield check_fixed_param_laplacian, batch_size, in_type, out_type, shape, layout, axes,\
                                window_sizes, scales, normalize


def check_build_time_fail(batch_size, shape, layout, axes, window_size, scale, normalize, err_regex):
    with assert_raises(RuntimeError, regex=err_regex):
        check_fixed_param_laplacian(
            batch_size, np.uint8, None, shape, layout, axes, window_size, scale, normalize)


def check_tensor_input_fail(batch_size, shape, layout, window_size, scale, normalize, dtype, err_regex):
    iterator = RandomlyShapedDataIterator(
        batch_size, max_shape=shape, dtype=np.uint8)

    def gen_params():
        return np.array(window_size, dtype=np.int32), np.array(scale, dtype=np.float32)

    @pipeline_def
    def pipeline():
        data = fn.external_source(iterator, layout=layout)
        window_size, scale = fn.external_source(
            gen_params, batch=False, num_outputs=2)
        edges = fn.laplacian(data, window_size=window_size,
                             scale=scale, normalize=normalize, dtype=dtype)
        return edges, data

    with assert_raises(RuntimeError, regex=err_regex):
        pipe = pipeline(device_id=types.CPU_ONLY_DEVICE_ID,
                        num_threads=4, batch_size=batch_size)
        pipe.build()
        pipe.run()


def test_fail_laplacian():
    args = [
        ((20, 20, 30, 3), "DHCW", 3,
            "Only channel-first or channel-last layouts are supported, got: .*\."),
        ((5, 20, 30, 3), "HFWC", 2,
            "For sequences, layout should begin with 'F' or 'CF', got: .*\."),
        ((5, 10, 10, 10, 7, 3), "FWXYZC", 4,
            "Too many dimensions, found: \d+ data axes, maximum supported is: 3\."),
        ((5, 3, 20, 3, 30), "FCHCW", 2,
            "Only channel-first or channel-last layouts are supported, got: .*\."),
        ((5, 3, 20, 3, 30), "FCCHW", 2,
            "Found more the one occurrence of 'F' or 'C' axes in layout: .*\.")
    ]
    for shape, layout, axes, err_regex in args:
        yield check_build_time_fail, 10, shape, layout, axes, 11, 1., False, err_regex
    yield check_tensor_input_fail, 10, (10, 10, 3), "HWC", 11, 1., False, types.UINT16, \
        "Output data type must be same as input, FLOAT or skipped"

    yield check_build_time_fail, 10, (10, 10, 3), "HWC", 2, 11, 1., True, \
        "Parameter ``scale`` cannot be specified when ``normalize`` is set to True"
    for window_size in [-3, 10, max_window_size + 1]:
        yield check_build_time_fail, 10, (10, 10, 3), "HWC", 2, window_size, 1., False, \
            "Derivative window size must be an odd integer between 3 and \d"
        yield check_tensor_input_fail, 10, (10, 10, 3), "HWC", window_size, 1., False, types.FLOAT, \
            "Derivative window size must be an odd integer between 3 and \d"
    for window_size in [[3, 6], [5, -1], [7, 7, max_window_size + 1, 9]]:
        yield check_build_time_fail, 10, (10, 10, 3), "HWC", 2, window_size, 1., False, \
            "Smoothing window size must be an odd integer between 1 and \d"
        yield check_tensor_input_fail, 10, (10, 10, 3), "HWC", window_size, 1., False, types.FLOAT, \
            "Smoothing window size must be an odd integer between 1 and \d"
    for window_size in [[3, 7, 3], [7, 7, 7, 7, 7]]:
        yield check_build_time_fail, 10, (10, 10, 3), "HWC", 2, window_size, 1., False, \
            "Argument `window_size` is expected to have 1, 2 or 2x2 elements, got {}".format(
                len(window_size))
        yield check_tensor_input_fail, 10, (10, 10, 3), "HWC", window_size, 1., False, types.FLOAT, \
            "Argument `window_size` for sample \d is expected to have 1, 2 or 2x2 elements, got {}".format(
                len(window_size))
    for scale in [[3, 7, 3], [7, 7, 7, 7, 7]]:
        yield check_build_time_fail, 10, (10, 10, 3), "HWC", 2, 3, scale, False, \
            "Argument \"scale\" expects either a single value or a list of 2 elements. {} given.".format(
                len(scale))
        yield check_tensor_input_fail, 10, (10, 10, 3), "HWC", 5, scale, False, types.FLOAT, \
            "Argument scale for sample 0 is expected to have 1 or 2 elements, got: {}".format(
                len(scale))
