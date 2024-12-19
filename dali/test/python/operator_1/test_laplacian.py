# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nose_utils import assert_raises, attr

from sequences_test_utils import video_suite_helper, ArgCb
from test_utils import get_dali_extra_path, check_batch, RandomlyShapedDataIterator


data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")

test_iters = 4
min_window_size = 3
max_window_size = 31  # it is maximal window size supported by opencv

shape_layout_axes_cases = [
    ((20, 20, 30, 3), "DHWC", 3),
    ((20, 20, 30), "", 3),
    ((20, 30, 3), "HWC", 2),
    ((20, 30), "HW", 2),
    ((3, 30, 20), "CWH", 2),
    ((5, 20, 30, 3), "FHWC", 2),
    ((5, 10, 10, 7, 3), "FDHWC", 3),
    ((5, 3, 20, 30), "FCHW", 2),
    ((3, 5, 10, 10, 7), "FCDHW", 3),
]


def to_batch(tl, batch_size):
    return [np.array(tl[i]) for i in range(batch_size)]


# Simple check if laplacian of a square matrix that has zeros everywhere except
# the middle cell, gives sum of separable convolution kernels used to compute
# the partial derivatives, i.e. sum of the products of 1D convolution windows.
def _test_kernels(device, num_dims, smoothing, normalize):
    batch_size = (max_window_size + 2 - min_window_size) // 2

    def get_inputs():
        ones = []
        window_sizes = []
        smoothing_sizes = []
        scales = []
        padding = 2
        for win_size in range(min_window_size, max_window_size + 2, 2):
            a_size = win_size + padding
            a = np.zeros((a_size,) * num_dims, dtype=np.float32)
            a[(a_size // 2,) * num_dims] = 1
            ones.append(a)
            window_sizes.append(np.array(win_size, dtype=np.int32))
            if smoothing:
                smoothing_sizes.append(np.array(win_size, dtype=np.int32))
                exponent = num_dims * win_size - 2 - num_dims
            else:
                smoothing_sizes.append(np.array(1, dtype=np.int32))
                exponent = win_size - 3
            scales.append(np.array(2.0 ** (-exponent), dtype=np.float32))
        return ones, window_sizes, smoothing_sizes, scales

    @pipeline_def
    def pipeline():
        ones, window_sizes, smoothing_sizes, scales = fn.external_source(get_inputs, num_outputs=4)
        if device == "gpu":
            ones = ones.gpu()
        kernels = fn.laplacian(
            ones,
            window_size=window_sizes,
            smoothing_size=smoothing_sizes,
            dtype=types.FLOAT,
            normalized_kernel=normalize,
            device=device,
        )
        return kernels, scales

    def outer(*vs):
        acc = np.array([1.0])
        for v in vs:
            acc = np.outer(acc, v)
        return acc.reshape(tuple(len(v) for v in vs))

    def get_cv2_kernel(win_size, smoothing):
        d, s = cv2.getDerivKernels(2, 0, win_size)
        if not smoothing:
            s = np.zeros(win_size)
            s[win_size // 2] = 1.0
        windows = [[d if i == j else s for j in range(num_dims)] for i in range(num_dims)]
        return sum(outer(*ws) for ws in windows)

    pipe = pipeline(num_threads=4, batch_size=batch_size, device_id=0)
    (kernels, scales) = pipe.run()
    if device == "gpu":
        kernels = kernels.as_cpu()
    kernels = [np.array(ker)[(slice(1, -1),) * num_dims] for ker in kernels]
    scales = [np.array(sf).item() for sf in scales]
    win_sizes = range(min_window_size, max_window_size + 2, 2)
    assert len(kernels) == len(win_sizes) == len(scales)
    baseline_kernels = [
        get_cv2_kernel(win_size, smoothing) * scale for win_size, scale in zip(win_sizes, scales)
    ]
    if not normalize:  # output was not normalized by the op
        kernels = [kernel * scale for kernel, scale in zip(kernels, scales)]
    check_batch(
        kernels, baseline_kernels, batch_size, max_allowed_error=1e-5, expected_layout="HWC"
    )


def test_kernels():
    for device in ["cpu", "gpu"]:
        for num_dims in [1, 2, 3]:
            for normalize in [True, False]:
                for smoothing in [True, False]:
                    yield _test_kernels, device, num_dims, smoothing, normalize


@pipeline_def
def laplacian_pipe(device, window_size, in_type, out_type, normalize, grayscale):
    # use OpenCV convention - window size 1 implies deriv kernel of size 3 and no smoothing
    if window_size == 1:
        window_size, smoothing_size = 3, 1
    else:
        smoothing_size = None
    imgs, _ = fn.readers.file(file_root=images_dir, shard_id=0, num_shards=1)
    output_type = types.GRAY if grayscale else types.RGB
    imgs = fn.decoders.image(imgs, device="cpu", output_type=output_type)
    if in_type != types.UINT8:
        imgs = fn.cast(imgs, dtype=in_type)
    if device == "gpu":
        imgs = imgs.gpu()
    if out_type == in_type:
        out_type = None
    edges = fn.laplacian(
        imgs,
        window_size=window_size,
        smoothing_size=smoothing_size,
        normalized_kernel=normalize,
        dtype=out_type,
        device=device,
    )
    return edges, imgs


def laplacian_cv(imgs, window_size, in_type, out_type, scale, grayscale):
    if out_type == types.UINT8 or (out_type is None and in_type == types.UINT8):
        ddepth = cv2.CV_8U
    else:
        ddepth = cv2.CV_32F
    imgs = [
        cv2.Laplacian(
            img, ddepth=ddepth, ksize=window_size, borderType=cv2.BORDER_REFLECT_101, scale=scale
        )
        for img in imgs
    ]
    if grayscale:
        imgs = [np.expand_dims(img, axis=2) for img in imgs]
    return imgs


def normalization_factor(window_size):
    exponent = 0 if window_size == 1 else 2 * window_size - 4
    return 2.0 ** (-exponent)


def _test_vs_open_cv(device, batch_size, window_size, in_type, out_type, normalize, grayscale):
    pipe = laplacian_pipe(
        device_id=0,
        device=device,
        num_threads=4,
        batch_size=batch_size,
        window_size=window_size,
        in_type=in_type,
        out_type=out_type,
        normalize=normalize,
        grayscale=grayscale,
    )
    norm_factor = normalization_factor(window_size)
    scale = 1 if not normalize else norm_factor
    for _ in range(test_iters):
        edges, imgs = pipe.run()
        if device == "gpu":
            edges = edges.as_cpu()
            imgs = imgs.as_cpu()
        imgs = to_batch(imgs, batch_size)
        baseline_cv = laplacian_cv(imgs, window_size, in_type, out_type, scale, grayscale)
        edges = to_batch(edges, batch_size)
        actual_out_type = out_type if out_type is not None else in_type
        assert len(edges) == len(baseline_cv)
        if actual_out_type == types.FLOAT:
            max_error = 1e-7 if window_size <= 11 else 1e-4
        else:
            max_error = 1
        # values in the array raise exponentially with the window_size, so without normalization
        # the absolute error will also be big - normalize the values before the comparison
        if not normalize:
            edges = [a * norm_factor for a in edges]
            baseline_cv = [a * norm_factor for a in baseline_cv]
        check_batch(
            edges, baseline_cv, batch_size, max_allowed_error=max_error, expected_layout="HWC"
        )


def test_vs_open_cv():
    batch_size = 10
    for device in ["cpu", "gpu"]:
        # they are independent parameters, it's just not to go overboard with test cases
        for normalize, grayscale in ((True, False), (False, True)):
            # For bigger windows and uint8 mode, cv2 seems to use some integral type that
            # saturates too early (in any case the resulting picture is mostly black and
            # different from the result of running cv2.laplacian with floats and then
            # clamping the results)
            for window_size in range(1, 13, 2):
                yield (
                    _test_vs_open_cv,
                    device,
                    batch_size,
                    window_size,
                    types.UINT8,
                    None,
                    normalize,
                    grayscale,
                )


@attr("slow")
def slow_test_vs_open_cv():
    batch_size = 10
    for device in ["cpu", "gpu"]:
        # they are independent parameters, it's just not to go overboard with test cases
        for normalize, grayscale in ((True, False), (False, True)):
            for in_type, out_type in ((types.UINT8, types.FLOAT), (types.FLOAT, None)):
                for window_size in [1] + list(range(3, max_window_size + 2, 4)):
                    yield (
                        _test_vs_open_cv,
                        device,
                        batch_size,
                        window_size,
                        in_type,
                        out_type,
                        normalize,
                        grayscale,
                    )


def laplacian_sp(input, out_type):
    output = [sp_filters.laplace(sample, output=out_type, mode="mirror") for sample in input]
    return output


def _test_vs_scipy(device, batch_size, num_dims, in_type, out_type):
    shape = (30,) * num_dims
    # scipy supports only windows of size 3 and does not use smoothing
    window_size, smoothing_size = 3, 1
    data = RandomlyShapedDataIterator(batch_size, max_shape=shape, dtype=in_type)

    @pipeline_def
    def pipeline():
        if out_type == np.float32:
            dtype_args = {"dtype": types.FLOAT}
        else:
            dtype_args = {}
        input = fn.external_source(data)
        if device == "gpu":
            input = input.gpu()
        edges = fn.laplacian(
            input,
            window_size=window_size,
            device=device,
            smoothing_size=smoothing_size,
            **dtype_args,
        )
        return edges, input

    pipe = pipeline(device_id=0, num_threads=4, batch_size=batch_size)

    for _ in range(test_iters):
        edges, input = pipe.run()
        if device == "gpu":
            edges = edges.as_cpu()
            input = input.as_cpu()
        edges = to_batch(edges, batch_size)
        input = to_batch(input, batch_size)
        baseline = laplacian_sp(input, out_type)
        max_error = 1e-6
        check_batch(edges, baseline, batch_size, max_allowed_error=max_error)


def test_vs_scipy():
    batch_size = 10
    for device in ["cpu", "gpu"]:
        for num_dims in [1, 2, 3]:
            # scipy simply wraps integers instead of saturating them, so uint8 inputs won't match
            for in_type in [np.int16, np.int32, np.int64, np.float32]:
                output_types = [None] if in_type == np.float32 else [None, np.float32]
                for out_type in output_types:
                    yield _test_vs_scipy, device, batch_size, num_dims, in_type, out_type


def convert_sat(img, out_type):
    iinfo = np.iinfo(out_type)
    min_v, max_v = iinfo.min, iinfo.max
    img = np.clip(img, min_v, max_v)
    return img.astype(out_type)


def spread_values(out, axes):
    out = out.reshape(-1)
    if len(out) == 0:
        return [3] * axes
    if len(out) == 1:
        return [out[0]] * axes
    if len(out) == axes:
        return [out[i] for i in range(axes)]
    assert False


def get_windows(window_sizes):
    axes = len(window_sizes)
    d_windows = {window_sizes[i][i]: None for i in range(axes)}
    s_windows = {window_sizes[i][j]: None for i in range(axes) for j in range(axes) if i != j}
    for window_size in d_windows:
        d, s = cv2.getDerivKernels(2, 0, ksize=window_size)
        d_windows[window_size] = d.reshape(-1)
        if window_size > 1 and window_size in s_windows and s_windows[window_size] is None:
            s_windows[window_size] = s.reshape(-1)
    for window_size in s_windows:
        if s_windows[window_size] is None:
            if window_size == 1:
                s_windows[window_size] = np.array([1.0], dtype=np.float32)
            else:
                _, s = cv2.getDerivKernels(2, 0, ksize=window_size)
                s_windows[window_size] = s.reshape(-1)
    return [
        [(d_windows if i == j else s_windows)[window_sizes[i][j]] for j in range(axes)]
        for i in range(axes)
    ]


def get_window_sizes(window_size, smoothing_size, axes):
    window_sizes = spread_values(window_size, axes)
    if len(smoothing_size.reshape(-1)) == 0:
        return [[window_sizes[i]] * axes for i in range(axes)]
    else:
        smoothing_sizes = spread_values(smoothing_size, axes)
        return [
            [window_sizes[j] if i == j else smoothing_sizes[j] for j in range(axes)]
            for i in range(axes)
        ]


def laplacian_baseline(img, out_type, window_size, smoothing_size, scale, axes, skip_axes=0):
    scales = spread_values(scale, axes)
    all_sizes = get_window_sizes(window_size, smoothing_size, axes)
    acc = np.zeros(img.shape, dtype=np.float32)
    img = np.float32(img)
    all_windows = get_windows(all_sizes)
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
        return convert_sat(acc, out_type)


def count_skip_axes(layout):
    if layout.startswith("FC") or layout.startswith("CF"):
        return 2
    if layout.startswith("F") or layout.startswith("C"):
        return 1
    return 0


@pipeline_def
def laplacian_per_sample_pipeline(
    device, iterator, layout, window_dim, smoothing_dim, axes, normalize, out_type
):
    data = fn.external_source(iterator, layout=layout)
    if window_dim is None:
        window_size = 3
        w_exponent = 0
        window_arg = None
    else:
        window_shape = [axes for _ in range(window_dim)]
        window_size = (
            fn.random.uniform(
                range=[1, max_window_size // 2], shape=window_shape, dtype=types.INT32
            )
            * 2
            + 1
        )
        window_arg = window_size
        w_exponent = window_size - 3

    if smoothing_dim is None:
        smoothing_size = None
        s_exponent = (window_size - 1) * (axes - 1)
    else:
        smoothing_shape = [axes for _ in range(smoothing_dim)]
        smoothing_size = (
            fn.random.uniform(
                range=[0, max_window_size // 2], shape=smoothing_shape, dtype=types.INT32
            )
            * 2
            + 1
        )
        if smoothing_dim == 1:
            s_exponent = fn.reductions.sum(smoothing_size, axes=0) - smoothing_size - axes + 1
        else:
            s_exponent = (smoothing_size - 1) * (axes - 1)

    exponent = w_exponent + s_exponent
    scale = 2.0 ** (-exponent)
    kwargs = {"normalized_kernel": True} if normalize else {"scale": scale}

    if out_type == np.float32:
        kwargs["dtype"] = types.FLOAT

    if device == "gpu":
        data = data.gpu()
    edges = fn.laplacian(
        data, window_size=window_arg, device=device, smoothing_size=smoothing_size, **kwargs
    )

    if smoothing_size is None:
        smoothing_size = np.array([], dtype=np.int32)
    if window_arg is None:
        window_arg = np.array([], dtype=np.int32)
    return edges, data, window_arg, smoothing_size, scale


def check_per_sample_laplacian(
    device, batch_size, window_dim, smoothing_dim, normalize, shape, layout, axes, in_type, out_type
):
    iterator = RandomlyShapedDataIterator(batch_size, max_shape=shape, dtype=in_type)

    pipe = laplacian_per_sample_pipeline(
        device_id=0,
        device=device,
        num_threads=4,
        batch_size=batch_size,
        seed=42,
        iterator=iterator,
        layout=layout,
        window_dim=window_dim,
        smoothing_dim=smoothing_dim,
        axes=axes,
        normalize=normalize,
        out_type=out_type,
    )

    for _ in range(test_iters):
        edges, data, window_size, smoothing_size, scale = pipe.run()
        if device == "gpu":
            edges = edges.as_cpu()
            data = data.as_cpu()
        edges, data, window_size, smoothing_size, scale = [
            to_batch(out, batch_size) for out in (edges, data, window_size, smoothing_size, scale)
        ]
        baseline = []
        for i in range(batch_size):
            skip_axes = count_skip_axes(layout)
            sample_baseline = laplacian_baseline(
                data[i],
                out_type or in_type,
                window_size[i],
                smoothing_size[i],
                scale[i],
                axes,
                skip_axes,
            )
            baseline.append(sample_baseline)
        if out_type == np.float32:
            # Normalized abs values are up to 2 * `axes` * 255 so it still gives
            # over 5 decimal digits of precision
            max_error = 1e-3
        else:
            max_error = 1
        check_batch(
            edges, baseline, batch_size, max_allowed_error=max_error, expected_layout=layout
        )


def test_per_sample_laplacian():
    batch_size = 10
    for device in ["cpu", "gpu"]:
        for in_type in [np.uint8]:
            for out_type in [None, np.float32]:
                for shape, layout, axes in shape_layout_axes_cases:
                    for normalize in [True, False]:
                        yield (
                            check_per_sample_laplacian,
                            device,
                            batch_size,
                            1,
                            1,
                            normalize,
                            shape,
                            layout,
                            axes,
                            in_type,
                            out_type,
                        )


@attr("slow")
def slow_test_per_sample_laplacian():
    batch_size = 10
    for device in ["cpu", "gpu"]:
        for in_type in [np.int16, np.int32, np.float32]:
            for out_type in [None, np.float32]:
                if out_type == in_type:
                    continue
                for shape, layout, axes in shape_layout_axes_cases:
                    full_test = [None, 0, 1]
                    for window_dim in full_test if in_type == np.float32 else [1]:
                        for smoothing_dim in full_test if in_type == np.float32 else [1]:
                            for normalize in [True, False]:
                                yield (
                                    check_per_sample_laplacian,
                                    device,
                                    batch_size,
                                    window_dim,
                                    smoothing_dim,
                                    normalize,
                                    shape,
                                    layout,
                                    axes,
                                    in_type,
                                    out_type,
                                )


def check_fixed_param_laplacian(
    device,
    batch_size,
    in_type,
    out_type,
    shape,
    layout,
    axes,
    window_size,
    smoothing_size,
    scales,
    normalize,
):
    iterator = RandomlyShapedDataIterator(batch_size, max_shape=shape, dtype=in_type)

    @pipeline_def
    def pipeline():
        data = fn.external_source(iterator, layout=layout)
        if out_type != np.float32:
            dtype_arg = {}
        else:
            dtype_arg = {"dtype": types.FLOAT}
        if device == "gpu":
            data = data.gpu()
        edges = fn.laplacian(
            data,
            window_size=window_size,
            smoothing_size=smoothing_size,
            scale=scales,
            normalized_kernel=normalize,
            **dtype_arg,
        )
        return edges, data

    pipe = pipeline(device_id=0, num_threads=4, batch_size=batch_size, seed=42)

    for _ in range(test_iters):
        edges, data = pipe.run()
        if device == "gpu":
            edges = edges.as_cpu()
            data = data.as_cpu()
        edges = to_batch(edges, batch_size)
        data = to_batch(data, batch_size)
        baseline = []
        for i in range(batch_size):
            skip_axes = count_skip_axes(layout)
            window_size = (
                np.array([]) if window_size is None else np.array(window_size, dtype=np.int32)
            )
            smoothing_size = (
                np.array([]) if smoothing_size is None else np.array(smoothing_size, dtype=np.int32)
            )
            if normalize:
                all_sizes = get_window_sizes(window_size, smoothing_size, axes)
                scales = [2.0 ** (-sum(sizes) + axes + 2) for sizes in all_sizes]
            scales = np.array(scales, dtype=np.float32)
            sample = laplacian_baseline(
                data[i], out_type or in_type, window_size, smoothing_size, scales, axes, skip_axes
            )
            baseline.append(sample)
        if out_type == np.float32:
            max_error = 1e-3
        else:
            max_error = 1
        check_batch(
            edges, baseline, batch_size, max_allowed_error=max_error, expected_layout=layout
        )


@attr("slow")
def slow_test_fixed_params_laplacian():
    batch_size = 10
    window_size_cases = {
        1: [None, 3, 5, 9, 21],
        2: [None, [3, 3], 11, [9, 5], [3, 17]],
        3: [None, [3, 5, 7], [3, 3, 3], 11, [23, 7, 11]],
    }
    smoothing_size_cases = {
        1: [None, 1, 3, 11, 21],
        2: [None, [1, 3], 1, 11, [9, 5]],
        3: [None, [3, 5, 7], 1, 11, [9, 7, 1]],
    }

    def window_scales(window_sizes, smoothing_sizes, axes):
        window_sizes = np.array([]) if window_sizes is None else np.array(window_sizes)
        smoothing_sizes = np.array([]) if smoothing_sizes is None else np.array(smoothing_sizes)
        all_sizes = get_window_sizes(window_sizes, smoothing_sizes, axes)
        scales = [2.0 ** (-sum(sizes) + axes + 2) for sizes in all_sizes]
        cases = [scales]
        if all(scales[0] == s for s in scales):
            cases.append([scales[0]])
        return [[v * factor for v in case] for case in cases for factor in [1 / 16, 4.0]]

    for device in ["cpu", "gpu"]:
        for in_type in [np.uint8, np.int32, np.int64, np.float32]:
            for out_type in [None, np.float32]:
                if in_type == out_type:
                    continue
                for shape, layout, axes in shape_layout_axes_cases:
                    for window_sizes in window_size_cases[axes]:
                        for smooth_sizes in smoothing_size_cases[axes]:
                            for normalize in [True, False]:
                                if normalize:
                                    scale_cases = [None]
                                else:
                                    scale_cases = window_scales(window_sizes, smooth_sizes, axes)
                                for scales in scale_cases:
                                    yield (
                                        check_fixed_param_laplacian,
                                        device,
                                        batch_size,
                                        in_type,
                                        out_type,
                                        shape,
                                        layout,
                                        axes,
                                        window_sizes,
                                        smooth_sizes,
                                        scales,
                                        normalize,
                                    )


def check_build_time_fail(
    device,
    batch_size,
    shape,
    layout,
    axes,
    window_size,
    smoothing_size,
    scale,
    normalize,
    err_regex,
):
    with assert_raises(RuntimeError, regex=err_regex):
        check_fixed_param_laplacian(
            device,
            batch_size,
            np.uint8,
            None,
            shape,
            layout,
            axes,
            window_size,
            smoothing_size,
            scale,
            normalize,
        )


def check_tensor_input_fail(
    device,
    batch_size,
    shape,
    layout,
    window_size,
    smoothing_size,
    scale,
    normalize,
    dtype,
    err_regex,
):
    iterator = RandomlyShapedDataIterator(batch_size, max_shape=shape, dtype=np.uint8)

    def gen_params():
        return (
            np.array(window_size, dtype=np.int32),
            np.array(smoothing_size, dtype=np.int32),
            np.array(scale, dtype=np.float32),
        )

    @pipeline_def
    def pipeline():
        data = fn.external_source(iterator, layout=layout)
        window_size, smoothing_size, scale = fn.external_source(
            gen_params, batch=False, num_outputs=3
        )
        if device == "gpu":
            data = data.gpu()
        edges = fn.laplacian(
            data,
            window_size=window_size,
            smoothing_size=smoothing_size,
            scale=scale,
            normalized_kernel=normalize,
            dtype=dtype,
            device=device,
        )
        return edges, data

    with assert_raises(RuntimeError, regex=err_regex):
        pipe = pipeline(device_id=0, num_threads=4, batch_size=batch_size)
        pipe.run()


def test_fail_laplacian():
    args = [
        (
            (20, 20, 30, 3),
            "DHCW",
            3,
            "Only channel-first or channel-last layouts are supported, got: .*\\.",
        ),
        (
            (5, 20, 30, 3),
            "HFWC",
            2,
            "For sequences, layout should begin with 'F' or 'C', got: .*\\.",
        ),
        (
            (5, 10, 10, 10, 7, 3),
            "FWXYZC",
            4,
            "Too many dimensions, found: \\d+ data axes, maximum supported is: 3\\.",
        ),
        (
            (5, 3, 20, 3, 30),
            "FCHCW",
            2,
            "Only channel-first or channel-last layouts are supported, got: .*\\.",
        ),
        (
            (5, 3, 20, 3, 30),
            "FCCHW",
            2,
            "Found more the one occurrence of 'F' or 'C' axes in layout: .*\\.",
        ),
        ((5, 3), "CF", 2, "No spatial axes found in the layout"),
    ]
    for device in "cpu", "gpu":
        for shape, layout, axes, err_regex in args:
            yield (
                check_build_time_fail,
                device,
                10,
                shape,
                layout,
                axes,
                11,
                11,
                1.0,
                False,
                err_regex,
            )
        yield (
            check_tensor_input_fail,
            device,
            10,
            (
                10,
                10,
                3,
            ),
            "HWC",
            11,
            11,
            1.0,
            False,
            types.UINT16,
            "Output data type must be same as input, FLOAT or skipped",
        )

        yield (
            check_build_time_fail,
            device,
            10,
            (
                10,
                10,
                3,
            ),
            "HWC",
            2,
            11,
            11,
            1.0,
            True,
            "Parameter ``scale`` cannot be specified when ``normalized_kernel`` is set to True",
        )
        for window_size in [-3, 10, max_window_size + 1]:
            yield (
                check_build_time_fail,
                device,
                10,
                (
                    10,
                    10,
                    3,
                ),
                "HWC",
                2,
                window_size,
                5,
                1.0,
                False,
                "Window size must be an odd integer between 3 and \\d",
            )
            yield (
                check_tensor_input_fail,
                device,
                10,
                (
                    10,
                    10,
                    3,
                ),
                "HWC",
                window_size,
                5,
                1.0,
                False,
                types.FLOAT,
                "Window size must be an odd integer between 3 and \\d",
            )
        for window_size in [[3, 6], -1, max_window_size + 1]:
            yield (
                check_build_time_fail,
                device,
                10,
                (
                    10,
                    10,
                    3,
                ),
                "HWC",
                2,
                3,
                window_size,
                1.0,
                False,
                "Smoothing window size must be an odd integer between 1 and \\d",
            )
        for window_size in [6, -1, max_window_size + 1]:
            yield (
                check_tensor_input_fail,
                device,
                10,
                (
                    10,
                    10,
                    3,
                ),
                "HWC",
                3,
                window_size,
                1.0,
                False,
                types.FLOAT,
                "Smoothing window size must be an odd integer between 1 and \\d",
            )
        for window_size in [[3, 7, 3], [7, 7, 7, 7, 7]]:
            yield check_build_time_fail, device, 10, (
                10,
                10,
                3,
            ), "HWC", 2, window_size, 11, 1.0, False, (
                f'Argument "window_size" expects either a single value '
                f"or a list of 2 elements. {len(window_size)} given"
            )
            yield check_tensor_input_fail, device, 10, (
                10,
                10,
                3,
            ), "HWC", window_size, 11, 1.0, False, types.FLOAT, (
                f"Argument window_size for sample 0 is expected to have "
                f"1 or 2 elements, got: {len(window_size)}"
            )
        for scale in [[3, 7, 3], [7, 7, 7, 7, 7]]:
            yield check_build_time_fail, device, 10, (10, 10, 3), "HWC", 2, 3, 3, scale, False, (
                f'Argument "scale" expects either a single value or a list '
                f"of 2 elements. {len(scale)} given."
            )
            yield check_tensor_input_fail, device, 10, (
                10,
                10,
                3,
            ), "HWC", 5, 5, scale, False, types.FLOAT, (
                f"Argument scale for sample 0 is expected to have "
                f"1 or 2 elements, got: {len(scale)}"
            )


def test_per_frame():
    def window_size(sample_desc):
        return np.array(2 * sample_desc.rng.randint(1, 15) + 1, dtype=np.int32)

    def per_axis_window_size(sample_desc):
        return np.array([window_size(sample_desc) for _ in range(2)])

    def per_axis_smoothing_size(sample_desc):
        return np.array([2 * sample_desc.rng.randint(0, 15) + 1 for _ in range(2)], dtype=np.int32)

    def per_axis_scale(sample_desc):
        def scale(sample_desc):
            k = 2 * sample_desc.rng.randint(0, 15) + 1
            return np.array(2.0**-k, dtype=np.float32)

        return np.array([scale(sample_desc) for _ in range(2)])

    video_test_cases = [
        (fn.laplacian, {}, []),
        (fn.laplacian, {}, [ArgCb("window_size", window_size, True)]),
        (fn.laplacian, {}, [ArgCb("window_size", per_axis_window_size, True)]),
        (fn.laplacian, {"dtype": types.FLOAT}, [ArgCb("scale", per_axis_scale, True)]),
        (
            fn.laplacian,
            {},
            [
                ArgCb("window_size", per_axis_window_size, True),
                ArgCb("smoothing_size", per_axis_smoothing_size, True),
            ],
        ),
        (
            fn.laplacian,
            {},
            [
                ArgCb("window_size", per_axis_window_size, True),
                ArgCb("smoothing_size", per_axis_smoothing_size, True),
                ArgCb("scale", per_axis_scale, True),
            ],
        ),
    ]

    yield from video_suite_helper(video_test_cases, expand_channels=True)
