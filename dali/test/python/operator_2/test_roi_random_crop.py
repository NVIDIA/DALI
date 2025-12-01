# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import random
from nose_utils import assert_raises

np.random.seed(4321)


def random_shape(min_sh, max_sh, ndim):
    return np.array([np.random.randint(min_sh, max_sh) for s in range(ndim)], dtype=np.int32)


def batch_gen(max_batch_size, sample_shape_fn, dtype=np.float32):
    bs = np.random.randint(1, max_batch_size)
    data = []
    for i in range(bs):
        sample_sh = sample_shape_fn()
        data += [np.zeros(sample_sh, dtype=dtype)]
    return data


def check_roi_random_crop(
    ndim=2,
    max_batch_size=16,
    roi_min_start=0,
    roi_max_start=100,
    roi_min_extent=20,
    roi_max_extent=50,
    crop_min_extent=20,
    crop_max_extent=50,
    in_shape_min=400,
    in_shape_max=500,
    niter=3,
):
    pipe = dali.pipeline.Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        assert in_shape_min < in_shape_max

        def shape_gen_fn():
            return random_shape(in_shape_min, in_shape_max, ndim)

        def data_gen_f():
            return batch_gen(max_batch_size, shape_gen_fn)

        shape_like_in = dali.fn.external_source(data_gen_f, device="cpu")
        in_shape = shape_like_in.shape(dtype=types.INT32)

        if random.choice([True, False]):
            crop_shape = [(crop_min_extent + crop_max_extent) // 2] * ndim
        else:
            crop_shape = fn.random.uniform(
                range=(crop_min_extent, crop_max_extent + 1),
                shape=(ndim,),
                dtype=types.INT32,
                device="cpu",
            )

        if random.choice([True, False]):
            roi_shape = [(roi_min_extent + roi_max_extent) // 2] * ndim
            roi_start = [(roi_min_start + roi_max_start) // 2] * ndim
            roi_end = [roi_start[d] + roi_shape[d] for d in range(ndim)]
        else:
            roi_shape = fn.random.uniform(
                range=(roi_min_extent, roi_max_extent + 1),
                shape=(ndim,),
                dtype=types.INT32,
                device="cpu",
            )
            roi_start = fn.random.uniform(
                range=(roi_min_start, roi_max_start + 1),
                shape=(ndim,),
                dtype=types.INT32,
                device="cpu",
            )
            roi_end = roi_start + roi_shape

        outs = [
            fn.roi_random_crop(
                crop_shape=crop_shape, roi_start=roi_start, roi_shape=roi_shape, device="cpu"
            ),
            fn.roi_random_crop(
                crop_shape=crop_shape, roi_start=roi_start, roi_end=roi_end, device="cpu"
            ),
            fn.roi_random_crop(
                shape_like_in,
                crop_shape=crop_shape,
                roi_start=roi_start,
                roi_shape=roi_shape,
                device="cpu",
            ),
            fn.roi_random_crop(
                shape_like_in,
                crop_shape=crop_shape,
                roi_start=roi_start,
                roi_end=roi_end,
                device="cpu",
            ),
            fn.roi_random_crop(
                in_shape=in_shape,
                crop_shape=crop_shape,
                roi_start=roi_start,
                roi_shape=roi_shape,
                device="cpu",
            ),
            fn.roi_random_crop(
                in_shape=in_shape,
                crop_shape=crop_shape,
                roi_start=roi_start,
                roi_end=roi_end,
                device="cpu",
            ),
        ]

    outputs = [in_shape, roi_start, roi_shape, crop_shape, *outs]
    pipe.set_outputs(*outputs)
    for _ in range(niter):
        outputs = pipe.run()
        batch_size = len(outputs[0])
        for s in range(batch_size):
            in_shape = np.array(outputs[0][s]).tolist()
            roi_start = np.array(outputs[1][s]).tolist()
            roi_shape = np.array(outputs[2][s]).tolist()
            crop_shape = np.array(outputs[3][s]).tolist()

            def check_crop_start(crop_start, roi_start, roi_shape, crop_shape, in_shape=None):
                ndim = len(crop_start)
                roi_end = [roi_start[d] + roi_shape[d] for d in range(ndim)]
                crop_end = [crop_start[d] + crop_shape[d] for d in range(ndim)]
                for d in range(ndim):
                    if in_shape is not None:
                        assert crop_start[d] >= 0
                        assert crop_end[d] <= in_shape[d]

                    if crop_shape[d] >= roi_shape[d]:
                        assert crop_start[d] <= roi_start[d]
                        assert crop_end[d] >= roi_end[d]
                    else:
                        assert crop_start[d] >= roi_start[d]
                        assert crop_end[d] <= roi_end[d]

            for idx in range(4, 6):
                check_crop_start(
                    np.array(outputs[idx][s]).tolist(), roi_start, roi_shape, crop_shape
                )
            for idx in range(6, 10):
                check_crop_start(
                    np.array(outputs[idx][s]).tolist(), roi_start, roi_shape, crop_shape, in_shape
                )


def test_roi_random_crop():
    batch_size = 16
    niter = 3
    for ndim in (2, 3):
        in_shape_min = 250
        in_shape_max = 300
        for (
            roi_start_min,
            roi_start_max,
            roi_extent_min,
            roi_extent_max,
            crop_extent_min,
            crop_extent_max,
        ) in [(20, 50, 10, 20, 30, 40), (20, 50, 100, 140, 30, 40), (0, 1, 10, 20, 80, 100)]:
            yield (
                check_roi_random_crop,
                ndim,
                batch_size,
                roi_start_min,
                roi_start_max,
                roi_extent_min,
                roi_extent_max,
                crop_extent_min,
                crop_extent_max,
                in_shape_min,
                in_shape_max,
                niter,
            )


def check_roi_random_crop_error(
    shape_like_in=None,
    in_shape=None,
    crop_shape=None,
    roi_start=None,
    roi_shape=None,
    roi_end=None,
    error_msg="",
):
    batch_size = 3
    niter = 3
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        inputs = [] if shape_like_in is None else [shape_like_in]
        out = fn.roi_random_crop(
            *inputs,
            in_shape=in_shape,
            crop_shape=crop_shape,
            roi_start=roi_start,
            roi_shape=roi_shape,
            roi_end=roi_end,
            device="cpu",
        )
    pipe.set_outputs(out)
    with assert_raises(RuntimeError, regex=error_msg):
        for _ in range(niter):
            pipe.run()


def test_roi_random_crop_error_incompatible_args():
    in_shape = np.array([4, 4])
    crop_shape = np.array([2, 2])
    roi_start = np.array([1, 1])
    roi_shape = np.array([1, 1])
    roi_end = np.array([2, 2])
    yield (
        check_roi_random_crop_error,
        np.zeros(in_shape),
        in_shape,
        crop_shape,
        roi_start,
        roi_shape,
        None,
        "``in_shape`` argument is incompatible with providing an input.",
    )
    yield (
        check_roi_random_crop_error,
        np.zeros(in_shape),
        None,
        crop_shape,
        roi_start,
        roi_shape,
        roi_end,
        "Either ROI end or ROI shape should be defined, but not both",
    )


def test_roi_random_crop_error_wrong_args():
    in_shape = np.array([4, 4])
    crop_shape = np.array([2, 2])
    roi_start = np.array([1, 1])
    roi_shape = np.array([1, 1])
    # Negative shape
    yield (
        check_roi_random_crop_error,
        None,
        np.array([-4, 4]),
        crop_shape,
        roi_start,
        roi_shape,
        None,
        "Input shape can't be negative.",
    )
    yield (
        check_roi_random_crop_error,
        None,
        in_shape,
        np.array([1, -1]),
        roi_start,
        roi_shape,
        None,
        "Crop shape can't be negative",
    )
    # Out of bounds ROI
    yield (
        check_roi_random_crop_error,
        None,
        in_shape,
        crop_shape,
        np.array([-1, -1]),
        roi_shape,
        None,
        "ROI can't be out of bounds.",
    )
    yield (
        check_roi_random_crop_error,
        None,
        in_shape,
        crop_shape,
        roi_start,
        np.array([4, 4]),
        None,
        "ROI can't be out of bounds.",
    )
    yield (
        check_roi_random_crop_error,
        None,
        in_shape,
        crop_shape,
        roi_start,
        None,
        np.array([5, 5]),
        "ROI can't be out of bounds.",
    )
    # Out of bounds crop
    yield (
        check_roi_random_crop_error,
        None,
        in_shape,
        np.array([10, 10]),
        roi_start,
        roi_shape,
        None,
        "Cropping shape can't be bigger than the input shape.",
    )
