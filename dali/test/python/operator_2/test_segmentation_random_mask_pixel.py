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

import nose_utils  # noqa:F401
import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as math

np.random.seed(4321)


def check_random_mask_pixel(ndim=2, batch_size=3, min_extent=20, max_extent=50):
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        # Input mask
        in_shape_dims = [
            fn.cast(fn.random.uniform(range=(min_extent, max_extent + 1)), dtype=types.INT32)
            for _ in range(ndim)
        ]
        in_shape = fn.stack(*in_shape_dims)
        in_mask = fn.cast(fn.random.uniform(range=(0, 2), shape=in_shape), dtype=types.INT32)

        #  > 0
        fg_pixel1 = fn.segmentation.random_mask_pixel(in_mask, foreground=1)
        #  >= 0.99
        fg_pixel2 = fn.segmentation.random_mask_pixel(in_mask, foreground=1, threshold=0.99)
        #  == 2
        fg_pixel3 = fn.segmentation.random_mask_pixel(in_mask, foreground=1, value=2)

        rnd_pixel = fn.segmentation.random_mask_pixel(in_mask, foreground=0)

        coin_flip = fn.random.coin_flip(probability=0.7)
        fg_biased = fn.segmentation.random_mask_pixel(in_mask, foreground=coin_flip)

        # Demo purposes: Taking a random pixel and produce a valid anchor to feed slice
        # We want to force the center adjustment, thus the large crop shape
        crop_shape = in_shape - 2
        anchor = fn.cast(fg_pixel1, dtype=types.INT32) - crop_shape // 2
        anchor = math.min(math.max(0, anchor), in_shape - crop_shape)
        out_mask = fn.slice(in_mask, anchor, crop_shape, axes=tuple(range(ndim)))

    pipe.set_outputs(
        in_mask,
        fg_pixel1,
        fg_pixel2,
        fg_pixel3,
        rnd_pixel,
        coin_flip,
        fg_biased,
        anchor,
        crop_shape,
        out_mask,
    )
    for iter in range(3):
        outputs = pipe.run()
        for idx in range(batch_size):
            in_mask = outputs[0].at(idx)
            fg_pixel1 = outputs[1].at(idx).tolist()
            fg_pixel2 = outputs[2].at(idx).tolist()
            fg_pixel3 = outputs[3].at(idx).tolist()
            rnd_pixel = outputs[4].at(idx).tolist()
            coin_flip = outputs[5].at(idx).tolist()
            fg_biased = outputs[6].at(idx).tolist()
            anchor = outputs[7].at(idx).tolist()
            crop_shape = outputs[8].at(idx).tolist()
            out_mask = outputs[9].at(idx)

            assert in_mask[tuple(fg_pixel1)] > 0
            assert in_mask[tuple(fg_pixel2)] > 0.99
            assert in_mask[tuple(fg_pixel3)] == 2
            assert in_mask[tuple(fg_biased)] > 0 or not coin_flip

            for d in range(ndim):
                assert 0 <= anchor[d] and anchor[d] + crop_shape[d] <= in_mask.shape[d]
            assert out_mask.shape == tuple(crop_shape)


def test_random_mask_pixel():
    for ndim in (2, 3):
        yield check_random_mask_pixel, ndim
