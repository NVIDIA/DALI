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

import math
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import random
from nvidia.dali import pipeline_def

from sequences_test_utils import ArgCb, video_suite_helper
from test_utils import RandomDataIterator


def dali_type_to_np(dtype):
    if dtype == types.FLOAT:
        return np.single
    elif dtype == types.INT16:
        return np.short
    elif dtype == types.INT32:
        return np.intc
    elif dtype == types.UINT8:
        return np.ubyte
    else:
        assert False


@pipeline_def()
def ColorTwistPipeline(data_iterator, is_input_float, inp_dtype, out_dtype):
    imgs = fn.external_source(source=data_iterator)
    o_dtype = dali_type_to_np(out_dtype)
    # converting float inputs to integer outs leads to binary images as
    # input is in -1 to 1 range in such case
    if is_input_float and not np.issubdtype(o_dtype, np.floating):
        imgs *= 255
    H = fn.random.uniform(range=[-20, 20])
    S = fn.random.uniform(range=[0, 2])
    brightness = fn.random.uniform(range=[0, 2])
    contrast = fn.random.uniform(range=[0, 2])

    out_dtype_arg = out_dtype if out_dtype != inp_dtype else None
    out_cpu, out_gpu = (
        fn.color_twist(
            input,
            hue=H,
            saturation=S,
            brightness=brightness,
            contrast=contrast,
            dtype=out_dtype_arg,
        )
        for input in (imgs, imgs.gpu())
    )
    return imgs, out_cpu, out_gpu, H, S, brightness, contrast


rgb2yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]])

yiq2rgb = np.linalg.inv(rgb2yiq)


def convert_sat(data, out_dtype):
    clipped = data
    if not np.issubdtype(out_dtype, np.floating):
        max_range = np.iinfo(out_dtype).max
        min_range = np.iinfo(out_dtype).min
        clipped = np.clip(clipped, min_range, max_range)
        clipped = np.round(clipped)
    return clipped.astype(out_dtype)


def ref_color_twist(img, H, S, brightness, contrast, out_dtype):
    inp_dtype = img.dtype
    angle = math.radians(H)
    s, c = math.sin(angle), math.cos(angle)
    # Rotate the color components by angle and scale by S.
    # The fun part is that it doesn't really matter that much which
    hmat = np.array([[1, 0, 0], [0, c * S, s * S], [0, -s * S, c * S]])

    m = np.matmul(yiq2rgb, np.matmul(hmat, rgb2yiq))

    num_pixels = np.prod(img.shape[:-1])
    pixels = img.reshape([num_pixels, img.shape[-1]])
    pixels = np.matmul(pixels, m.transpose())

    if np.issubdtype(inp_dtype, np.floating):
        grey = 0.5
    else:
        grey = 128
    pixels = ((pixels - grey) * contrast + grey) * brightness
    img = pixels.reshape(img.shape)

    return convert_sat(img, out_dtype)


def check(input, out_cpu, out_gpu, H, S, brightness, contrast, out_dtype):
    ref = ref_color_twist(input, H, S, brightness, contrast, out_dtype)
    if np.issubdtype(out_dtype, np.floating):
        rel_err = 1e-3
        abs_err = 1e-3
    else:
        rel_err = 1 / 512
        # due to rounding error for integer out type can be off by 1
        abs_err = 1

    assert np.allclose(out_cpu, ref, rel_err, abs_err)
    assert np.allclose(out_gpu, ref, rel_err, abs_err)


def check_ref(inp_dtype, out_dtype, has_3_dims):
    batch_size = 32
    n_iters = 8
    shape = (128, 32, 3) if not has_3_dims else (random.randint(2, 5), 128, 32, 3)
    inp_dtype = dali_type_to_np(inp_dtype)
    ri1 = RandomDataIterator(batch_size, shape=shape, dtype=inp_dtype)
    pipe = ColorTwistPipeline(
        seed=2139,
        batch_size=batch_size,
        num_threads=4,
        device_id=0,
        data_iterator=ri1,
        is_input_float=np.issubdtype(inp_dtype, np.floating),
        inp_dtype=inp_dtype,
        out_dtype=out_dtype,
    )
    for _ in range(n_iters):
        inp, out_cpu, out_gpu, H, S, B, C = pipe.run()
        out_gpu = out_gpu.as_cpu()
        for i in range(batch_size):
            h, s, b, c = H.at(i), S.at(i), B.at(i), C.at(i)
            check(inp.at(i), out_cpu.at(i), out_gpu.at(i), h, s, b, c, dali_type_to_np(out_dtype))


def test_color_twist():
    for inp_dtype in [types.FLOAT, types.INT16, types.UINT8]:
        for out_dtype in [types.FLOAT, types.INT16, types.UINT8]:
            has_3_dims = random.choice([False, True])
            yield check_ref, inp_dtype, out_dtype, has_3_dims


def test_video():
    def hue(sample_desc):
        return np.float32(360 * sample_desc.rng.random())

    def saturation(sample_desc):
        return np.float32(sample_desc.rng.random())

    def value(sample_desc):
        return np.float32(sample_desc.rng.random())

    def contrast(sample_desc):
        return np.float32(2 * sample_desc.rng.random())

    def brightness(sample_desc):
        return np.float32(2 * sample_desc.rng.random())

    video_test_cases = [
        (fn.hue, {}, [ArgCb("hue", hue, True)]),
        (fn.saturation, {}, [ArgCb("saturation", saturation, True)]),
        (
            fn.hsv,
            {},
            [
                ArgCb("hue", hue, True),
                ArgCb("saturation", saturation, True),
                ArgCb("value", value, True),
            ],
        ),
        (
            fn.hsv,
            {},
            [
                ArgCb("hue", hue, False),
                ArgCb("saturation", saturation, True),
                ArgCb("value", value, False),
            ],
        ),
        (
            fn.color_twist,
            {},
            [
                ArgCb("brightness", brightness, True),
                ArgCb("hue", hue, True),
                ArgCb("saturation", saturation, True),
                ArgCb("contrast", contrast, True),
            ],
        ),
        (fn.color_twist, {}, [ArgCb("brightness", brightness, True), ArgCb("hue", hue, False)]),
    ]

    yield from video_suite_helper(video_test_cases, test_channel_first=False)


def test_color_twist_default_dtype():
    np_types = [types.FLOAT, types.INT32, types.INT16, types.UINT8]  # Just some types

    def impl(op, device, type):
        @pipeline_def(batch_size=1, num_threads=3, device_id=0)
        def pipeline():
            data = types.Constant(255, shape=(10, 10, 3), dtype=type, device=device)
            return op(data)

        pipe = pipeline()
        (data,) = pipe.run()
        assert data[0].dtype == type, f"{data[0].dtype} != {type}"

    for device in ["gpu", "cpu"]:
        for type in np_types:
            for op in [fn.hue]:
                yield impl, op, device, type
