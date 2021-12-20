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

import math
import numpy as np
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
from test_utils import RandomDataIterator
import random
from nvidia.dali import math as dali_math

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
def ColorTwistPipeline(data_iterator, out_type):
    imgs = fn.external_source(source=data_iterator)
    imgs = dali_math.abs(imgs)
    H = fn.random.uniform(range=[-20, 20])
    S = fn.random.uniform(range=[0, 2])
    brightness = fn.random.uniform(range=[0, 2])
    contrast = fn.random.uniform(range=[0, 2])
    out_cpu = fn.color_twist(imgs, hue=H, saturation=S, brightness=brightness, contrast=contrast, dtype=out_type)
    out_gpu = fn.color_twist(imgs.gpu(), hue=H, saturation=S, brightness=brightness, contrast=contrast, dtype=out_type)
    return imgs, out_cpu, out_gpu, H, S, brightness, contrast

rgb2yiq = np.array([[.299,  .587,  .114],
                    [.596, -.274, -.321],
                    [.211, -.523,  .311]])

#yiq2rgb = np.linalg.inv(rgb2yiq)
yiq2rgb = np.array([[1, .956, .621],
                    [1, -.272, -.647],
                    [1, -1.107, 1.705]])

def ref_color_twist(img, H, S, brightness, contrast):
    if np.issubdtype(np.array(img).dtype, np.floating):
        grey = 0.5
        max_range = 1
    else:
        grey = 128
        max_range = 255
    angle = math.radians(H)
    s, c = math.sin(angle), math.cos(angle)
    # Rotate the color components by angle and scale by S.
    # The fun part is that it doesn't really matter that much which
    hmat = np.array([[1,    0,    0],
                     [0,  c*S,  s*S],
                     [0, -s*S,  c*S]])

    m = np.matmul(yiq2rgb, np.matmul(hmat, rgb2yiq))

    num_pixels = np.prod(img.shape[:-1])
    pixels = img.reshape([num_pixels, img.shape[-1]])
    pixels = np.matmul(pixels, m.transpose())
    pixels = ((pixels - grey) * contrast + grey) * brightness
    img = pixels.reshape(img.shape)

    return np.clip(img, 0, max_range)

def check(input, out_cpu, out_gpu, H, S, brightness, contrast):
    ref = ref_color_twist(input, H, S, brightness, contrast)
    print(ref, "vs", out_cpu)
    assert np.allclose(out_cpu, ref, 1/512, 0.55)
    assert np.allclose(out_gpu, ref, 1/512, 0.55)

def check_ref(inp_dtype, is_video):
    batch_size = 32
    n_iters = 8
    shape = (128, 32, 3) if not is_video else (random.randint(2, 5), 128, 32, 3)
    ri1 = RandomDataIterator(batch_size, shape=shape, dtype=dali_type_to_np(inp_dtype))
    pipe = ColorTwistPipeline(seed=2139, batch_size=batch_size, num_threads=4, device_id=0, data_iterator=ri1, out_type=inp_dtype)
    pipe.build()
    for _ in range(n_iters):
        inp, out_cpu, out_gpu, H, S, B, C = pipe.run()
        out_gpu = out_gpu.as_cpu()
        for i in range(batch_size):
            h, s, b, c = H.at(i), S.at(i), B.at(i), C.at(i)
            check(inp.at(i), out_cpu.at(i), out_gpu.at(i), h, s, b, c)

def test_color_twist():
    for inp_dtype in [types.FLOAT, types.INT16, types.UINT8]:
        is_video = random.choice([False, True])
        yield check_ref, inp_dtype, is_video
