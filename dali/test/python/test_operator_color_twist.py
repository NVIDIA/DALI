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

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
import numpy as np
from test_utils import check_batch
from test_utils import RandomDataIterator
import random
from skimage.color import rgb2hsv


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

def max_range(dtype):
    if dtype == np.half or dtype == np.single or dtype == np.double:
        return 1.0
    else:
        return np.iinfo(dtype).max

def type_range(dtype):
    if dtype in [np.half, np.single, np.double]:
        return (np.finfo(dtype).min, np.finfo(dtype).max)
    else:
        return (np.iinfo(dtype).min, np.iinfo(dtype).max)

def convert_sat(data, out_dtype):
    clipped = np.clip(data, *type_range(out_dtype))
    if out_dtype not in [np.half, np.single, np.double]:
        clipped = np.round(clipped)
    return clipped.astype(out_dtype)

def color_twist_ref(input, hue, sat, bri, con, out_dtype):
    # input = bri * con * input
    # hsv_img = rgb2hsv(input)
    # hue_img = hsv_img[:, :, 0]
    # value_img = hsv_img[:, :, 2]

    if np.issubdtype(input.dtype, np.floating):
        contrast_center = 0.5
    else:
        contrast_center = 128
    output_range = max_range(out_dtype)
    output = bri * (contrast_center + con * (input - contrast_center))
    return convert_sat(output, out_dtype)

def ref_operator(out_dtype):
  return lambda input, hue, sat, bri, con: \
                color_twist_ref(input, hue, sat, bri,con, out_dtype)

@pipeline_def(num_threads=4, device_id=0, seed=1234, exec_pipelined=False, exec_async=False)
def color_twist_pipe(data_iterator, out_dtype, device):
    inp = fn.external_source(source=data_iterator)
    # hue = fn.random.uniform(range=[-20., 20.])
    # sat = fn.random.uniform(range=[0., 1.])
    hue = 0
    sat = 1
    bri = 1
    # bri = fn.random.uniform(range=[0., 2.])
    con = fn.random.uniform(range=[0., 2.])
    twist_inp = inp if device == 'cpu' else inp.gpu()
    ret = fn.color_twist(twist_inp, device=device, hue=hue, saturation=sat, brightness=bri, contrast=con)
    ref = fn.python_function(inp, hue, sat, bri ,con, function=ref_operator(dali_type_to_np(out_dtype)))
    ref = fn.reshape(ref, layout="HWC")
    return ref, ret

def check_vs_ref(device, inp_dtype, out_dtype, is_video):
    batch_size=32
    n_iters = 8
    shape = (128, 32, 3) if not is_video else (random.randint(2, 5), 128, 32, 3)
    ri1 = RandomDataIterator(batch_size, shape=shape, dtype=dali_type_to_np(inp_dtype))
    pipe = color_twist_pipe(ri1, out_dtype, device, batch_size=batch_size)
    if out_dtype in [np.half, np.single, np.double]:
        eps = 1e-4
    else:
        eps = 1
    pipe.build()
    for _ in range(n_iters):
        out  = pipe.run()
        check_batch(out[0], out[1], eps=1)

def test_vs_ref():
    for device in ['cpu', 'gpu']:
        for inp_dtype in [types.FLOAT, types.INT16, types.UINT8]:
            for out_dtype in [types.FLOAT, types.INT16, types.UINT8]:
                is_video = random.choice([False])
                yield check_vs_ref, device, inp_dtype, out_dtype, is_video
