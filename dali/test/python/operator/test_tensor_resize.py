# Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.types as types
from nvidia.dali import pipeline_def, fn

def tensor_resize_last_dim_unchanged():
    ndim = 2
    batch_size = 3
    shape = (4, 4, 3)
    out_shape = (3, 5, 3)
    in_dtype = types.UINT8
    out_dtype = types.UINT8
    device = 'cpu'

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
    def pipe():
        data = fn.random.uniform(range=(0, 255), shape=shape, dtype=in_dtype)
        return data, fn.experimental.tensor_resize(data, sizes=out_shape)

    p = pipe()
    p.build()
    input_data, output_data = p.run()
    print(input_data)
    print(output_data)

def tensor_resize_last_dim_resized():
    ndim = 2
    batch_size = 3
    shape = (4, 4, 3)
    out_shape = (3, 5, 2)
    in_dtype = types.UINT8
    out_dtype = types.FLOAT
    device = 'cpu'

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
    def pipe():
        data = fn.random.uniform(range=(0, 255), shape=shape, dtype=in_dtype)
        return data, fn.experimental.tensor_resize(data, dtype=out_dtype, sizes=out_shape)

    p = pipe()
    p.build()
    input_data, output_data = p.run()
    print(input_data)
    print(output_data)

def tensor_resize_scales_last_dim_unchanged():
    ndim = 2
    batch_size = 3
    shape = (4, 4, 3)
    scales = (0.7, 2.0, 1.0)
    in_dtype = types.UINT8
    out_dtype = types.FLOAT
    device = 'cpu'

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
    def pipe():
        data = fn.random.uniform(range=(0, 255), shape=shape, dtype=in_dtype)
        return data, fn.experimental.tensor_resize(data, dtype=out_dtype, scales=scales, scales_rounding='round')

    p = pipe()
    p.build()
    input_data, output_data = p.run()
    print(input_data)
    print(output_data)


def tensor_resize_scales_axes_last_dim_unchanged():
    ndim = 2
    batch_size = 1
    shape = (4, 4, 3)
    scales = (2.0, 0.7)
    axes = (1, 0)
    in_dtype = types.UINT8
    out_dtype = types.FLOAT
    device = 'cpu'

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
    def pipe():
        data = fn.random.uniform(range=(0, 255), shape=shape, dtype=in_dtype)
        return data, fn.experimental.tensor_resize(data, dtype=out_dtype, axes=axes, scales=scales, scales_rounding='round')

    p = pipe()
    p.build()
    input_data, output_data = p.run()
    print(input_data)
    print(output_data)

tensor_resize_scales_axes_last_dim_unchanged()
