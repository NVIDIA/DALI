# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import numpy as np
import os

def translate_affine_mat(offset):
    ndim = len(offset)
    affine_mat = np.identity(ndim + 1)
    affine_mat[:ndim, -1] = offset
    return affine_mat

def check_translate_transform_op(offset, T0_data = None, batch_size=1, num_threads=4, device_id=0):
    ndim = len(offset)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        outs = []
        if T0_data is None:
            T1 = fn.translate_transform(device='cpu', offset=offset)
            outs = [T1]
        else:
            T0 = types.Constant(device = 'cpu', value = T0_data, dtype=types.FLOAT)
            T1 = fn.translate_transform(T0, device='cpu', offset=offset)
            outs = [T1, T0]
        pipe.set_outputs(*outs)
    pipe.build()
    outs = pipe.run()

    ref_mat = translate_affine_mat(offset=offset)
    ref_MT = ref_mat[:ndim, :]

    if T0_data is not None:
        mat_T0 = np.identity(ndim+1)
        mat_T0[:ndim, :] = T0_data

        mat_T1 = np.dot(ref_mat, mat_T0)
        ref_T1 = mat_T1[:ndim, :]

        for idx in range(batch_size):
            T1 = outs[0].at(idx)
            T0 = outs[1].at(idx)
            assert np.allclose(T1, ref_T1, rtol=1e-7)
    else:
        ref_T1 = ref_mat[:ndim, :]
        for idx in range(batch_size):
            T1 = outs[0].at(idx)
            assert np.allclose(T1, ref_T1, rtol=1e-7)

def test_translate_transform_op(batch_size=1, num_threads=4, device_id=0):
    for offset in [(0.0, 1.0), (2.0, 1.0, 3.0)]:
        yield check_translate_transform_op, offset, None, batch_size, num_threads, device_id
        T0_data = np.array(np.random.rand(len(offset), len(offset) + 1), dtype = np.float32)
        yield check_translate_transform_op, offset, T0_data, batch_size, num_threads, device_id
