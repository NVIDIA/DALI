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

def check_results(T1, batch_size, mat_ref, T0=None, reverse=False):
    ndim = mat_ref.shape[0] - 1
    if T0 is not None:
        for idx in range(batch_size):
            mat_T0 = np.identity(ndim+1)
            mat_T0[:ndim, :] = T0.at(idx)
            if reverse:
                mat_T1 = np.dot(mat_T0, mat_ref)
            else:
                mat_T1 = np.dot(mat_ref, mat_T0)
            ref_T1 = mat_T1[:ndim, :]
            assert np.allclose(T1.at(idx), ref_T1, rtol=1e-7)
    else:
        ref_T1 = mat_ref[:ndim, :]
        for idx in range(batch_size):
            assert np.allclose(T1.at(idx), ref_T1, rtol=1e-7)

def translate_affine_mat(offset):
    ndim = len(offset)
    affine_mat = np.identity(ndim + 1)
    affine_mat[:ndim, -1] = offset
    return affine_mat

def check_translate_transform_op(offset, has_input = False, reverse_order=False, batch_size=1, num_threads=4, device_id=0):
    ndim = len(offset)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        if has_input:
            T0 = fn.uniform(range=(-1, 1), shape=(ndim, ndim+1), seed = 1234)
            T1 = fn.translate_transform(T0, device='cpu', offset=offset, reverse_order=reverse_order)
            pipe.set_outputs(T1, T0)
        else:
            T1 = fn.translate_transform(device='cpu', offset=offset)
            pipe.set_outputs(T1)
    pipe.build()
    outs = pipe.run()
    ref_mat = translate_affine_mat(offset=offset)
    T0 = outs[1] if has_input else None
    check_results(outs[0], batch_size, ref_mat, T0, reverse_order)

def test_translate_transform_op(batch_size=3, num_threads=4, device_id=0):
    for offset in [(0.0, 1.0), (2.0, 1.0, 3.0)]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield check_translate_transform_op, offset, has_input, reverse_order, \
                                                    batch_size, num_threads, device_id

def scale_affine_mat(scale, center = None):
    ndim = len(scale)

    s_mat = np.identity(ndim + 1)
    for d in range(ndim):
        s_mat[d, d] = scale[d]

    if center is not None:
        neg_offset = tuple([-x for x in center])
        t1_mat = translate_affine_mat(neg_offset)
        t2_mat = translate_affine_mat(center)
        affine_mat = np.dot(t2_mat, np.dot(s_mat, t1_mat))
    else:
        affine_mat = s_mat

    return affine_mat

def check_scale_transform_op(scale, center=None, has_input = False, reverse_order=False, batch_size=1, num_threads=4, device_id=0):
    ndim = len(scale)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        if has_input:
            T0 = fn.uniform(range=(-1, 1), shape=(ndim, ndim+1), seed = 1234)
            T1 = fn.scale_transform(T0, device='cpu', scale=scale, center=center, reverse_order=reverse_order)
            pipe.set_outputs(T1, T0)
        else:
            T1 = fn.scale_transform(device='cpu', scale=scale, center=center)
            pipe.set_outputs(T1)
    pipe.build()
    outs = pipe.run()
    ref_mat = scale_affine_mat(scale=scale, center=center)
    T0 = outs[1] if has_input else None
    check_results(outs[0], batch_size, ref_mat, T0, reverse_order)

def test_scale_transform_op(batch_size=3, num_threads=4, device_id=0):
    for scale, center in [((0.0, 1.0), None),
                          ((2.0, 1.0, 3.0), None),
                          ((2.0, 1.0), (1.0, 0.5))]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield check_scale_transform_op, scale, center, has_input, reverse_order, \
                                                batch_size, num_threads, device_id
