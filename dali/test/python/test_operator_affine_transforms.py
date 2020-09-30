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

from scipy.spatial.transform import Rotation as scipy_rotate

def check_results(T1, batch_size, mat_ref, T0=None, reverse=False, rtol=1e-7):
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
            print("ref: ", ref_T1)
            print("actual: ", T1.at(idx))
            
            assert np.allclose(T1.at(idx), ref_T1, rtol=rtol)
    else:
        ref_T1 = mat_ref[:ndim, :]
        for idx in range(batch_size):
            print("ref: ", ref_T1)
            print("actual: ", T1.at(idx))
            assert np.allclose(T1.at(idx), ref_T1, rtol=rtol)

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
    assert center is None or len(center) == ndim

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
    assert center is None or len(center) == ndim

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

def rotate_affine_mat(angle, axis = None, center = None):
    assert axis is None or len(axis) == 3
    ndim = 3 if axis is not None else 2
    assert center is None or len(center) == ndim

    angle_rad = angle * np.pi / 180.0
    if ndim == 2:
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        r_mat = np.array(
            [[  c, -s,  0.],
             [  s,  c,  0.],
             [  0., 0., 1.]])
    else:  # ndim == 3
        norm_axis = axis / np.linalg.norm(axis)        
        r_mat = np.identity(ndim + 1)
        r_mat[:ndim, :ndim] = scipy_rotate.from_rotvec(angle_rad * norm_axis).as_matrix()
    if center is not None:
        neg_offset = tuple([-x for x in center])
        t1_mat = translate_affine_mat(neg_offset)
        t2_mat = translate_affine_mat(center)
        affine_mat = np.dot(t2_mat, np.dot(r_mat, t1_mat))
    else:
        affine_mat = r_mat

    return affine_mat

def check_rotate_transform_op(angle, axis=None, center=None, has_input = False, reverse_order=False, batch_size=1, num_threads=4, device_id=0):
    assert axis is None or len(axis) == 3
    ndim = 3 if axis is not None else 2
    assert center is None or len(center) == ndim

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        if has_input:
            T0 = fn.uniform(range=(-1, 1), shape=(ndim, ndim+1), seed = 1234)
            T1 = fn.rotate_transform(T0, device='cpu', angle=angle, axis=axis, center=center, reverse_order=reverse_order)
            pipe.set_outputs(T1, T0)
        else:
            T1 = fn.rotate_transform(device='cpu', angle=angle, axis=axis, center=center)
            pipe.set_outputs(T1)
    pipe.build()
    outs = pipe.run()
    ref_mat = rotate_affine_mat(angle=angle, axis=axis, center=center)
    T0 = outs[1] if has_input else None
    check_results(outs[0], batch_size, ref_mat, T0, reverse_order, rtol=1e-5)

def test_rotate_transform_op(batch_size=3, num_threads=4, device_id=0):
    for angle, axis, center in [(30.0, None, None),
                                (30.0, None, (1.0, 0.5)),
                                (40.0, (0.4, 0.3, 0.1), None),
                                (40.0, (0.4, 0.3, 0.1), (1.0, -0.4, 10.0))]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield check_rotate_transform_op, angle, axis, center, has_input, reverse_order, \
                                                 batch_size, num_threads, device_id

def shear_affine_mat(shear = None, angles = None, center = None):
    assert shear is not None or angles is not None
    if shear is None:
        shear = [np.tan(a * np.pi / 180.0) for a in angles]
    assert len(shear) == 2 or len(shear) == 6
    ndim = 3 if len(shear) == 6 else 2 
    assert center is None or len(center) == ndim

    if ndim == 2:
        sxy, syx = shear
        s_mat = np.array(
            [[  1. , sxy,  0.],
             [  syx,  1.,  0.],
             [  0. ,  0., 1.]])
    else:  # ndim == 3
        sxy, sxz, syx, syz, szx, szy = shear
        s_mat = np.array(
            [[  1  , sxy, sxz, 0 ],
             [  syx,   1, syz, 0 ],
             [  szx, szy,   1, 0 ],
             [    0,   0,   0, 1 ]])

    if center is not None:
        neg_offset = tuple([-x for x in center])
        t1_mat = translate_affine_mat(neg_offset)
        t2_mat = translate_affine_mat(center)
        affine_mat = np.dot(t2_mat, np.dot(s_mat, t1_mat))
    else:
        affine_mat = s_mat

    return affine_mat

def check_shear_transform_op(shear=None, angles=None, center=None, has_input = False, reverse_order=False, batch_size=1, num_threads=4, device_id=0):
    assert shear is not None or angles is not None
    if shear is not None:
        assert len(shear) == 2 or len(shear) == 6
        ndim = 3 if len(shear) == 6 else 2 
    else:
        assert len(angles) == 2 or len(angles) == 6
        ndim = 3 if len(angles) == 6 else 2 
    assert center is None or len(center) == ndim

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        if has_input:
            T0 = fn.uniform(range=(-1, 1), shape=(ndim, ndim+1), seed = 1234)
            T1 = fn.shear_transform(T0, device='cpu', shear=shear, angles=angles, center=center, reverse_order=reverse_order)
            pipe.set_outputs(T1, T0)
        else:
            T1 = fn.shear_transform(device='cpu', shear=shear, angles=angles, center=center)
            pipe.set_outputs(T1)
    pipe.build()
    outs = pipe.run()
    ref_mat = shear_affine_mat(shear=shear, angles=angles, center=center)
    T0 = outs[1] if has_input else None
    check_results(outs[0], batch_size, ref_mat, T0, reverse_order, rtol=1e-5)

def test_shear_transform_op(batch_size=3, num_threads=4, device_id=0):
    for shear, angles, center in [((1., 2.), None, None),
                                  ((1., 2.), None, (0.4, 0.5)),
                                  ((1., 2., 3., 4., 5., 6.), None, None),
                                  ((1., 2., 3., 4., 5., 6.), None, (0.4, 0.5, 0.6)),
                                  (None, (30., 10.), None),
                                  (None, (30., 10.), (0.4, 0.5)),
                                  (None, (40., 30., 10., 35., 25., 15.), None),
                                  (None, (40., 30., 10., 35., 25., 15.), (0.4, 0.5, 0.6))]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield check_shear_transform_op, shear, angles, center, has_input, reverse_order, \
                                                batch_size, num_threads, device_id
