# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import os
from numba import cfunc, types, carray, njit

from nvidia.dali import pipeline_def
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as dali_types
import nvidia.dali.numba_func as dali_numba
from test_utils import get_dali_extra_path
from nvidia.dali.plugin.numba.fn import numba_func

test_data_root = get_dali_extra_path()
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')

def set_all_values_to_255(*args):
    pass

@cfunc(dali_numba.run_fn_sig(types.float32, types.float32), nopython=True)
def set_all_values_to_float(out_ptr, out_shape_ptr, ndim_out, in_ptr, in_shape_ptr, ndim_in):
    out_shape = carray(out_shape_ptr, ndim_out)
    out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
    out_arr[:] = 0.5

dali_int32 = int(dali_types.INT32)
@cfunc(dali_numba.setup_fn_sig(1, 1), nopython=True)
def setup_change_out_shape(out_shape_ptr, out1_ndim, out_dtype, in_shape_ptr, in1_ndim, in_dtype, num_samples):
    in_shapes = carray(in_shape_ptr, (num_samples, in1_ndim))
    out_shapes = carray(out_shape_ptr, (num_samples, out1_ndim))
    perm = [1, 2, 0]
    for sample_idx in range(num_samples):
        for d in range(len(perm)):
            out_shapes[sample_idx][d] = in_shapes[sample_idx][perm[d]]
    out_type = carray(out_dtype, 1)
    out_type[0] = dali_int32

@cfunc(dali_numba.run_fn_sig(types.int32, types.int64), nopython=True)
def change_out_shape(out_ptr, out_shape_ptr, ndim_out, in_ptr, in_shape_ptr, ndim_in):
    out_shape = carray(out_shape_ptr, ndim_out)
    out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
    out_arr[:] = 42

def get_data(shapes, dtype):
    return [np.empty(shape, dtype = dtype) for shape in shapes]

@pipeline_def
def numba_func_pipe(shapes, dtype, fn_ptr=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None):
    data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device = "cpu")
    return numba_func(data, run_fn=fn_ptr, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn)

def _testimpl_numba_func(shapes, dtype, fn_ptr, out_types, in_types, outs_ndim, ins_ndim, setup_fn, expected_out):
    print("TU")
    batch_size = len(shapes)
    pipe = numba_func_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, dtype=dtype,
        fn_ptr=fn_ptr, setup_fn=setup_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim)
    pipe.build()
    for _ in range(1): # todo
        print("TU2")
        outs = pipe.run()
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            # assert np.array_equal(out_arr, expected_out[i])

def test_numba_func():
    # shape, dtype, func address, expected_out
    args = [
        # ([(10, 10, 10)], np.uint8, set_all_values_to_255.address, None, [np.full((10, 10, 10), 255, dtype=np.uint8)]),
        # ([(10, 10, 10)], np.float32, set_all_values_to_float.address, None, [np.full((10, 10, 10), 0.5, dtype=np.float32)]),
        # ([(10, 20, 30), (20, 10, 30)], np.int64, change_out_shape.address, setup_change_out_shape.address, [np.full((20, 30, 10), 42, dtype=np.int32), np.full((10, 30, 20), 42, dtype=np.int32)]),
        ([(10, 10, 10)], np.uint8, set_all_values_to_255, [dali_types.UINT8], [dali_types.UINT8], [3], [3], None, [np.full((10, 10, 10), 255, dtype=np.uint8)])
    ]

    for shape, dtype, fn_ptr, out_types, in_types, outs_ndim, ins_ndim, setup_fn, expected_out in args:
        yield _testimpl_numba_func, shape, dtype, fn_ptr, out_types, in_types, outs_ndim, ins_ndim, setup_fn, expected_out

@pipeline_def
def numba_func_image_pipe(fn_ptr=None, setup_fn=None):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder)
    images_in = dali.fn.decoders.image(files, device="cpu")
    images_out = dali.fn.experimental.numba_func(images_in, fn_ptr=fn_ptr, setup_fn=setup_fn)
    return images_in, images_out

def _testimpl_numba_func_image(fn_ptr, setup_fn, transform):
    pipe = numba_func_image_pipe(batch_size=8, num_threads=1, device_id=0, fn_ptr=fn_ptr, setup_fn=setup_fn)
    pipe.build()
    for _ in range(3):
        images_in, images_out = pipe.run()
        for i in range(len(images_in)):
            image_in_transformed = transform(images_in.at(i))
            assert np.array_equal(image_in_transformed, images_out.at(i))

@cfunc(dali_numba.run_fn_sig(types.uint8, types.uint8), nopython=True)
def reverse_col(out_ptr, out_shape_ptr, ndim_out, in_ptr, in_shape_ptr, ndim_in):
    out_shape = carray(out_shape_ptr, ndim_out)
    in_shape = carray(in_shape_ptr, ndim_in)
    in_arr = carray(in_ptr, (in_shape[0], in_shape[1], in_shape[2]))
    out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
    out_arr[:] = 255 - in_arr[:]

@cfunc(dali_numba.run_fn_sig(types.uint8, types.uint8), nopython=False)
def rot_image(out_ptr, out_shape_ptr, ndim_out, in_ptr, in_shape_ptr, ndim_in):
    out_shape = carray(out_shape_ptr, ndim_out)
    in_shape = carray(in_shape_ptr, ndim_in)
    in_arr = carray(in_ptr, (in_shape[0], in_shape[1], in_shape[2]))
    out_arr = carray(out_ptr, (out_shape[0], out_shape[1], out_shape[2]))
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            out_arr[i][j] = in_arr[j][out_shape[0] - i - 1]

@cfunc(dali_numba.setup_fn_sig(1, 1), nopython=True)
def rot_image_setup(out_shape_ptr, out1_ndim, out_dtype, in_shape_ptr, in1_ndim, in_dtype, num_samples):
    in_arr = carray(in_shape_ptr, num_samples * out1_ndim)
    out_arr = carray(out_shape_ptr, num_samples * in1_ndim)
    out_type = carray(out_dtype, 1)
    out_type[0] = in_dtype
    for i in range(0, num_samples * out1_ndim, 3):
        out_arr[i] = in_arr[i + 1]
        out_arr[i + 1] = in_arr[i]
        out_arr[i + 2] = in_arr[i + 2]

def test_numba_func_image():
    args = [
        # (reverse_col.address, None, lambda x: 255 - x),
        # (rot_image.address, rot_image_setup.address, lambda x: np.rot90(x)),
    ]
    for fn_ptr, setup_fn, transform in args:
        yield _testimpl_numba_func_image, fn_ptr, setup_fn, transform
