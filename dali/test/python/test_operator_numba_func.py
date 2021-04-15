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

def set_all_values_to_255_batch(out0, in0):
    out0[0][:] = 255

def set_all_values_to_255_sample(out0, in0):
    out0[:] = 255

def set_all_values_to_float_batch(out0, in0):
    out0[0][:] = 0.5

def set_all_values_to_float_sample(out0, in0):
    out0[:] = 0.5

def setup_change_out_shape(out_shape, in_shape):
    out0_shape = out_shape[0]
    in0_shape = in_shape[0]
    perm = [1, 2, 0]
    for sample_idx in range(len(out0_shape)):
        for d in range(len(perm)):
            out0_shape[sample_idx][d] = in0_shape[sample_idx][perm[d]]

def change_out_shape_batch(out0, in0):
    for sample_id in range(len(out0)):
        out0[sample_id][:] = 42

def change_out_shape_sample(out0, in0):
    out0[:] = 42

def get_data(shapes, dtype):
    return [np.empty(shape, dtype = dtype) for shape in shapes]

@pipeline_def
def numba_func_pipe(shapes, dtype, run_fn=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None, batch_processing=None):
    data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device = "cpu")
    return numba_func(data, run_fn=run_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn, batch_processing=batch_processing)

def _testimpl_numba_func(shapes, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out):
    batch_size = len(shapes)
    pipe = numba_func_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, dtype=dtype,
        run_fn=run_fn, setup_fn=setup_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, batch_processing=batch_processing)
    pipe.build()
    for _ in range(3): # todo
        outs = pipe.run()
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            assert np.array_equal(out_arr, expected_out[i])

def test_numba_func():
    # shape, dtype, run_fn, out_types, in_types, out_ndim, in_ndim, setup_fn, batch_processing, expected_out
    args = [
        # ([(10, 10, 10)], np.uint8, set_all_values_to_255_batch, [dali_types.UINT8], [dali_types.UINT8], [3], [3], None, True, [np.full((10, 10, 10), 255, dtype=np.uint8)]),
        # ([(10, 10, 10)], np.uint8, set_all_values_to_255_sample, [dali_types.UINT8], [dali_types.UINT8], [3], [3], None, None, [np.full((10, 10, 10), 255, dtype=np.uint8)]),
        # ([(10, 10, 10)], np.float32, set_all_values_to_float_batch, [dali_types.FLOAT], [dali_types.FLOAT], [3], [3], None, True, [np.full((10, 10, 10), 0.5, dtype=np.float32)]),
        # ([(10, 10, 10)], np.float32, set_all_values_to_float_sample, [dali_types.FLOAT], [dali_types.FLOAT], [3], [3], None, None, [np.full((10, 10, 10), 0.5, dtype=np.float32)]),
        # ([(10, 20, 30), (20, 10, 30)], np.int64, change_out_shape_batch, [dali_types.INT64], [dali_types.INT64], [3], [3], setup_change_out_shape, True, [np.full((20, 30, 10), 42, dtype=np.int32), np.full((10, 30, 20), 42, dtype=np.int32)]),
        # ([(10, 20, 30), (20, 10, 30)], np.int64, change_out_shape_sample, [dali_types.INT64], [dali_types.INT64], [3], [3], setup_change_out_shape, None, [np.full((20, 30, 10), 42, dtype=np.int32), np.full((10, 30, 20), 42, dtype=np.int32)]),
    ]

    for shape, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out in args:
        yield _testimpl_numba_func, shape, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out

@pipeline_def
def numba_func_image_pipe(run_fn=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None, batch_processing=None):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder)
    images_in = dali.fn.decoders.image(files, device="cpu")
    images_out = numba_func(images_in, run_fn=run_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn, batch_processing=batch_processing)
    return images_in, images_out

def _testimpl_numba_func_image(run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, transform):
    pipe = numba_func_image_pipe(batch_size=8, num_threads=1, device_id=0, run_fn=run_fn, setup_fn=setup_fn,
        out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, batch_processing=batch_processing)
    pipe.build()
    for _ in range(1):
        images_in, images_out = pipe.run()
        for i in range(len(images_in)):
            print(i)
            image_in_transformed = transform(images_in.at(i))
            # assert np.array_equal(image_in_transformed, images_out.at(i))

def reverse_col_batch(out0, in0):
    # out0[0][:] = 255 - in0[0][:]
    # out0[1][:] = 255 - in0[1][:]
    # out0[2][:] = 255 - in0[2][:]
    # out0[3][:] = 255 - in0[3][:]
    # out0[4][:] = 255 - in0[4][:]
    # out0[5][:] = 255 - in0[5][:]
    # out0[6][:] = 255 - in0[6][:]
    out0[7][:] = 255 - in0[7][:]

    # for sample_id in range(len(out0)):
    #     out0[sample_id][:] = 255 - in0[sample_id][:]

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
        (reverse_col_batch, [dali_types.UINT8], [dali_types.UINT8], [3], [3], None, True, lambda x: 255 - x),
        # (rot_image.address, rot_image_setup.address, lambda x: np.rot90(x)),
    ]
    for run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, transform in args:
        yield _testimpl_numba_func_image, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, transform
