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
from test_utils import get_dali_extra_path
from nvidia.dali.plugin.numba.fn.experimental import numba_function

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

def get_data_zeros(shapes, dtype):
    return [np.zeros(shape, dtype = dtype) for shape in shapes]

@pipeline_def
def numba_func_pipe(shapes, dtype, run_fn=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None, batch_processing=None):
    data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device = "cpu")
    return numba_function(data, run_fn=run_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn, batch_processing=batch_processing)

def _testimpl_numba_func(shapes, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out):
    batch_size = len(shapes)
    pipe = numba_func_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, dtype=dtype,
        run_fn=run_fn, setup_fn=setup_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, batch_processing=batch_processing)
    pipe.build()
    for _ in range(3):
        outs = pipe.run()
        for i in range(batch_size):
            out_arr = np.array(outs[0][i])
            assert np.array_equal(out_arr, expected_out[i])

def test_numba_func():
    # shape, dtype, run_fn, out_types, in_types, out_ndim, in_ndim, setup_fn, batch_processing, expected_out
    args = [
        ([(10, 10, 10)], np.uint8, set_all_values_to_255_batch, [dali_types.UINT8], [dali_types.UINT8], [3], [3], None, True, [np.full((10, 10, 10), 255, dtype=np.uint8)]),
        ([(10, 10, 10)], np.uint8, set_all_values_to_255_sample, [dali_types.UINT8], [dali_types.UINT8], [3], [3], None, None, [np.full((10, 10, 10), 255, dtype=np.uint8)]),
        ([(10, 10, 10)], np.float32, set_all_values_to_float_batch, [dali_types.FLOAT], [dali_types.FLOAT], [3], [3], None, True, [np.full((10, 10, 10), 0.5, dtype=np.float32)]),
        ([(10, 10, 10)], np.float32, set_all_values_to_float_sample, [dali_types.FLOAT], [dali_types.FLOAT], [3], [3], None, None, [np.full((10, 10, 10), 0.5, dtype=np.float32)]),
        ([(10, 20, 30), (20, 10, 30)], np.int64, change_out_shape_batch, [dali_types.INT64], [dali_types.INT64], [3], [3], setup_change_out_shape, True, [np.full((20, 30, 10), 42, dtype=np.int32), np.full((10, 30, 20), 42, dtype=np.int32)]),
        ([(10, 20, 30), (20, 10, 30)], np.int64, change_out_shape_sample, [dali_types.INT64], [dali_types.INT64], [3], [3], setup_change_out_shape, None, [np.full((20, 30, 10), 42, dtype=np.int32), np.full((10, 30, 20), 42, dtype=np.int32)]),
    ]

    for shape, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out in args:
        yield _testimpl_numba_func, shape, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out

@pipeline_def
def numba_func_image_pipe(run_fn=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None, batch_processing=None):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder)
    images_in = dali.fn.decoders.image(files, device="cpu")
    images_out = numba_function(images_in, run_fn=run_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn, batch_processing=batch_processing)
    return images_in, images_out

def _testimpl_numba_func_image(run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, transform):
    pipe = numba_func_image_pipe(batch_size=8, num_threads=3, device_id=0, run_fn=run_fn, setup_fn=setup_fn,
        out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, batch_processing=batch_processing)
    pipe.build()
    for _ in range(3):
        images_in, images_out = pipe.run()
        for i in range(len(images_in)):
            image_in_transformed = transform(images_in.at(i))
            assert np.array_equal(image_in_transformed, images_out.at(i))

def reverse_col_batch(out0, in0):
    for sample_id in range(len(out0)):
        out0[sample_id][:] = 255 - in0[sample_id][:]

def reverse_col_sample(out0, in0):
    out0[:] = 255 - in0[:]

def rot_image_batch(out0, in0):
    for out_sample, in_sample in zip(out0, in0):
        for i in range(out_sample.shape[0]):
            for j in range(out_sample.shape[1]):
                out_sample[i][j] = in_sample[j][out_sample.shape[0] - i - 1]

def rot_image_sample(out0, in0):
    for i in range(out0.shape[0]):
        for j in range(out0.shape[1]):
            out0[i][j] = in0[j][out0.shape[0] - i - 1]

def rot_image_setup(outs, ins):
    out0 = outs[0]
    in0 = ins[0]
    for sample_id in range(len(out0)):
        out0[sample_id][0] = in0[sample_id][1]
        out0[sample_id][1] = in0[sample_id][0]
        out0[sample_id][2] = in0[sample_id][2]

def test_numba_func_image():
    args = [
        (reverse_col_batch, [dali_types.UINT8], [dali_types.UINT8], [3], [3], None, True, lambda x: 255 - x),
        (reverse_col_sample, [dali_types.UINT8], [dali_types.UINT8], [3], [3], None, None, lambda x: 255 - x),
        (rot_image_batch, [dali_types.UINT8], [dali_types.UINT8], [3], [3], rot_image_setup, True, lambda x: np.rot90(x)),
        (rot_image_sample, [dali_types.UINT8], [dali_types.UINT8], [3], [3], rot_image_setup, None, lambda x: np.rot90(x)),
    ]
    for run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, transform in args:
        yield _testimpl_numba_func_image, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, transform

def split_images_col_sample(out0, out1, out2, in0):
    for i in range(in0.shape[0]):
        for j in range(in0.shape[1]):
            out0[i][j] = in0[i][j][0]
            out1[i][j] = in0[i][j][1]
            out2[i][j] = in0[i][j][2]

def setup_split_images_col(outs, ins):
    out0 = outs[0]
    out1 = outs[1]
    out2 = outs[2]
    for sample_id in range(len(out0)):
        out0[sample_id][0] = ins[0][sample_id][0]
        out0[sample_id][1] = ins[0][sample_id][1]
        out1[sample_id][0] = ins[0][sample_id][0]
        out1[sample_id][1] = ins[0][sample_id][1]
        out2[sample_id][0] = ins[0][sample_id][0]
        out2[sample_id][1] = ins[0][sample_id][1]

@pipeline_def
def numba_func_split_image_pipe(run_fn=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None, batch_processing=None):
    files, _ = dali.fn.readers.caffe(path=lmdb_folder)
    images_in = dali.fn.decoders.image(files, device="cpu")
    out0, out1, out2 = numba_function(images_in, run_fn=run_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn, batch_processing=batch_processing)
    return images_in, out0, out1, out2

def test_split_images_col():
    pipe = numba_func_split_image_pipe(batch_size=8, num_threads=1, device_id=0, run_fn=split_images_col_sample, setup_fn=setup_split_images_col,
        out_types=[dali_types.UINT8 for i in range(3)], in_types=[dali_types.UINT8], outs_ndim=[2, 2, 2], ins_ndim=[3])
    pipe.build()
    for _ in range(3):
        images_in, R, G, B = pipe.run()
        for i in range(len(images_in)):
            assert np.array_equal(images_in.at(i), np.stack([R.at(i), G.at(i), B.at(i)], axis=2))

def multiple_ins_setup(outs, ins):
    out0 = outs[0]
    in0 = ins[0]
    for sample_id in range(len(out0)):
        out0[sample_id][0] = in0[sample_id][0]
        out0[sample_id][1] = in0[sample_id][1]
        out0[sample_id][2] = 3

def multiple_ins_run(out0, in0, in1, in2):
    for i in range(out0.shape[0]):
        for j in range(out0.shape[1]):
            out0[i][j][0] = in0[i][j]
            out0[i][j][1] = in1[i][j]
            out0[i][j][2] = in2[i][j]

@pipeline_def
def numba_multiple_ins_pipe(shapes, dtype, run_fn=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None, batch_processing=None):
    data0 = fn.external_source(lambda: get_data_zeros(shapes, dtype), batch=True, device = "cpu")
    data1 = fn.external_source(lambda: get_data_zeros(shapes, dtype), batch=True, device = "cpu")
    data2 = fn.external_source(lambda: get_data_zeros(shapes, dtype), batch=True, device = "cpu")
    return numba_function(data0, data1, data2, run_fn=run_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn, batch_processing=batch_processing)

def test_multiple_ins():
    pipe = numba_multiple_ins_pipe(shapes=[(10, 10)], dtype=np.uint8, batch_size=8, num_threads=1, device_id=0, run_fn=multiple_ins_run, setup_fn=multiple_ins_setup,
        out_types=[dali_types.UINT8], in_types=[dali_types.UINT8 for i in range(3)], outs_ndim=[3], ins_ndim=[2, 2, 2])
    pipe.build()
    for _ in range(3):
        outs = pipe.run()
        out_arr = np.array(outs[0][0])
        assert np.array_equal(out_arr, np.zeros((10, 10, 3), dtype=np.uint8))
