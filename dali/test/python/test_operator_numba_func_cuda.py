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
from numba import cfunc, types, carray, njit, cuda

from nvidia.dali import pipeline_def
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as dali_types
from test_utils import get_dali_extra_path
from nvidia.dali.plugin.numba.fn.experimental import numba_function, numba_function_cuda

test_data_root = get_dali_extra_path()
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')

# @cuda.jit
# def set_all_values_to_255_sample(io_array):
#     x, y = cuda.grid(2)
#     if x < io_array.shape[0] and y < io_array.shape[1]:
#         io_array[x][y] = 255
def set_all_values_to_255_sample():
    x, y = cuda.grid(2)


def get_data(shapes, dtype):
    return [np.ones(shape, dtype = dtype) for shape in shapes]

@pipeline_def
def numba_func_pipe(shapes, dtype, run_fn=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None, batch_processing=None):
    data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device = "cpu")
    return numba_function(data, run_fn=run_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn, batch_processing=batch_processing, device='cpu')

@pipeline_def
def numba_func_cuda_pipe(shapes, dtype, run_fn=None, out_types=None, in_types=None, outs_ndim=None, ins_ndim=None, setup_fn=None, batch_processing=None):
    data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device = "gpu")
    return numba_function_cuda(data, run_fn=run_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, setup_fn=setup_fn, batch_processing=batch_processing, device='gpu')


def _testimpl_numba_func(shapes, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out):
    batch_size = len(shapes)
    pipe = numba_func_cuda_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes, dtype=dtype,
        run_fn=run_fn, setup_fn=setup_fn, out_types=out_types, in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim, batch_processing=batch_processing)
    pipe.build()
    for _ in range(3):
        outs = pipe.run()
        for i in range(batch_size):
            out_arr = np.array(outs[0][i].as_cpu())
            assert np.array_equal(out_arr, expected_out[i])

def test_numba_func():
    # shape, dtype, run_fn, out_types, in_types, out_ndim, in_ndim, setup_fn, batch_processing, expected_out
    args = [
        ([(10, 10)], np.uint8, set_all_values_to_255_sample, [dali_types.UINT8], [dali_types.UINT8], [2], [2], None, True, [np.full((10, 10), 255, dtype=np.uint8)]),
    ]

    for shape, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out in args:
        # yield _testimpl_numba_func, shape, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out
        _testimpl_numba_func(shape, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, setup_fn, batch_processing, expected_out)

test_numba_func()
# next(gen)