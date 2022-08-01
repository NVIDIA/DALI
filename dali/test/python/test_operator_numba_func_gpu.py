# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import numba
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as dali_types
import os
from distutils.version import LooseVersion
from nose import SkipTest, with_setup
from numba import cuda
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.numba.fn.experimental import numba_function

from nose_utils import raises
from test_utils import get_dali_extra_path


def check_env_compatibility():
    if LooseVersion(numba.__version__) < LooseVersion('0.55.2'):
        raise SkipTest()
    if cuda.runtime.get_version() > cuda.driver.driver.get_version():
        raise SkipTest()


test_data_root = get_dali_extra_path()
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')


def set_all_values_to_255_sample(in_arr, out_arr):
    x, y = cuda.grid(2)
    if x < out_arr.shape[0] and y < out_arr.shape[1]:
        out_arr[x][y] = 255


def set_output_to_input_plus_5_sample(inp, out):
    x, y = cuda.grid(2)
    if x < inp.shape[0] and y < inp.shape[1] and x < out.shape[0] and y < out.shape[1]:
        out[x][y] = inp[x][y] + 5


def set_consecutive_values_sample(inp, out):
    x, y = cuda.grid(2)
    if x < inp.shape[0] and y < inp.shape[1] and x < out.shape[0] and y < out.shape[1]:
        out[x][y] = y + x * inp.shape[1]


def get_data(shapes, dtype):
    return [np.ones(shape, dtype=dtype) for shape in shapes]


@pipeline_def
def numba_func_cuda_pipe(shapes, dtype, run_fn=None,
                         out_types=None, in_types=None, outs_ndim=None, ins_ndim=None,
                         blocks=None, threads_per_block=None,
                         setup_fn=None,
                         batch_processing=None):
    data = fn.external_source(lambda: get_data(shapes, dtype), batch=True, device="gpu")
    return numba_function(data, run_fn=run_fn, out_types=out_types, in_types=in_types,
                          outs_ndim=outs_ndim, ins_ndim=ins_ndim, blocks=blocks,
                          threads_per_block=threads_per_block, setup_fn=setup_fn,
                          batch_processing=batch_processing, device='gpu')


def _testimpl_numba_func(shapes, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, blocks,
                         threads_per_block, setup_fn, batch_processing, expected_out):
    batch_size = len(shapes)
    pipe = numba_func_cuda_pipe(batch_size=batch_size, num_threads=1, device_id=0, shapes=shapes,
                                dtype=dtype, run_fn=run_fn, setup_fn=setup_fn, out_types=out_types,
                                in_types=in_types, outs_ndim=outs_ndim, ins_ndim=ins_ndim,
                                blocks=blocks, threads_per_block=threads_per_block,
                                batch_processing=batch_processing)
    pipe.build()
    for _ in range(3):
        outs = pipe.run()
        for i in range(batch_size):
            out_arr = np.array(outs[0][i].as_cpu())
            assert np.array_equal(out_arr, expected_out[i])


@with_setup(check_env_compatibility)
def test_numba_func():
    # shape, dtype, run_fn, out_types, in_types,
    # outs_ndim, ins_ndim, blocks, threads_per_block, setup_fn, batch_processing, expected_out
    args = [
        ([(10, 5)], np.uint8, set_all_values_to_255_sample, [dali_types.UINT8], [dali_types.UINT8],
         [2], [2], [1, 1, 1], [10, 5, 1], None, False, [np.full((10, 5), 255, dtype=np.uint8)]),
        ([(10, 5), (10, 5)], np.uint8, set_all_values_to_255_sample, [dali_types.UINT8],
         [dali_types.UINT8], [2], [2], [1, 1, 1], [10, 5, 1], None, False,
         [np.full((10, 5), 255, dtype=np.uint8), np.full((10, 5), 255, dtype=np.uint8)]),
        ([(10, 5)], np.float32, set_output_to_input_plus_5_sample, [dali_types.FLOAT],
         [dali_types.FLOAT], [2], [2], [1, 1, 1], [10, 5, 1], None, False,
         [np.full((10, 5), 6, dtype=np.float32)]),
        ([(20, 5), (20, 5)], np.float32, set_output_to_input_plus_5_sample, [dali_types.FLOAT],
         [dali_types.FLOAT], [2], [2], [1, 1, 1], [20, 5, 1], None, False,
         [np.full((20, 5), 6, dtype=np.float32), np.full((20, 5), 6, dtype=np.float32)]),
        ([(10, 5)], np.float32, set_consecutive_values_sample, [dali_types.FLOAT],
         [dali_types.FLOAT], [2], [2], [1, 1, 1], [10, 5, 1], None, False,
         [np.arange(10 * 5, dtype=np.float32).reshape((10, 5))]),
        ([(20, 10), (20, 10)], np.float32, set_consecutive_values_sample, [dali_types.FLOAT],
         [dali_types.FLOAT], [2], [2], [1, 1, 1], [20, 10, 1], None, False,
         [np.arange(20 * 10, dtype=np.float32).reshape((20, 10)),
          np.arange(20 * 10, dtype=np.float32).reshape((20, 10))]),
    ]

    for shape, dtype, run_fn, out_types, in_types, outs_ndim, ins_ndim, blocks, threads_per_block, \
            setup_fn, batch_processing, expected_out in args:
        yield _testimpl_numba_func, shape, dtype, run_fn, out_types, in_types, outs_ndim, \
              ins_ndim, blocks, threads_per_block, setup_fn, batch_processing, expected_out


@with_setup(check_env_compatibility)
@raises(AssertionError, glob='Currently batch processing for GPU is not supported.')
def test_batch_processing_assertion():
    _testimpl_numba_func([(10, 5)], np.uint8, set_all_values_to_255_sample, [dali_types.UINT8],
                         [dali_types.UINT8], [2], [2], [1, 1, 1], [10, 5, 1], None, True,
                         [np.full((10, 5), 255, dtype=np.uint8)])


@with_setup(check_env_compatibility)
@raises(RuntimeError,
        regex="Shape of input [0-9]+, sample at index [0-9]+ doesn't match the shape of first sample")  # noqa: E501
def test_samples_shapes_assertion():
    _testimpl_numba_func([(10, 20), (10, 10), (10, 20)], np.float32, set_consecutive_values_sample,
                         [dali_types.FLOAT], [dali_types.FLOAT], [2], [2], [1, 1, 1], [20, 10, 1],
                         None, False, [np.arange(20 * 10, dtype=np.float32).reshape((20, 10)),
                                       np.arange(20 * 10, dtype=np.float32).reshape((20, 10))])
