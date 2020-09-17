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
import nvidia.dali as dali
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os
import tempfile
import nose.tools

class NumpyReaderPipeline(Pipeline):
    def __init__(self, path, batch_size, file_list="",
                 path_filter="*.npy", num_threads=1, device_id=0, num_gpus=1):
        super(NumpyReaderPipeline, self).__init__(batch_size,
                                                  num_threads,
                                                  device_id)
        
        self.input = ops.NumpyReader(file_root = path,
                                     file_list = file_list,
                                     shard_id = device_id,
                                     num_shards = num_gpus)

    def define_graph(self):
        inputs = self.input(name="Reader")

        return inputs


all_numpy_types = set(
    [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc, np.int_, np.uint,
     np.longlong, np.ulonglong, np.half, np.float16, np.single, np.double, np.longdouble,
     np.csingle, np.cdouble, np.clongdouble, np.int8, np.int16, np.int32, np.int64, np.uint8,
     np.uint16, np.uint32, np.uint64, np.intp, np.uintp, np.float32, np.float64, np.float_,
     np.complex64, np.complex128, np.complex_])
unsupported_numpy_types = set(
    [np.bool_, np.csingle, np.cdouble, np.clongdouble, np.complex64, np.complex128, np.longdouble,
     np.complex_])
test_np_shapes = [(), (11), (4, 7), (6, 2, 5), (1, 2, 7, 4)]
rng = np.random.RandomState(12345)
    
# test: compare reader with numpy
# with different batch_size and num_threads
def test_types_and_shapes():
    with tempfile.TemporaryDirectory() as test_data_root:
        index = 0
        for fortran_order in [False, True]:
            for type in all_numpy_types - unsupported_numpy_types:
                for shape in test_np_shapes:
                    filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
                    index += 1
                    yield check_array, filename, shape, type, fortran_order


def test_unsupported_types():
    with tempfile.TemporaryDirectory() as test_data_root:
        index = 0
        filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
        shape = test_np_shapes[1]
        fortran_order = True
        for type in unsupported_numpy_types:
            nose.tools.assert_raises(RuntimeError, check_array, filename, shape, type,
                                     fortran_order)


# test batch_size > 1
def test_batch():
    with tempfile.TemporaryDirectory() as test_data_root:
        # create files
        num_samples = 20
        batch_size = 4
        filenames = []
        for index in range(0,num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)

        # create pipe
        for num_threads in [1, 2, 4, 8]:
            pipe = NumpyReaderPipeline(path = test_data_root,
                                       path_filter = "test_*.npy",
                                       batch_size = batch_size,
                                       num_threads = num_threads,
                                       device_id = 0)
            pipe.build()

            for batch in range(0, num_samples, batch_size):
                pipe_out = pipe.run()
                arr_rd = pipe_out[0].as_array()
                arr_np = np.stack([np.load(x)
                                   for x in filenames[batch:batch+batch_size]], axis=0)
                
                # compare
                assert_array_equal(arr_rd, arr_np)


def test_batch_file_list():
    with tempfile.TemporaryDirectory() as test_data_root:
        # create files
        num_samples = 20
        batch_size = 4
        filenames = []
        arr_np_list = []
        for index in range(0,num_samples):
            filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
            filenames.append(filename)
            create_numpy_file(filename, (5, 2, 8), np.float32, False)
            arr_np_list.append(np.load(filename))
        arr_np_all = np.stack(arr_np_list, axis=0)
        
        # save filenames
        with open(os.path.join(test_data_root, "input.lst"), "w") as f:
            f.writelines("\n".join([os.path.basename(x) for x in filenames]))

        # create pipe
        for num_threads in [1, 2, 4, 8]:
            pipe = NumpyReaderPipeline(path = test_data_root,
                                       file_list = os.path.join(test_data_root, "input.lst"),
                                       batch_size = batch_size,
                                       num_threads = num_threads,
                                       device_id = 0)
            pipe.build()

            for batch in range(0, num_samples, batch_size):
                pipe_out = pipe.run()
                arr_rd = pipe_out[0].as_array()
                arr_np = arr_np_all[batch:batch+batch_size, ...]
                
                # compare
                assert_array_equal(arr_rd, arr_np)


def create_numpy_file(filename, shape, typ, fortran_order):
    # generate random array
    arr = rng.random_sample(shape) * 10.
    arr = arr.astype(typ)
    if fortran_order:
        arr = np.asfortranarray(arr)
    np.save(filename, arr)

def delete_numpy_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)

def check_array(filename, shape, typ, fortran_order=False):
    # setup file
    create_numpy_file(filename, shape, typ, fortran_order)
    
    for num_threads in [1, 2, 4, 8]:
        # load with numpy reader
        pipe = NumpyReaderPipeline(path = os.path.dirname(filename),
                                   path_filter = os.path.basename(filename),
                                   batch_size = 1,
                                   num_threads = num_threads,
                                   device_id = 0)
        pipe.build()
        pipe_out = pipe.run()
        arr_rd = np.squeeze(pipe_out[0].as_array(), axis=0)
        
        # load manually
        arr_np = np.load(filename)
        
        # compare
        assert_array_equal(arr_rd, arr_np)

    # delete temp files
    delete_numpy_file(filename)
