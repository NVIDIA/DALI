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
import sys
import tempfile
import nose.tools

class NumpyReaderPipeline(Pipeline):
    def __init__(self, path, batch_size, file_list=None, path_filter="*.npy",
                 num_threads=1, device_id=0, num_gpus=1,
                 slice_anchor=None, slice_shape=None, io_size=0,
                 use_mmap = True):
        super(NumpyReaderPipeline, self).__init__(batch_size,
                                                  num_threads,
                                                  device_id)

        self.input = None
        if file_list is not None:
            self.input = ops.NumpyReader(file_root = path,
                                         file_list = file_list,
		                                 shard_id = device_id,
                                         num_shards = num_gpus,
                                         dont_use_mmap = not use_mmap)
        else:
            if (slice_anchor is not None) and (slice_shape is not None):
                self.input = ops.NumpyReader(file_root = path,
                                             file_filter = path_filter,
                                             anchor = slice_anchor,
                                             shape = slice_shape,
                                             shard_id = device_id,
                                             num_shards = num_gpus,
                                             target_io_bytes = io_size,
                                             dont_use_mmap = not use_mmap)
            else:
                self.input = ops.NumpyReader(file_root = path,
                                             file_filter = path_filter,
                                             shard_id = device_id,
                                             num_shards = num_gpus,
                                             dont_use_mmap = not use_mmap)

    def define_graph(self):
        inputs = self.input(name="Reader")

        return inputs

    
class NumpyReaderDynamicPipeline(Pipeline):
    def __init__(self, path, batch_size, path_filter="*.npy",
                 num_threads=1, device_id=0, num_gpus=1,
                 use_mmap = True):
        super(NumpyReaderDynamicPipeline, self).__init__(batch_size,
                                                         num_threads,
                                                         device_id)

        class RNGIterator(object):
            def __init__(self):
                self.rng = np.random.RandomState(12345)

            def __iter__(self):
                self.i = 0
                self.n = 10
                return self
                
            def __next__(self):
                anchor = np.expand_dims(self.rng.random_integers(low = 0, high = 5, size=(3)), axis=0).astype(np.int32)
                shape = np.expand_dims(self.rng.random_integers(low = 1, high = 5, size=(3)), axis=0).astype(np.int32)
                return anchor, shape
    
        rngiter = RNGIterator()
        self.source = ops.ExternalSource(source = rngiter, num_outputs = 2)
        self.input = ops.NumpyReader(file_root = path,
                                     file_filter = path_filter,
                                     shard_id = device_id,
                                     num_shards = num_gpus,
                                     dont_use_mmap = not use_mmap)
        
    def define_graph(self):
        anchor, shape = self.source()
        inputs = self.input(name="Reader",
                            anchor = anchor,
                            shape = shape)

        return inputs, anchor, shape
    

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
def test_types_and_shapes():
    with tempfile.TemporaryDirectory() as test_data_root:
        for fortran_order in [False, True]:
            for use_mmap in [True, False]:
                for type in all_numpy_types - unsupported_numpy_types:
                    for shape in test_np_shapes:
                        yield check_array, test_data_root, shape, type, fortran_order, use_mmap

def test_unsupported_types():
    with tempfile.TemporaryDirectory() as test_data_root:
        shape = test_np_shapes[1]
        fortran_order = True
        use_mmap = True
        for type in unsupported_numpy_types:
            nose.tools.assert_raises(RuntimeError, check_array, test_data_root, shape, type,
                                     fortran_order, use_mmap)

# test batch_size > 1
def test_batch():
    with tempfile.TemporaryDirectory() as test_data_root:
        # create files
        num_samples = 20
        batch_size = 4
        files = []
        for index in range(0, num_samples):
            tmpfile = tempfile.NamedTemporaryFile(prefix='batch', suffix='.npy', dir=test_data_root)
            files.append(tmpfile)
            create_numpy_file(tmpfile.name, (5, 2, 8), np.float32, False)

        # sort files according to name
        files.sort(key=lambda x: x.name)

        # create pipe
        for num_threads in [1, 2, 4, 8]:
            pipe = NumpyReaderPipeline(path = test_data_root,
                                       path_filter = "batch*.npy",
                                       batch_size = batch_size,
                                       num_threads = num_threads,
                                       device_id = 0)
            pipe.build()

            for batch in range(0, num_samples, batch_size):
                pipe_out = pipe.run()
                arr_rd = pipe_out[0].as_array()
                arr_np = np.stack([np.load(x.name)
                                   for x in files[batch:batch+batch_size]], axis=0)
                
                # compare
                assert_array_equal(arr_rd, arr_np)

        # clean up
        for f in files:
            f.close()

def test_list_batch():
    with tempfile.TemporaryDirectory() as test_data_root:
        # create temporary file to write list into
        listfile = tempfile.NamedTemporaryFile(suffix='.txt', dir=test_data_root)
        
        # create files
        num_samples = 20
        batch_size = 4
        files = create_numpy_file_list(listfile.name, test_data_root, num_samples, (5, 2, 8), np.float32, False)

        # create pipe
        for num_threads in [1, 2, 4, 8]:
            pipe = NumpyReaderPipeline(path = test_data_root,
                                       file_list = listfile.name,
                                       batch_size = batch_size,
                                       num_threads = num_threads,
                                       device_id = 0)
            pipe.build()

            for batch in range(0, num_samples, batch_size):
                pipe_out = pipe.run()
                arr_rd = pipe_out[0].as_array()
                arr_np = np.stack([np.load(x.name)
                                   for x in files[batch:batch+batch_size]], axis=0)
                
                # compare
                assert_array_equal(arr_rd, arr_np)

        # clean up
        listfile.close()
        for f in files:
            f.close()

test_np_chunk_sizes = [0, 128, 4096]
test_np_slice_shapes = [[12], (10, 10), (5, 20, 10), (4, 8, 3, 6)]
test_np_slice_anchors = [[5], (2, 4), (1, 3, 2), (1, 3, 1, 4)]
test_np_slice_subshapes = [[4], (7, 5), (4, 10, 1), (2, 4, 2, 1)]

# test slice read
def test_static_slice():
    with tempfile.TemporaryDirectory() as test_data_root:
        for fortran_order in [False, True]:
            for io_size in test_np_chunk_sizes:
                for typ in all_numpy_types - unsupported_numpy_types:
                    for idx,shape in enumerate(test_np_slice_shapes):
                        slice_anchor = test_np_slice_anchors[idx]
                        slice_shape = test_np_slice_subshapes[idx]
                        yield check_array_static_slice, test_data_root, shape, slice_anchor, slice_shape, typ, io_size, fortran_order, True

                
def test_dynamic_slice():
    with tempfile.TemporaryDirectory() as test_data_root:
        for fortran_order in [False, True]:
            for use_mmap in [True, False]:
                for typ in all_numpy_types - unsupported_numpy_types:
                    yield check_array_dynamic_slice, test_data_root, (10, 10, 10), typ, fortran_order, use_mmap


# test fused reads
test_np_fused_slice_shapes    = [(10, 10), (5, 8, 10), (5, 8, 10), (4, 8, 3, 6), (4, 8, 3, 6)]
test_np_fused_slice_anchors   = [( 2,  0), (1, 0,  2), (1, 0,  0), (1, 0, 1, 4), (1, 0, 1, 0)]
test_np_fused_slice_subshapes = [( 7, 10), (4, 8,  5), (4, 8, 10), (2, 8, 2, 1), (2, 8, 2, 6)]

def test_static_fused_slice():
    with tempfile.TemporaryDirectory() as test_data_root:
        for fortran_order in [False, True]:
            for use_mmap in [True, False]:
                for io_size in test_np_chunk_sizes:
                    for typ in all_numpy_types - unsupported_numpy_types:
                        for idx,shape in enumerate(test_np_fused_slice_shapes):
                            slice_anchor = test_np_fused_slice_anchors[idx]
                            slice_shape = test_np_fused_slice_subshapes[idx]
                            yield check_array_static_slice, test_data_root, shape, slice_anchor, slice_shape, typ, io_size, fortran_order, use_mmap


# generic helper routines
def create_numpy_file(filename, shape, typ, fortran_order):
    # generate random array
    arr = rng.random_sample(shape) * 10.
    arr = arr.astype(typ)
    if fortran_order:
        arr = np.asfortranarray(arr)
    np.save(filename, arr)

# create a list of numpy files
def create_numpy_file_list(list_name, file_directory, num_samples, shape, typ, fortran_order):
    # generate random array
    filelist = []
    for _ in range(num_samples):
        tmpfile = tempfile.NamedTemporaryFile(suffix='.npy', dir=file_directory)
        create_numpy_file(tmpfile.name, shape, typ, fortran_order)
        filelist.append(tmpfile)
        
    # sort files
    filelist.sort(key=lambda x: x.name)
    
    # write the list into a file
    with open(list_name, 'w') as f:
        f.write("\n".join([os.path.basename(x.name) for x in filelist]))
    
    return filelist

def check_array(directory, shape, typ, fortran_order = False, use_mmap = True):
    # setup file
    tmpfile = tempfile.NamedTemporaryFile(suffix='.npy', dir=directory);
    create_numpy_file(tmpfile.name, shape, typ, fortran_order)
    
    for num_threads in [1, 2, 4, 8]:
        # load with numpy reader
        pipe = NumpyReaderPipeline(path = directory,
                                   path_filter = "*.npy",
                                   batch_size = 1,
                                   num_threads = num_threads,
                                   device_id = 0,
                                   use_mmap = use_mmap)
        pipe.build()
        pipe_out = pipe.run()
        arr_rd = np.squeeze(pipe_out[0].as_array(), axis=0)
        
        # load manually
        arr_np = np.load(tmpfile.name)
        
        # compare
        assert_array_equal(arr_rd, arr_np)

    # delete temp files
    tmpfile.close()


def check_array_static_slice(directory, shape, slice_anchor, slice_shape, typ,
                            io_size=0, fortran_order = False, use_mmap = True):
    # setup file
    tmpfile = tempfile.NamedTemporaryFile(suffix='.npy', dir=directory)
    create_numpy_file(tmpfile.name, shape, typ, fortran_order)
    
    for num_threads in [1, 2, 4, 8]:
	# load with numpy reader
        pipe = NumpyReaderPipeline(path = directory,
                                   path_filter = "*.npy",
                                   batch_size = 1,
                                   num_threads = num_threads,
                                   device_id = 0,
                                   slice_anchor = slice_anchor,
                                   slice_shape = slice_shape,
                                   io_size = io_size,
                                   use_mmap = use_mmap)
        pipe.build()
        pipe_out = pipe.run()
        arr_rd = np.squeeze(pipe_out[0].as_array(), axis=0)

        # load manually
        arr_np = np.load(tmpfile.name)
        slab = [slice(x[0], x[0]+x[1]) for x in zip(slice_anchor, slice_shape)]
        
        # compare
        assert_array_equal(arr_rd, arr_np[slab])

    # delete temp files
    tmpfile.close()


def check_array_dynamic_slice(directory, shape, typ, fortran_order = False, use_mmap = True):
    # setup file
    tmpfile = tempfile.NamedTemporaryFile(suffix='.npy', dir=directory)
    create_numpy_file(tmpfile.name, shape, typ, fortran_order)

    for num_threads in [1, 2, 4, 8]:
        # load with numpy reader
        pipe = NumpyReaderDynamicPipeline(path = directory,
                                          path_filter = "*.npy",
                                          batch_size = 1,
                                          num_threads = num_threads,
                                          device_id = 0,
                                          use_mmap = use_mmap)
        pipe.build()
        pipe_out = pipe.run()
        arr_rd = np.squeeze(pipe_out[0].as_array(), axis=0)

        # load manually
        arr_np = np.load(tmpfile.name)
        slice_anchor = np.squeeze(pipe_out[1].as_array()).tolist()
        slice_shape = np.squeeze(pipe_out[2].as_array()).tolist()
        slab = [slice(x[0], x[0]+x[1]) for x in zip(slice_anchor, slice_shape)]
        
        # compare
        assert_array_equal(arr_rd, arr_np[slab])

    # delete temp files
    tmpfile.close()
