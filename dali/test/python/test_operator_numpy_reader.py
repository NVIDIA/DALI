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

class NumpyReaderPipeline(Pipeline):
    def __init__(self, path, batch_size, path_filter="*.npy",
                 num_threads=1, device_id=0, num_gpus=1,
                 slab_anchor=None, slab_shape=None):
        super(NumpyReaderPipeline, self).__init__(batch_size,
                                                  num_threads,
                                                  device_id)

        self.input = None
        if (slab_anchor is not None) and (slab_shape is not None):
            self.input = ops.NumpyReader(file_root = path,
                                         file_filter = path_filter,
                                         anchor = slab_anchor,
                                         shape = slab_shape,
                                         shard_id = device_id,
                                         num_shards = num_gpus)
        else:
            self.input = ops.NumpyReader(file_root = path,
                                         file_filter = path_filter,
		                         shard_id = device_id,
                                         num_shards = num_gpus)

        
    def define_graph(self):
        inputs = self.input(name="Reader")

        return inputs

    
class NumpyReaderDynamicPipeline(Pipeline):
    def __init__(self, path, batch_size, file_shape, path_filter="*.npy",
                 num_threads=1, device_id=0, num_gpus=1):
        super(NumpyReaderPipeline, self).__init__(batch_size,
                                                  num_threads,
                                                  device_id)

        self.rng = np.random(12345)
        self.input = ops.NumpyReader(file_root = path,
                                     file_filter = path_filter,
                                     shard_id = device_id,
                                     num_shards = num_gpus)
        

    def define_graph(self):
	inputs = self.input(name="Reader")

        return inputs
    
    

test_np_types = [np.float32, np.float64, np.int32, np.int64]
test_np_shapes = [(), (11), (4, 7), (6, 2, 5), (1, 2, 7, 4)]
rng = np.random.RandomState(12345)
    
# test: compare reader with numpy

# with different batch_size and num_threads
def test_types_and_shapes():
    with tempfile.TemporaryDirectory() as test_data_root:
        index = 0
        for fortran_order in [False, True]:
            for typ in test_np_types:
                for shape in test_np_shapes:
                    filename = os.path.join(test_data_root, "test_{:02d}.npy".format(index))
                    index += 1
                    yield check_array, filename, shape, typ, fortran_order


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
            

test_np_slab_shapes = [[12], (10, 10), (5, 20, 10), (4, 8, 3, 6)]
test_np_slab_anchors = [[5], (2, 4), (0, 3, 2), (1, 3, 0, 4)]
test_np_slab_subshapes = [[4], (7, 5), (5, 10, 1), (2, 4, 3, 1)]
                
# test slab read
def test_static_slab():

    with tempfile.TemporaryDirectory() as test_data_root:
        index  = 0
        for typ in test_np_types:
            for idx,shape in enumerate(test_np_slab_shapes):
                filename = os.path.join(test_data_root, "test_slab_{:02d}.npy".format(index))
                create_numpy_file(filename, shape, typ, False)
                slab_anchor = test_np_slab_anchors[idx]
                slab_shape = test_np_slab_subshapes[idx]
                index += 1
                yield check_array_static_slab, filename, shape, slab_anchor, slab_shape, typ, False


# generic helper routines                
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


def check_array_static_slab(filename, shape, slab_anchor, slab_shape, typ, fortran_order=False):
    # setup file
    create_numpy_file(filename, shape, typ, fortran_order)
    
    for num_threads in [1, 2, 4, 8]:
	# load with numpy reader
        pipe = NumpyReaderPipeline(path = os.path.dirname(filename),
                                   path_filter = os.path.basename(filename),
                                   batch_size = 1,
                                   num_threads = num_threads,
                                   device_id = 0,
                                   slab_anchor = slab_anchor,
                                   slab_shape = slab_shape)
        pipe.build()
        pipe_out = pipe.run()
        arr_rd = np.squeeze(pipe_out[0].as_array(), axis=0)

        # load manually
        arr_np = np.load(filename)
        slab = [slice(x[0], x[0]+x[1]) for x in zip(slab_anchor, slab_shape)]

        # compare
        assert_array_equal(arr_rd, arr_np[slab])

    # delete temp files
    delete_numpy_file(filename)
