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
    def __init__(self, path, batch_size, path_filter="*.npy", num_threads=1, device_id=0, num_gpus=1):
        super(NumpyReaderPipeline, self).__init__(batch_size,
                                                  num_threads,
                                                  device_id)
        
        self.input = ops.NumpyReader(file_root = path,
                                     file_filter = path_filter,
                                     shard_id = device_id,
                                     num_shards = num_gpus)

        
    def define_graph(self):
        inputs = self.input(name="Reader")

        return inputs

# test: compare reader with numpy
# with different batch_size and num_threads

def test_reader_vs_numpy():

    # create temporary directory
    with tempfile.TemporaryDirectory() as test_data_root:
    
        # we need that
        rng = np.random.RandomState(12345)
    
        # 1D array integer
        arr = rng.randint(0,10, (15))
        np.save(os.path.join(test_data_root, "test_01.npy"), arr)

        # 1D array float
        arr = rng.random_sample((25)).astype(np.float32)
        np.save(os.path.join(test_data_root, "test_02.npy"), arr)

        # 1D array double
        arr = rng.random_sample((8)).astype(np.float64)
        np.save(os.path.join(test_data_root, "test_03.npy"), arr)

        # 2D array float
        arr = rng.random_sample((25,10)).astype(np.float32)
        np.save(os.path.join(test_data_root, "test_04.npy"), arr)

        # 3D array double
        arr = rng.random_sample((1,20,10)).astype(np.float64)
        np.save(os.path.join(test_data_root, "test_05.npy"), arr)

        # 3D array float fortran order
        arr = rng.random_sample((3,4,2)).astype(np.float32)
        arr = np.asfortranarray(arr)
        np.save(os.path.join(test_data_root, "test_06.npy"), arr)

        # 3D Fortran with singleton dim
        arr = rng.randint(0, 10, (3,1,2))
        arr = np.asfortranarray(arr)
        np.save(os.path.join(test_data_root, "test_07.npy"), arr)

        # singleton file
        arr = 3
        np.save(os.path.join(test_data_root, "test_08.npy"), arr)
    
        # get files:
        filelist = sorted([os.path.join(test_data_root, x)
                           for x in os.listdir(test_data_root) if x.endswith('.npy')])

        # batch size 1:
        for num_threads in [1, 2, 4, 8]:
            # create pipe and build
            pipe = NumpyReaderPipeline(path = test_data_root,
                                       batch_size = 1,
                                       num_threads = num_threads,
                                       device_id = 0)
            pipe.build()

            for fname in filelist:
                pipe_out = pipe.run()
                arr_rd = np.squeeze(pipe_out[0].as_array(), axis=0)
                arr_np = np.load(fname)
            
                assert_array_equal(arr_rd, arr_np)

        # batch size 4, 20 samples
        num_samples = 20
        batch_size = 4
        for sample in range(num_samples):
            arr = rng.random_sample((10,15,3)).astype(np.float32)
            np.save(os.path.join(test_data_root, "sample_{:02d}.npy".format(sample)), arr)
            
        filelist = sorted([os.path.join(test_data_root, x)
                           for x in os.listdir(test_data_root) if x.startswith("sample_") and x.endswith('.npy')])

        for num_threads in [1, 2, 4, 8]:
            # create pipe and build
            pipe = NumpyReaderPipeline(path = test_data_root,
                                       batch_size = batch_size,
                                       path_filter = "sample_*.npy",
                                       num_threads = num_threads,
                                       device_id = 0)
            pipe.build()

            for idx in range(0, num_samples, batch_size):
                pipe_out = pipe.run()
                arr_rd = pipe_out[0].as_array()
                arr_np = np.stack([ np.load(filelist[x]) for x in range(idx, idx+batch_size) ], axis = 0)

                assert_array_equal(arr_rd, arr_np)
                
            
if __name__ == "__main__":
    test_reader_vs_numpy()
