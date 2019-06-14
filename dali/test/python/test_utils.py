# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from nvidia.dali.edge import EdgeReference
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import subprocess

# those functions import modules on demand to no impose additional dependency on numpy or matplot
# to test that are using these utilities
np = None
assert_array_equal = None
assert_allclose = None
def import_numpy():
    global np
    global assert_array_equal
    global assert_allclose
    import numpy as np
    from numpy.testing import assert_array_equal, assert_allclose

Image = None
def import_pil():
    global Image
    from PIL import Image

def save_image(image, file_name):
    import_numpy()
    import_pil()
    if image.dtype == np.float32:
        min = np.min(image)
        max = np.max(image)
        if min >= 0 and max <= 1:
            image = image * 256
        elif min >= -1 and max <= 1:
            image = ((image + 1) * 128)
        elif min >= -128 and max <= 127:
            image = image + 128
    else:
        image = (image - np.iinfo(image.dtype).min) / (np.iinfo(image.dtype).max - np.iinfo(image.dtype).min)
    image = image.astype(np.uint8)
    Image.fromarray(image).save(file_name)

def get_gpu_num():
    sp = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out_str = sp.communicate()
    out_list = out_str[0].split('\n')
    out_list = [elm for elm in out_list if len(elm) > 0]
    return len(out_list)

def check_batch(batch1, batch2, batch_size, eps = 1e-07):
    import_numpy()
    if isinstance(batch1, dali.backend_impl.TensorListGPU):
        batch1 = batch1.as_cpu()
    if isinstance(batch2, dali.backend_impl.TensorListGPU):
        batch2 = batch2.as_cpu()

    for i in range(batch_size):
        is_failed = False
        assert(batch1.at(i).shape == batch2.at(i).shape), \
            "Shape mismatch {} != {}".format(batch1.at(i).shape, batch2.at(i).shape)
        assert(batch1.at(i).size == batch2.at(i).size), \
            "Size mismatch {} != {}".format(batch1.at(i).size, batch2.at(i).size)
        if batch1.at(i).size != 0:
            err = np.mean( np.abs(batch1.at(i) - batch2.at(i)) )
            try:
                err = np.mean( np.abs(batch1.at(i) - batch2.at(i)) )
            except:
                is_failed = True
            if is_failed or err > eps:
                try:
                    print("failed[{}] err[{}]".format(is_failed, err))
                    save_image(batch1.at(i), "err_1.png")
                    save_image(batch2.at(i), "err_2.png")
                except:
                    print("Batch at {} can't be saved as an image".format(i))
                    print(batch1.at(i))
                    print(batch2.at(i))
                assert(False)

def compare_pipelines(pipe1, pipe2, batch_size, N_iterations, eps = 1e-07):
    pipe1.build()
    pipe2.build()
    for k in range(N_iterations):
        out1 = pipe1.run()
        out2 = pipe2.run()
        assert len(out1) == len(out2)
        for i in range(len(out1)):
            out1_data = out1[i].as_cpu() if isinstance(out1[i].at(0), dali.backend_impl.TensorGPU) else out1[i]
            out2_data = out2[i].as_cpu() if isinstance(out2[i].at(0), dali.backend_impl.TensorGPU) else out2[i]
            check_batch(out1_data, out2_data, batch_size, eps)
    print("OK: ({} iterations)".format(N_iterations))

class RandomDataIterator(object):
    import_numpy()
    def __init__(self, batch_size, shape=(10, 600, 800, 3)):
        self.batch_size = batch_size
        self.test_data = []
        for _ in range(self.batch_size):
            np.random.seed(0)
            self.test_data.append(np.array(np.random.rand(*shape) * 255,
                                  dtype = np.uint8 ) )

    def __iter__(self):
        self.i = 0
        self.n = self.batch_size
        return self

    def __next__(self):
        batch = self.test_data
        self.i = (self.i + 1) % self.n
        return (batch)

    next = __next__