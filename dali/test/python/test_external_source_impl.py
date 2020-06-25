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
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.backend as backend
import numpy as np
from nose.tools import assert_raises
from test_utils import check_output
import random
from collections import Iterable
datapy = np

make_array = np.array

# to use this it is enough to just import all functions from it by `from test_internals_operator_external_source import *`
# nose will query for the methods available and will run them
# the code for CPU and GPU input is 99% the same and the biggest difference is between importing numpy or cupy
# so it is better to store everything in one file and just call `use_cupy` to switch between the default numpy and cupy

def _to_numpy(x):
    assert(False)

def cast_to(x, dtype):
    return x.astype(dtype)

def asnumpy(x):
    if x is None:
        return None
    if isinstance(x, list):
        return [asnumpy(y) for y in x]
    if isinstance(x, np.ndarray):
        return x
    return _to_numpy(x)

def use_cupy():
    global cp
    global datapy
    global make_array
    global _to_numpy
    import cupy as cp
    datapy = cp
    make_array = cp.array
    _to_numpy = cp.asnumpy

random_seed = datapy.random.seed
random_array = datapy.random.ranf
random_int = datapy.random.randint

def use_torch(gpu):
    global torch
    global datapy
    global _to_numpy
    global cast_to
    import torch
    datapy = torch
    def torch2numpy(tensor):
        return np.array(tensor.cpu())
    _to_numpy = torch2numpy
    global random_array
    def make_torch_tensor(*args, **kwargs):
        t = torch.tensor(*args, **kwargs)
        return t.cuda() if gpu else t
    def torch_cast(x, dtype):
        return x.type(dtype)
    cast_to = torch_cast
    random_array = lambda shape: make_torch_tensor(np.random.ranf(shape))
    global make_array
    make_array = make_torch_tensor

class TestIterator():
    def __init__(self, n, batch_size, dims = [2], as_tensor = False):
        self.batch_size = batch_size
        self.dims = dims
        self.n = n
        self.as_tensor = as_tensor
        self.i = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        # return a copy, so that the iteration number doesn't collide
        return TestIterator(self.n, self.batch_size, self.dims, self.as_tensor)

    def __next__(self):
        random_seed(12345 * self.i + 4321)
        def generate(dim):
            shape = random_int(1, 10, [dim]).tolist()
            if self.as_tensor:
                return random_array([self.batch_size] + shape)
            else:
                return [random_array(shape) for _ in range(self.batch_size)]
        if self.i < self.n:
            self.i += 1
            if isinstance(self.dims, (list, tuple)):
                return [generate(d) for d in self.dims]
            else:
                return generate(self.dims)
        else:
            self.i = 0
            raise StopIteration
    next = __next__

def run_and_check(pipe, ref_iterable):
    iter_ref = iter(ref_iterable)
    i = 0
    while True:
        try:
            pipe_out = pipe.run()
            data = next(iter_ref)
            data = asnumpy(data)
            check_output(pipe_out, data)
            i += 1
        except StopIteration:
            break
    assert(i == len(ref_iterable))

def _test_iter_setup(use_fn_api, by_name, device):
    batch_size = 7
    class IterSetupPipeline(Pipeline):
        def __init__(self, iterator, num_threads, device_id, device):
            super(IterSetupPipeline, self).__init__(
                batch_size = iterator.batch_size,
                num_threads = num_threads,
                device_id = device_id)

            self.iterator = iterator
            self._device = device

        def define_graph(self):
            if use_fn_api:
                self.batch_1 = fn.external_source(device = self._device, name = "src1")
                self.batch_2 = fn.external_source(device = self._device, name = "src2")
            else:
                input_1 = ops.ExternalSource(device = self._device)
                input_2 = ops.ExternalSource(device = self._device)
                self.batch_1 = input_1(name = "src1")
                self.batch_2 = input_2(name = "src2")
            return [self.batch_1, self.batch_2]

        def iter_setup(self):
            batch_1, batch_2 = next(self.iterator)
            if by_name:
                self.feed_input("src1", batch_1)
                self.feed_input("src2", batch_2)
            else:
                self.feed_input(self.batch_1, batch_1)
                self.feed_input(self.batch_2, batch_2)

    iter_num = 5
    source = TestIterator(iter_num, batch_size, [2, 3])
    pipe = IterSetupPipeline(iter(source), 3, 0, device)
    pipe.build()

    run_and_check(pipe, source)

def test_iter_setup():
    for use_fn_api in [False, True]:
        for by_name in [False, True]:
            for device in ["cpu", "gpu"]:
                yield _test_iter_setup, use_fn_api, by_name, device

def _test_external_source_callback(use_fn_api, device):
    iter_num = 5
    batch_size = 9
    pipe = Pipeline(batch_size, 3, 0)

    # this should produce a single Tensor / TensorList per batch,
    # not wrapped in additional list
    source = TestIterator(iter_num, batch_size, 4, device == "gpu")
    iter_in = iter(source)

    if use_fn_api:
        input = fn.external_source(lambda: next(iter_in), device = device)
    else:
        ext_source = ops.ExternalSource(lambda: next(iter_in), device = device)
        input = ext_source()
    pipe.set_outputs(input)
    pipe.build()

    run_and_check(pipe, source)

def test_external_source_callback():
    for use_fn_api in [False, True]:
        for device in ["cpu", "gpu"]:
            yield _test_external_source_callback, use_fn_api, device

def _test_external_source_callback_split(use_fn_api, device):
    iter_num = 5
    batch_size = 9
    pipe = Pipeline(batch_size, 3, 0)

    # this should produce a two-element list of Tensor(Lists), the first
    # being 2D, the second being 3D (+ batch dimension)
    source = TestIterator(iter_num, batch_size, [2, 3], device == "gpu")
    iter_in = iter(source)

    if use_fn_api:
        inputs = fn.external_source(lambda: next(iter_in), 2, device = device)
    else:
        ext_source = ops.ExternalSource(lambda: next(iter_in), num_outputs = 2, device = device)
        inputs = ext_source()
    pipe.set_outputs(*inputs)
    pipe.build()

    run_and_check(pipe, source)

def test_external_source_callback_split():
    for use_fn_api in [False, True]:
        for device in ["cpu", "gpu"]:
            yield _test_external_source_callback_split, use_fn_api, device


def _test_external_source_iter(use_fn_api, device):
    iter_num = 5
    batch_size = 9
    pipe = Pipeline(batch_size, 3, 0)

    # this should produce a single Tensor / TensorList per batch,
    # not wrapped in additional list
    source = TestIterator(iter_num, batch_size, 4, device == "gpu")

    if use_fn_api:
        input = fn.external_source(source, device = device)
    else:
        ext_source = ops.ExternalSource(source, device = device)
        input = ext_source()
    pipe.set_outputs(input)
    pipe.build()

    run_and_check(pipe, source)

def test_external_source_iter():
    for use_fn_api in [False, True]:
        for device in ["cpu", "gpu"]:
            yield _test_external_source_callback, use_fn_api, device

def _test_external_source_iter_split(use_fn_api, device):
    iter_num = 5
    batch_size = 9
    pipe = Pipeline(batch_size, 3, 0)

    # this should produce a three-element list of Tensor(Lists), the first
    # being 4D, the second being 2D and the third 3D (+ batch dimension)
    source = TestIterator(iter_num, batch_size, [4, 2, 3], device == "gpu")

    if use_fn_api:
        inputs = fn.external_source(source, 3, device = device)
    else:
        ext_source = ops.ExternalSource(source, num_outputs = 3, device = device)
        inputs = ext_source()
    pipe.set_outputs(*inputs)
    pipe.build()

    run_and_check(pipe, source)

def test_external_source_iter_split():
    for use_fn_api in [False, True]:
        for device in ["cpu", "gpu"]:
            yield _test_external_source_callback_split, use_fn_api, device

def test_external_source_collection():
    pipe = Pipeline(1, 3, 0)

    batches = [
        [make_array([1.5,2.5], dtype=datapy.float32)],
        [make_array([-1, 3.5,4.5], dtype=datapy.float32)]
    ]

    pipe.set_outputs(fn.external_source(batches))
    pipe.build()
    run_and_check(pipe, batches)


def test_external_source_collection_cycling():
    pipe = Pipeline(1, 3, 0)

    batches = [
        [make_array([1.5,2.5], dtype=datapy.float32)],
        [make_array([-1, 3.5,4.5], dtype=datapy.float32)]
    ]

    pipe.set_outputs(fn.external_source(batches, cycle = True))
    pipe.build()

    # epochs are cycles over the source iterable
    for _ in range(3):
        for batch in batches:
            batch = asnumpy(batch)
            check_output(pipe.run(), batch)

def test_external_source_with_iter():
    for attempt in range(10):
        pipe = Pipeline(1, 3, 0)

        pipe.set_outputs(fn.external_source(lambda i: [make_array([attempt * 100 + i * 10 + 1.5], dtype=datapy.float32)]))
        pipe.build()

        for i in range(10):
            check_output(pipe.run(), [np.array([attempt * 100 + i * 10 + 1.5], dtype=np.float32)])

def test_external_source_generator():
    pipe = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [make_array([i + 1.5], dtype=datapy.float32)]

    pipe.set_outputs(fn.external_source(gen()))
    pipe.build()

    for i in range(5):
        check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])

def test_external_source_gen_function_cycle():
    pipe = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [make_array([i + 1.5], dtype=datapy.float32)]

    pipe.set_outputs(fn.external_source(gen, cycle = True))
    pipe.build()

    for _ in range(3):
        for i in range(5):
            check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])

def test_external_source_generator_cycle_error():
    _ = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [make_array([i + 1.5], dtype=datapy.float32)]

    fn.external_source(gen(), cycle = False)     # no cycle - OK
    with assert_raises(TypeError):
        fn.external_source(gen(), cycle = True)  # cycle over generator - error expected

def test_external_source():
    class TestIterator():
        def __init__(self, n):
            self.n = n

        def __iter__(self):
            self.i = 0
            return self

        def __next__(self):
            batch_1 = []
            batch_2 = []
            if self.i < self.n:
                batch_1.append(datapy.arange(0, 1, dtype=datapy.float32))
                batch_2.append(datapy.arange(0, 1, dtype=datapy.float32))
                self.i += 1
                return batch_1, batch_2
            else:
                self.i = 0
                raise StopIteration
        next = __next__

    class IterSetupPipeline(Pipeline):
        def __init__(self, iterator, num_threads, device_id):
            super(IterSetupPipeline, self).__init__(1, num_threads, device_id)
            self.input_1 = ops.ExternalSource()
            self.input_2 = ops.ExternalSource()
            self.iterator = iterator

        def define_graph(self):
            self.batch_1 = self.input_1()
            self.batch_2 = self.input_2()
            return [self.batch_1, self.batch_2]

        def iter_setup(self):
            batch_1, batch_2 = next(self.iterator)
            self.feed_input(self.batch_1, batch_1)
            self.feed_input(self.batch_2, batch_2)

    iter_num = 5
    iterator = iter(TestIterator(iter_num))
    pipe = IterSetupPipeline(iterator, 3, 0)
    pipe.build()

    i = 0
    while True:
        try:
            pipe_out = pipe.run()
            i += 1
        except StopIteration:
            break
    assert(iter_num == i)

def test_external_source_fail():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_s_size, num_threads, device_id):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_s_size_ = external_s_size

        def define_graph(self):
            self.batch = self.input()
            return [self.batch]

        def iter_setup(self):
            batch = datapy.zeros([self.external_s_size_,4,5])
            self.feed_input(self.batch, batch)

    batch_size = 3
    pipe = ExternalSourcePipeline(batch_size, batch_size - 1, 3, 0)
    pipe.build()
    assert_raises(RuntimeError, pipe.run)

def test_external_source_fail_missing_output():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_s_size, num_threads, device_id):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.input_2 = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_s_size_ = external_s_size

        def define_graph(self):
            self.batch = self.input()
            self.batch_2 = self.input_2()
            return [self.batch]

        def iter_setup(self):
            batch = datapy.zeros([self.external_s_size_,4,5])
            self.feed_input(self.batch, batch)
            self.feed_input(self.batch_2, batch)

    batch_size = 3
    pipe = ExternalSourcePipeline(batch_size, batch_size, 3, 0)
    pipe.build()
    assert_raises(RuntimeError, pipe.run)

def test_external_source_fail_list():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_s_size, num_threads, device_id):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_s_size_ = external_s_size

        def define_graph(self):
            self.batch = self.input()
            return [self.batch]

        def iter_setup(self):
            batch = []
            for _ in range(self.external_s_size_):
                batch.append(datapy.zeros([3,4,5]))
            self.feed_input(self.batch, batch)

    batch_size = 3
    pipe = ExternalSourcePipeline(batch_size, batch_size - 1, 3, 0)
    pipe.build()
    assert_raises(RuntimeError, pipe.run)

def external_data_veri(external_data, batch_size):
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_data, num_threads, device_id):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_data = external_data

        def define_graph(self):
            self.batch = self.input()
            return [self.batch]

        def iter_setup(self):
            batch = []
            for elm in self.external_data:
                batch.append(make_array(elm, dtype=datapy.uint8))
            self.feed_input(self.batch, batch)

    pipe = ExternalSourcePipeline(batch_size, external_data, 3, 0)
    pipe.build()
    for _ in range(10):
        out = pipe.run()
        for i in range(batch_size):
            assert out[0].as_array()[i] == external_data[i]

def test_external_source_scalar_list():
    batch_size = 3
    label_data = 10
    lists = []
    scalars = []
    for i in range(batch_size):
        lists.append([label_data + i])
        scalars.append(label_data + i * 10)
    for external_data in [lists, scalars]:
        yield external_data_veri, external_data, batch_size

def test_external_source_gpu():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, use_list):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource(device="gpu")
            self.crop = ops.Crop(device="gpu", crop_h=32, crop_w=32, crop_pos_x=0.2, crop_pos_y=0.2)
            self.use_list = use_list

        def define_graph(self):
            self.batch = self.input()
            output = self.crop(self.batch)
            return output

        def iter_setup(self):
            if use_list:
                batch_data = [cast_to(random_array([100, 100, 3]) * 256, datapy.uint8) for _ in range(self.batch_size)]
            else:
                batch_data = cast_to(random_array([self.batch_size, 100, 100, 3]) * 256, datapy.uint8)
            self.feed_input(self.batch, batch_data, layout="HWC")

    for batch_size in [1, 10]:
        for use_list in (True, False):
            pipe = ExternalSourcePipeline(batch_size, 3, 0, use_list)
            pipe.build()
            pipe.run()
