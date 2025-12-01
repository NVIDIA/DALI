# Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
from nvidia.dali import Pipeline

from nose_utils import assert_raises
from test_utils import check_output

datapy = np

make_array = np.array
random_seed = np.random.seed
random_array = np.random.ranf
random_int = np.random.randint

# to use this it is enough to just import all functions from it by
# `from test_internals_operator_external_source import *`
# nose will query for the methods available and will run them
# the code for CPU and GPU input is 99% the same and the biggest
# difference is between importing numpy or cupy so it is better to store everything in one file
# and just call `use_cupy` to switch between the default numpy and cupy

cpu_input = True


def _to_numpy(x):
    assert False


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
    global random_seed
    global random_array
    global random_int
    import cupy as cp

    datapy = cp
    make_array = cp.array
    _to_numpy = cp.asnumpy
    random_seed = datapy.random.seed
    random_array = datapy.random.ranf
    random_int = datapy.random.randint
    global cpu_input
    cpu_input = False


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

    def random_array(shape):
        return make_torch_tensor(np.random.ranf(shape))

    global make_array

    make_array = make_torch_tensor
    global cpu_input
    cpu_input = not gpu


class TestIterator:
    def __init__(self, n, batch_size, dims=[2], as_tensor=False):
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
            if self.as_tensor:
                shape = random_int(1, 10, [dim]).tolist()
                return random_array([self.batch_size] + shape)
            else:
                return [
                    random_array(random_int(1, 10, [dim]).tolist()) for _ in range(self.batch_size)
                ]

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


class SampleIterator:
    def __init__(self, batch_iterator, is_multioutput=False):
        self.src = batch_iterator
        self.is_multioutput = is_multioutput
        self.batch = ([],) if is_multioutput else []
        self.idx = 0

    def __iter__(self):
        return SampleIterator(iter(self.src), self.is_multioutput)

    def __next__(self):
        batch_size = len(self.batch[0]) if self.is_multioutput else len(self.batch)
        if self.idx >= batch_size:
            self.idx = 0
            self.batch = next(self.src)
        if self.is_multioutput:
            ret = tuple(b[self.idx] for b in self.batch)
        else:
            ret = self.batch[self.idx]
        self.idx += 1
        return ret

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
    assert i == len(ref_iterable)


def _test_iter_setup(use_fn_api, by_name, device):
    batch_size = 7

    class IterSetupPipeline(Pipeline):
        def __init__(self, iterator, num_threads, device_id, device):
            super(IterSetupPipeline, self).__init__(
                batch_size=iterator.batch_size, num_threads=num_threads, device_id=device_id
            )

            self.iterator = iterator
            self._device = device

        def define_graph(self):
            if use_fn_api:
                self.batch_1 = fn.external_source(device=self._device, name="src1")
                self.batch_2 = fn.external_source(device=self._device, name="src2")
            else:
                input_1 = ops.ExternalSource(device=self._device)
                input_2 = ops.ExternalSource(device=self._device)
                self.batch_1 = input_1(name="src1")
                self.batch_2 = input_2(name="src2")
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

    run_and_check(pipe, source)


def test_iter_setup():
    for use_fn_api in [False, True]:
        for by_name in [False, True]:
            for device in ["cpu", "gpu"]:
                yield _test_iter_setup, use_fn_api, by_name, device


def _test_external_source_callback(use_fn_api, batch, as_tensor, device):
    iter_num = 5
    batch_size = 9
    pipe = Pipeline(batch_size, 3, 0)

    # this should produce a single Tensor / TensorList per batch,
    # not wrapped in additional list
    source = TestIterator(iter_num, batch_size, 4, device == "gpu")

    iter_in = iter(source) if batch else iter(SampleIterator(iter(source)))

    if use_fn_api:
        input = fn.external_source(lambda: next(iter_in), device=device, batch=batch)
    else:
        ext_source = ops.ExternalSource(lambda: next(iter_in), device=device, batch=batch)
        input = ext_source()
    pipe.set_outputs(input)

    run_and_check(pipe, source)


def test_external_source_callback():
    for use_fn_api in [False, True]:
        for device in ["cpu", "gpu"]:
            for batch in [True, False]:
                for as_tensor in [False, True]:
                    yield _test_external_source_callback, use_fn_api, batch, as_tensor, device


def _test_external_source_callback_split(use_fn_api, batch, as_tensor, device):
    iter_num = 5
    batch_size = 9
    pipe = Pipeline(batch_size, 3, 0)

    # this should produce a two-element list of Tensor(Lists), the first
    # being 2D, the second being 3D (+ batch dimension)
    source = TestIterator(iter_num, batch_size, [2, 3], as_tensor)
    iter_in = iter(source) if batch else iter(SampleIterator(iter(source), True))

    if use_fn_api:
        inputs = fn.external_source(lambda: next(iter_in), 2, device=device, batch=batch)
    else:
        ext_source = ops.ExternalSource(
            lambda: next(iter_in), num_outputs=2, device=device, batch=batch
        )
        inputs = ext_source()
    pipe.set_outputs(*inputs)

    run_and_check(pipe, source)


def test_external_source_callback_split():
    for use_fn_api in [False, True]:
        for device in ["cpu", "gpu"]:
            for batch in [True, False]:
                for as_tensor in [False, True]:
                    yield _test_external_source_callback_split, use_fn_api, batch, as_tensor, device


def _test_external_source_iter(use_fn_api, device):
    iter_num = 5
    batch_size = 9
    pipe = Pipeline(batch_size, 3, 0)

    # this should produce a single Tensor / TensorList per batch,
    # not wrapped in additional list
    source = TestIterator(iter_num, batch_size, 4, device == "gpu")

    if use_fn_api:
        input = fn.external_source(source, device=device)
    else:
        ext_source = ops.ExternalSource(source, device=device)
        input = ext_source()
    pipe.set_outputs(input)

    run_and_check(pipe, source)


def test_external_source_iter():
    for use_fn_api in [False, True]:
        for device in ["cpu", "gpu"]:
            yield _test_external_source_iter, use_fn_api, device


def _test_external_source_iter_split(use_fn_api, device):
    iter_num = 5
    batch_size = 9
    pipe = Pipeline(batch_size, 3, 0)

    # this should produce a three-element list of Tensor(Lists), the first
    # being 4D, the second being 2D and the third 3D (+ batch dimension)
    source = TestIterator(iter_num, batch_size, [4, 2, 3], device == "gpu")

    if use_fn_api:
        inputs = fn.external_source(source, 3, device=device)
    else:
        ext_source = ops.ExternalSource(source, num_outputs=3, device=device)
        inputs = ext_source()
    pipe.set_outputs(*inputs)

    run_and_check(pipe, source)


def test_external_source_iter_split():
    for use_fn_api in [False, True]:
        for device in ["cpu", "gpu"]:
            yield _test_external_source_iter_split, use_fn_api, device


def test_external_source_collection():
    pipe = Pipeline(1, 3, 0)

    batches = [
        [make_array([1.5, 2.5], dtype=datapy.float32)],
        [make_array([-1, 3.5, 4.5], dtype=datapy.float32)],
    ]

    pipe.set_outputs(fn.external_source(batches))
    run_and_check(pipe, batches)


def test_external_source_iterate_ndarray():
    pipe = Pipeline(4, 3, 0)

    batch = make_array([1.5, 2.5, 2, 3], dtype=datapy.float32)

    pipe.set_outputs(fn.external_source(batch, batch=False))
    run_and_check(pipe, [batch])


def test_external_source_collection_cycling():
    pipe = Pipeline(1, 3, 0)

    batches = [
        [make_array([1.5, 2.5], dtype=datapy.float32)],
        [make_array([-1, 3.5, 4.5], dtype=datapy.float32)],
    ]

    pipe.set_outputs(fn.external_source(batches, cycle=True))

    # epochs are cycles over the source iterable
    for _ in range(3):
        for batch in batches:
            batch = asnumpy(batch)
            check_output(pipe.run(), batch)


def test_external_source_collection_cycling_raise():
    pipe = Pipeline(1, 3, 0, prefetch_queue_depth=1)

    batches = [
        [make_array([1.5, 2.5], dtype=datapy.float32)],
        [make_array([-1, 3.5, 4.5], dtype=datapy.float32)],
    ]

    def batch_gen():
        for b in batches:
            yield b

    pipe.set_outputs(
        fn.external_source(batches, cycle="raise"), fn.external_source(batch_gen, cycle="raise")
    )

    # epochs are cycles over the source iterable
    for _ in range(3):
        for batch in batches:
            pipe_out = pipe.run()
            batch = asnumpy(batch)
            batch = batch, batch
            check_output(pipe_out, batch)

        with assert_raises(StopIteration):
            pipe.run()
        pipe.reset()


def test_external_source_with_iter():
    for attempt in range(10):
        pipe = Pipeline(1, 3, 0)

        pipe.set_outputs(
            fn.external_source(
                lambda i: [make_array([attempt * 100 + i * 10 + 1.5], dtype=datapy.float32)]
            )
        )

        for i in range(10):
            check_output(pipe.run(), [np.array([attempt * 100 + i * 10 + 1.5], dtype=np.float32)])


def test_external_source_with_sample_info():
    batch_size = 7
    for attempt in range(10):
        pipe = Pipeline(batch_size, 3, 0)

        def src(si):
            assert si.idx_in_epoch == batch_size * si.iteration + si.idx_in_batch
            return make_array(
                [attempt * 100 + si.iteration * 10 + si.idx_in_batch + 1.5], dtype=datapy.float32
            )

        pipe.set_outputs(fn.external_source(src, batch=False))

        for i in range(10):
            batch = [
                np.array([attempt * 100 + i * 10 + s + 1.5], dtype=np.float32)
                for s in range(batch_size)
            ]
            check_output(pipe.run(), batch)


def test_external_source_generator():
    pipe = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [make_array([i + 1.5], dtype=datapy.float32)]

    pipe.set_outputs(fn.external_source(gen()))

    for i in range(5):
        check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])


def test_external_source_gen_function_cycle():
    pipe = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [make_array([i + 1.5], dtype=datapy.float32)]

    pipe.set_outputs(fn.external_source(gen, cycle=True))

    for _ in range(3):
        for i in range(5):
            check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])


def test_external_source_gen_function_partial():
    pipe = Pipeline(1, 3, 0)

    def gen(base):
        for i in range(5):
            yield [make_array([i + base], dtype=datapy.float32)]

    pipe.set_outputs(fn.external_source(functools.partial(gen, 1.5), cycle=True))

    for _ in range(3):
        for i in range(5):
            check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])


def test_external_source_generator_cycle_error():
    _ = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [make_array([i + 1.5], dtype=datapy.float32)]

    fn.external_source(gen(), cycle=False)  # no cycle - OK
    with assert_raises(
        TypeError, glob="Cannot cycle through a generator * pass that function instead as `source`."
    ):
        fn.external_source(gen(), cycle=True)  # cycle over generator - error expected


def test_external_source():
    class TestIterator:
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
            super().__init__(1, num_threads, device_id)
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

    i = 0
    while True:
        try:
            pipe.run()
            i += 1
        except StopIteration:
            break
    assert iter_num == i


def test_external_source_fail_missing_output():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_s_size, num_threads, device_id):
            super().__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.input_2 = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_s_size_ = external_s_size

        def define_graph(self):
            self.batch = self.input()
            self.batch_2 = self.input_2()
            return [self.batch]

        def iter_setup(self):
            batch = datapy.zeros([self.external_s_size_, 4, 5])
            self.feed_input(self.batch, batch)
            self.feed_input(self.batch_2, batch)

    batch_size = 3
    pipe = ExternalSourcePipeline(batch_size, batch_size, 3, 0)
    assert_raises(KeyError, pipe.run, regex=r"Could not find an input operator with name .*")


def external_data_veri(external_data, batch_size):
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_data, num_threads, device_id):
            super().__init__(batch_size, num_threads, device_id)
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
            super().__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource(device="gpu")
            self.crop = ops.Crop(device="gpu", crop_h=32, crop_w=32, crop_pos_x=0.2, crop_pos_y=0.2)
            self.use_list = use_list

        def define_graph(self):
            self.batch = self.input()
            output = self.crop(self.batch)
            return output

        def iter_setup(self):
            if use_list:
                batch_data = [
                    cast_to(random_array([100, 100, 3]) * 256, datapy.uint8)
                    for _ in range(self.batch_size)
                ]
            else:
                batch_data = cast_to(
                    random_array([self.batch_size, 100, 100, 3]) * 256, datapy.uint8
                )
            self.feed_input(self.batch, batch_data, layout="HWC")

    for batch_size in [1, 10]:
        for use_list in (True, False):
            pipe = ExternalSourcePipeline(batch_size, 3, 0, use_list)
            pipe.run()


class TestIteratorZeroCopy:
    def __init__(self, n, batch_size, dims=[2], as_tensor=False, num_keep_samples=2):
        self.batch_size = batch_size
        self.dims = dims
        self.n = n
        self.as_tensor = as_tensor
        self.i = 0
        self.data = []
        self.num_keep_samples = num_keep_samples

    def __len__(self):
        return self.n

    def __iter__(self):
        # return a copy, so that the iteration number doesn't collide
        return TestIteratorZeroCopy(
            self.n, self.batch_size, self.dims, self.as_tensor, self.num_keep_samples
        )

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
                data = [generate(d) for d in self.dims]
            else:
                data = generate(self.dims)

            # it needs to keep data alive
            self.data.append(data)

            def add_one(x):
                if isinstance(x, list):
                    for elm in x:
                        elm = add_one(elm)
                else:
                    x += 1
                return x

            if len(self.data) > self.num_keep_samples:
                tmp = self.data.pop(0)
                # change popped data to make sure it is corrupted
                tmp = add_one(tmp)
            return data
        else:
            self.i = 0
            raise StopIteration

    next = __next__


def _test_iter_setup_zero_copy(use_fn_api, by_name, as_tensor, device, additional_num_keep_samples):
    batch_size = 7
    prefetch_queue_depth = 5

    class IterSetupPipeline(Pipeline):
        def __init__(self, iterator, num_threads, device_id, device, prefetch_queue_depth=2):
            super().__init__(
                batch_size=iterator.batch_size,
                num_threads=num_threads,
                device_id=device_id,
                prefetch_queue_depth=prefetch_queue_depth,
            )
            self.iterator = iterator
            self._device = device

        def define_graph(self):
            if use_fn_api:
                self.batch_1 = fn.external_source(device=self._device, name="src1", no_copy=True)
                self.batch_2 = fn.external_source(device=self._device, name="src2", no_copy=True)
            else:
                input_1 = ops.ExternalSource(device=self._device, no_copy=True)
                input_2 = ops.ExternalSource(device=self._device, no_copy=True)
                self.batch_1 = input_1(name="src1")
                self.batch_2 = input_2(name="src2")
            return [self.batch_1, self.batch_2]

        def iter_setup(self):
            batch_1, batch_2 = next(self.iterator)
            if by_name:
                self.feed_input("src1", batch_1)
                self.feed_input("src2", batch_2)
            else:
                self.feed_input(self.batch_1, batch_1)
                self.feed_input(self.batch_2, batch_2)

    iter_num = 10
    # it is enough to keep only ``prefetch_queue_depth`` or ``cpu_queue_depth * gpu_queue_depth``
    # (when they are not equal), but they are equal in this case
    num_keep_samples = prefetch_queue_depth + additional_num_keep_samples
    source = TestIteratorZeroCopy(
        iter_num, batch_size, [2, 3], as_tensor=as_tensor, num_keep_samples=num_keep_samples
    )
    pipe = IterSetupPipeline(iter(source), 3, 0, device, prefetch_queue_depth)

    if (device == "cpu" and not cpu_input) or (device == "gpu" and cpu_input):
        input_types = ["CPU", "GPU"]
        if device == "cpu" and not cpu_input:
            input_types.reverse()
        assert_raises(
            RuntimeError,
            pipe.run,
            glob="no_copy is supported only for the same data source device type as "
            "operator. Received: {} input for {} operator".format(*input_types),
        )
    elif additional_num_keep_samples < 0:
        # assert_raises doesn't work here for the assertions from the test_utils.py
        if_raised = False
        try:
            # this tests bases on the race condition. Running it 5 times make this race more
            # likely to happen and tests pass in CI under high CPU load
            iterations = 5
            for _ in range(iterations):
                run_and_check(pipe, source)
        except AssertionError:
            if_raised = True
        assert if_raised
    else:
        run_and_check(pipe, source)


def test_iter_setup_zero_copy():
    for use_fn_api in [False, True]:
        for by_name in [False, True]:
            for as_tensor in [False, True]:
                for device in ["cpu", "gpu"]:
                    # make it -4 as -1 sometimes fails due to being close to the limit
                    for additional_num_keep_samples in [-4, 0, 1]:
                        yield (
                            _test_iter_setup_zero_copy,
                            use_fn_api,
                            by_name,
                            as_tensor,
                            device,
                            additional_num_keep_samples,
                        )
