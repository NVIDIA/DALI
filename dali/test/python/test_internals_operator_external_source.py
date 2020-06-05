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

cupy_used = False

def use_cupy():
    global cp
    global cupy_used
    global datapy
    import cupy as cp
    cupy_used = True
    datapy = cp

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
        datapy.random.seed(12345 * self.i + 4321)
        def generate(dim):
            shape = datapy.random.randint(1, 10, [dim]).tolist()
            if self.as_tensor:
                return datapy.random.ranf([self.batch_size] + shape)
            else:
                return [datapy.random.ranf(shape) for _ in range(self.batch_size)]
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
            if cupy_used:
                # convert cupy to numpy for the verification needs
                if isinstance(data, Iterable):
                    if isinstance(data[0], Iterable):
                        data = [[cp.asnumpy(d) for d in dd] for dd in data]
                    else:
                        data = [cp.asnumpy(d) for d in data]
                else:
                    data = cp.asnumpy(data)
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
        [datapy.array([1.5,2.5], dtype= datapy.float32)],
        [datapy.array([-1, 3.5,4.5], dtype= datapy.float32)]
    ]

    pipe.set_outputs(fn.external_source(batches))
    pipe.build()
    run_and_check(pipe, batches)


def test_external_source_collection_cycling():
    pipe = Pipeline(1, 3, 0)

    batches = [
        [datapy.array([1.5,2.5], dtype= datapy.float32)],
        [datapy.array([-1, 3.5,4.5], dtype= datapy.float32)]
    ]

    pipe.set_outputs(fn.external_source(batches, cycle = True))
    pipe.build()

    # epochs are cycles over the source iterable
    for _ in range(3):
        for batch in batches:
            if cupy_used:
                batch = cp.asnumpy(batch)
            check_output(pipe.run(), batch)

def test_external_source_with_iter():
    pipe = Pipeline(1, 3, 0)

    pipe.set_outputs(fn.external_source(lambda i: [datapy.array([i + 1.5], dtype=datapy.float32)]))
    pipe.build()

    for i in range(10):
        check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])

def test_external_source_generator():
    pipe = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [datapy.array([i + 1.5], dtype=datapy.float32)]

    pipe.set_outputs(fn.external_source(gen()))
    pipe.build()

    for i in range(5):
        check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])

def test_external_source_gen_function_cycle():
    pipe = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [datapy.array([i + 1.5], dtype=datapy.float32)]

    pipe.set_outputs(fn.external_source(gen, cycle = True))
    pipe.build()

    for _ in range(3):
        for i in range(5):
            check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])

def test_external_source_generator_cycle_error():
    _ = Pipeline(1, 3, 0)

    def gen():
        for i in range(5):
            yield [datapy.array([i + 1.5], dtype=datapy.float32)]

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
                batch_1.append(datapy.arange(0, 1, dtype=datapy.float))
                batch_2.append(datapy.arange(0, 1, dtype=datapy.float))
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

def external_data_veri(external_data):
    pass

def test_external_source_scalar_list():
    class ExternalSourcePipeline(Pipeline):
        def __init__(self, batch_size, external_data, num_threads, device_id, label_data):
            super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.ExternalSource()
            self.batch_size_ = batch_size
            self.external_data = external_data
            self.label_data_ = label_data

        def define_graph(self):
            self.batch = self.input()
            return [self.batch]

        def iter_setup(self):
            batch = []
            for elm in self.external_data:
                batch.append(datapy.array(elm, dtype=datapy.uint8))
            self.feed_input(self.batch, batch)

    batch_size = 3
    label_data = 10
    lists = []
    scalars = []
    for i in range(batch_size):
        lists.append([label_data + i])
        scalars.append(label_data + i * 10)
    for external_data in [lists, scalars]:
        print(external_data)
        pipe = ExternalSourcePipeline(batch_size, external_data, 3, 0, label_data)
        pipe.build()
        for _ in range(10):
            out = pipe.run()
            for i in range(batch_size):
                assert out[0].as_array()[i] == external_data[i]
        yield external_data_veri, external_data

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
                batch_data = [datapy.random.rand(100, 100, 3) for _ in range(self.batch_size)]
            else:
                batch_data = datapy.random.rand(self.batch_size, 100, 100, 3)
            self.feed_input(self.batch, batch_data)

    for batch_size in [1, 10]:
        for use_list in (True, False):
            pipe = ExternalSourcePipeline(batch_size, 3, 0, use_list)
            pipe.build()
            try:
                pipe.run()
            except RuntimeError:
                if not use_list:
                    assert(1), "For tensor list GPU external source should fail"