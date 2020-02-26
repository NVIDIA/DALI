from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.backend as backend
import numpy as np
from nose.tools import assert_raises
import random

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
        np.random.seed(12345 * self.i + 4321)
        def generate(dim):
            shape = np.random.randint(1, 10, [dim]).tolist()
            if self.as_tensor:
                return np.random.ranf([self.batch_size] + shape)
            else:
                return [np.random.ranf(shape) for _ in range(self.batch_size)]
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

def check_output(outputs, ref_out, ref_is_list_of_outputs = None):
    if ref_is_list_of_outputs is None:
        ref_is_list_of_outputs = len(outputs) > 1

    assert(ref_is_list_of_outputs or (len(outputs) == 1))

    for idx in range(len(outputs)):
        out = outputs[idx]
        ref = ref_out[idx] if ref_is_list_of_outputs else ref_out
        if isinstance(out, backend.TensorListGPU):
            out = out.as_cpu()
        for i in range(len(out)):
            if not np.array_equal(out[i], ref[i]):
                print("Out: ", out.at(i))
                print("Ref: ", ref[i])
            assert(np.array_equal(out[i], ref[i]))

def run_and_check(pipe, ref_iterable):
    iter_ref = iter(ref_iterable)
    i = 0
    while True:
        try:
            pipe_out = pipe.run()
            check_output(pipe_out, next(iter_ref))
            i += 1
        except StopIteration:
            break
    assert(i == len(ref_iterable))

def _test_iter_setup(use_fn_api, by_name, device):
    batch_size = 7
    class IterSetupPipeline(Pipeline):
        def __init__(self, iterator, num_threads, device_id):
            super(IterSetupPipeline, self).__init__(
                batch_size = iterator.batch_size,
                num_threads = num_threads,
                device_id = device_id)

            self.iterator = iterator

        def define_graph(self):
            if use_fn_api:
                self.batch_1 = fn.external_source(name = "src1")
                self.batch_2 = fn.external_source(name = "src2")
            else:
                input_1 = ops.ExternalSource()
                input_2 = ops.ExternalSource()
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
    source = TestIterator(iter_num, batch_size, [2, 3], device == "gpu")
    pipe = IterSetupPipeline(iter(source), 3, 0)
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
    iter_in = iter(source)

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

    # this should produce a two-element list of Tensor(Lists), the first
    # being 2D, the second being 3D (+ batch dimension)
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
        [np.array([1.5,2.5], dtype= np.float32)],
        [np.array([-1, 3.5,4.5], dtype= np.float32)]
    ]

    pipe.set_outputs(fn.external_source(batches))
    pipe.build()
    run_and_check(pipe, batches)


def test_external_source_collection_cycling():
    pipe = Pipeline(1, 3, 0)

    batches = [
        [np.array([1.5,2.5], dtype= np.float32)],
        [np.array([-1, 3.5,4.5], dtype= np.float32)]
    ]

    pipe.set_outputs(fn.external_source(batches, cycle = True))
    pipe.build()

    for epoch in range(3):
        for batch in batches:
            check_output(pipe.run(), batch)

def test_external_source_with_iter():
    pipe = Pipeline(1, 3, 0)

    pipe.set_outputs(fn.external_source(lambda i: [np.array([i + 1.5], dtype=np.float32)]))
    pipe.build()

    for i in range(10):
        check_output(pipe.run(), [np.array([i + 1.5], dtype=np.float32)])
