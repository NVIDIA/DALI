import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops

from nvidia.dali.pipeline import Pipeline
import numpy as np


class Batch:
    def __init__(self, data_type):
        self._data_type = data_type
        self._index = 0

    def __call__(self):
        batch = self._data[self._index]
        self._index = (self._index + 1 ) % self.batch_size()
        return batch

    def batch_size(self):
        return len(self._data[0])

    def num_iter(self):
        return 2 * len(self._data)

    def reset(self):
        self._index = 0    


class Batch1D(Batch):
    def __init__(self, data_type):
        super().__init__(data_type)
        self._data = [
            [
                np.array([ 1,  2,  3,  4], dtype = self._data_type),
                np.array([33,  2, 10, 10], dtype = self._data_type)
            ], [
                np.array([10, 20, 30, 20], dtype = self._data_type),
                np.array([33,  2, 15, 19], dtype = self._data_type)
            ]]

    def valid_axes(self):
        return [None, 0]


class Batch2D(Batch):
    def __init__(self, data_type):
        super().__init__(data_type)
        self._data = [
            [
                np.array([[  1,  0,  2], [  3,  1,  4]], dtype = self._data_type),
                np.array([[  5,  0,  6], [  7,  0,  8]], dtype = self._data_type)
            ], [
                np.array([[ 13, 23, 22], [ 23, 21, 14]], dtype = self._data_type),
                np.array([[ 23,  3,  6], [  7,  0, 20]], dtype = self._data_type)
            ]]

    def valid_axes(self):
        return [None, 0, 1, (0, 1)]


class Batch3D(Batch):
    def __init__(self, data_type):
        super().__init__(data_type)
        self._data = [
            [
                np.array([[[1, 0, 1], [2, 3, 1]], [[0, 4, 1], [0, 4, 1]]], dtype = self._data_type),
                np.array([[[5, 0, 1], [6, 7, 1]], [[0, 8, 1], [0, 4, 1]]], dtype = self._data_type)
            ], [
                np.array([[[9, 0, 3], [3, 3, 3]], [[7, 0, 3], [0, 6, 8]]], dtype = self._data_type),
                np.array([[[7, 2, 3], [7, 8, 2]], [[3, 9, 2], [2, 6, 2]]], dtype = self._data_type)
            ]]

    def valid_axes(self):
        return [None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def run_dali(reduce_fn, batch_fn, keep_dims, axes):
    batch_size = batch_fn.batch_size()

    # Needed due to how ExternalSource API works. It fails on methods, partials.
    def get_batch():
        return batch_fn()

    result_cpu = []
    result_gpu = []

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)

    with pipe:
        input = fn.external_source(source = get_batch)
        reduced_cpu = reduce_fn(
            input, keep_dims = keep_dims, axes = axes)
        reduced_gpu = reduce_fn(
            input.gpu(), keep_dims = keep_dims, axes = axes)
        pipe.set_outputs(reduced_cpu, reduced_gpu)

    pipe.build()

    for _ in range(batch_fn.num_iter()):
        output = pipe.run()
        reduced_cpu = output[0].as_array()
        reduced_gpu = output[1].as_cpu().as_array()
        result_cpu.append(reduced_cpu)
        result_gpu.append(reduced_gpu)
    
    return result_cpu, result_gpu


def run_numpy(reduce_fn, batch_fn, keep_dims, axes):
    result = []
    for _ in range(batch_fn.num_iter()):
        batch = batch_fn()
        sample_result = []
        for sample in batch:
            sample_reduced = reduce_fn(sample, keepdims = keep_dims, axis = axes)
            sample_result.append(sample_reduced)

        result.append(sample_result)
    return result


def compare(dali_res, np_res):
    for dali_sample, np_sample in zip(dali_res, np_res):
        assert np.array_equal(dali_sample, np_sample)


def run_reduce(keep_dims, reduce_fns, batch_gen, data_type):
    batch_fn = batch_gen(data_type)
    dali_reduce_fn, numpy_reduce_fn = reduce_fns

    for axes in batch_fn.valid_axes():
        dali_res_cpu, dali_res_gpu = run_dali(
            dali_reduce_fn, batch_fn, keep_dims = keep_dims, axes = axes)

        batch_fn.reset()

        np_res = run_numpy(
            numpy_reduce_fn, batch_fn, keep_dims = keep_dims, axes = axes)
        
        for iteration in range(batch_fn.num_iter()):
            compare(dali_res_cpu[iteration], np_res[iteration])
            compare(dali_res_gpu[iteration], np_res[iteration])


def test_reduce():
    reductions = [(fn.min, np.min), (fn.max, np.max)]
    batch_gens = [Batch1D, Batch2D, Batch3D]
    types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32]

    for keep_dims in [True, False]:
        for reduce_fns in reductions:
            for batch_gen in batch_gens:
                for type_id in types:
                    yield run_reduce, keep_dims, reduce_fns, batch_gen, type_id
