import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline

import numpy as np

from test_utils import np_types_to_dali


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


class Batch3DOverflow(Batch3D):
    def __init__(self, data_type):
        super().__init__(data_type)

        for batch in self._data:
            for sample in batch:
                sample *= 100000


def run_dali(reduce_fn, batch_fn, keep_dims, axes, output_type, add_mean_input = False):
    batch_size = batch_fn.batch_size()

    # Needed due to how ExternalSource API works. It fails on methods, partials.
    def get_batch():
        return batch_fn()

    result_cpu = []
    result_gpu = []

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)

    args = { 'keep_dims': keep_dims, 'axes': axes}
    if output_type is not None:
        args['dtype'] = np_types_to_dali(output_type)

    with pipe:
        input = fn.external_source(source = get_batch)
        if not add_mean_input:
            reduced_cpu = reduce_fn(input, **args)
            reduced_gpu = reduce_fn(input.gpu(), **args)
        else:
            mean = fn.reductions.mean(input, **args)
            reduced_cpu = reduce_fn(input, mean, **args)
            reduced_gpu = reduce_fn(input.gpu(), mean.gpu(), **args)
        pipe.set_outputs(reduced_cpu, reduced_gpu)

    pipe.build()

    for _ in range(batch_fn.num_iter()):
        output = pipe.run()
        reduced_cpu = output[0].as_array()
        reduced_gpu = output[1].as_cpu().as_array()
        result_cpu.append(reduced_cpu)
        result_gpu.append(reduced_gpu)
    
    return result_cpu, result_gpu


def run_numpy(reduce_fn, batch_fn, keep_dims, axes, output_type):
    result = []
    args = { 'keepdims': keep_dims, 'axis': axes}
    if output_type is not None:
        args['dtype'] = output_type

    for _ in range(batch_fn.num_iter()):
        batch = batch_fn()
        sample_result = []
        for sample in batch:
            sample_reduced = reduce_fn(sample, **args)
            sample_result.append(sample_reduced)

        result.append(sample_result)
    return result


def compare(dali_res, np_res):
    for dali_sample, np_sample in zip(dali_res, np_res):
        assert dali_sample.shape == np_sample.shape
        if dali_res[0].dtype == np.float32:
            assert np.allclose(dali_sample, np_sample)
        else:
            assert np.array_equal(dali_sample, np_sample)


def run_reduce(keep_dims, reduce_fns, batch_gen, input_type, output_type = None, add_mean_input = False):
    batch_fn = batch_gen(input_type)
    dali_reduce_fn, numpy_reduce_fn = reduce_fns

    for axes in batch_fn.valid_axes():
        dali_res_cpu, dali_res_gpu = run_dali(
            dali_reduce_fn, batch_fn, keep_dims = keep_dims, axes = axes, output_type = output_type, add_mean_input = add_mean_input)

        batch_fn.reset()

        np_res = run_numpy(
            numpy_reduce_fn, batch_fn, keep_dims = keep_dims, axes = axes, output_type = output_type)
        
        for iteration in range(batch_fn.num_iter()):
            compare(dali_res_cpu[iteration], np_res[iteration])
            compare(dali_res_gpu[iteration], np_res[iteration])


def test_reduce():
    reductions = [(fn.reductions.sum, np.sum), (fn.reductions.min, np.min), (fn.reductions.max, np.max)]

    batch_gens = [Batch1D, Batch2D, Batch3D]
    types = [
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32]

    for keep_dims in [False, True]:
        for reduce_fns in reductions:
            for batch_gen in batch_gens:
                for type_id in types:
                    yield run_reduce, keep_dims, reduce_fns, batch_gen, type_id


def mean_square(input, keepdims = False, axis = None, dtype = None):
    return np.mean(np.square(input), keepdims = keepdims, axis = axis, dtype = dtype)


def root_mean_square(input, keepdims = False, axis = None, dtype = None):
    return np.sqrt(mean_square(input, keepdims = keepdims, axis = axis, dtype = dtype))


def test_reduce_with_promotion():
    reductions = [
        (fn.reductions.root_mean_square, root_mean_square),
        (fn.reductions.mean_square, mean_square)]

    batch_gens = [Batch3D]
    types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32]

    for keep_dims in [False, True]:
        for reduce_fns in reductions:
            for batch_gen in batch_gens:
                for type_id in types:
                    yield run_reduce, keep_dims, reduce_fns, batch_gen, type_id


def test_reduce_with_promotion_with_overflow():
    reductions = [(fn.reductions.sum, np.sum), (fn.reductions.mean, np.mean)]

    batch_gens = [Batch3DOverflow]
    types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32]

    for keep_dims in [False, True]:
        for reduce_fns in reductions:
            for batch_gen in batch_gens:
                for type_id in types:
                    yield run_reduce, keep_dims, reduce_fns, batch_gen, type_id


def test_sum_with_output_type():
    reductions = [(fn.reductions.sum, np.sum)]

    batch_gens = [Batch3DOverflow]
    types = [
        (np.uint8, [np.uint64, np.float32]),
        (np.int8, [np.int64, np.float32]),
        (np.uint16, [np.uint64, np.float32]),
        (np.int16, [np.int64, np.float32]),
        (np.uint32, [np.uint64, np.float32]),
        (np.int32, [np.int32, np.int64, np.float32])]

    for keep_dims in [False, True]:
        for reduce_fns in reductions:
            for batch_gen in batch_gens:
                for type_map in types:
                    input_type = type_map[0]
                    for output_type in type_map[1]:
                        yield run_reduce, keep_dims, reduce_fns, batch_gen, input_type, output_type


def test_reduce_with_mean_input():
    reductions = [
        (fn.reductions.std, np.std),
        (fn.reductions.variance, np.var)]    

    batch_gens = [Batch1D, Batch2D, Batch3D]
    types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32]

    for keep_dims in [False, True]:
        for reduce_fns in reductions:
            for batch_gen in batch_gens:
                for type_id in types:
                    yield run_reduce, keep_dims, reduce_fns, batch_gen, type_id, None, True


if __name__ == "__main__":
    batch_fn = Batch1D(np.int32)
    batch_size = batch_fn.batch_size()

    # Needed due to how ExternalSource API works. It fails on methods, partials.
    def get_batch():
        return batch_fn()

    result_cpu = []
    result_gpu = []

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)

    args = { 'keep_dims': False, 'axes': None}
    np_args = { 'keepdims': False, 'axis': None}

    with pipe:
        input = fn.external_source(source = get_batch)
        mean = fn.reductions.mean(input, **args)
        reduced_cpu = fn.reductions.variance(input, mean, **args)
        reduced_gpu = fn.reductions.variance(input.gpu(), mean.gpu(), **args)
        pipe.set_outputs(reduced_cpu, reduced_gpu)
    pipe.build()

    for _ in range(batch_fn.num_iter()):
        output = pipe.run()
        reduced_cpu = output[0].as_array()
        reduced_gpu = output[1].as_cpu().as_array()
        print(reduced_cpu)
        print(reduced_gpu)

    batch_fn.reset()
    print()

    result = []

    for _ in range(batch_fn.num_iter()):
        batch = batch_fn()
        sample_result = []
        for sample in batch:
            sample_reduced = np.var(sample, **np_args)
            sample_result.append(sample_reduced)
        print(sample_result)

        result.append(sample_result)