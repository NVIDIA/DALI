import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops

from nvidia.dali.pipeline import Pipeline
import numpy as np


class Batch1D:
    def __init__(self, data_type):
        self._data_type = data_type

    def __call__(self):
        return [
            np.array([-1, -2,  -3, -4], dtype = self._data_type),
            np.array([99,  2, -10, 10], dtype = self._data_type)]

    def valid_axes(self):
        return [None, 0]


class Batch2D:
    def __init__(self, data_type):
        self._data_type = data_type

    def __call__(self):
        return [
            np.array([[ 1, 0,  2], [ 3, 1,  4]], dtype = self._data_type),
            np.array([[ 5, 0,  6], [ 7, 0, -8]], dtype = self._data_type),
            np.array([[-1, 0,  2], [-3, 1,  4]], dtype = self._data_type),
            np.array([[ 5, 0, -6], [ 7, 0,  8]], dtype = self._data_type)]

    def valid_axes(self):
        return [None, 0, 1, (0, 1)]


class Batch3D:
    def __init__(self, data_type):
        self._data_type = data_type

    def __call__(self):
        return [
            np.array([[[1, 0, 1],  [2, -3, 1]], [[0,  4, 1], [0, 4, 1]]], dtype = self._data_type),
            np.array([[[5, 0, 1],  [6,  7, 1]], [[0, -8, 1], [0, 4, 1]]], dtype = self._data_type)]

    def valid_axes(self):
        return [None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def run_dali(reduce_fn, batch_fn, keep_dims, axes):
    batch = batch_fn()
    batch_size = len(batch)

    # Needed due to how ExternalSource API works. It fails on methods, partials.
    def get_batch():
        return batch

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)

    with pipe:
        input = fn.external_source(
            source = get_batch, device = 'cpu')
        reduced = reduce_fn(
            input, device = 'cpu', keep_dims = keep_dims, axes = axes)
        pipe.set_outputs(reduced)

    pipe.build()
    output = pipe.run()
    reduced = output[0].as_array()
    
    return reduced


def run_numpy(reduce_fn, batch_fn, keep_dims, axes):
    batch = batch_fn()
    result = []
    for sample in batch:
        sample_sum = reduce_fn(sample, keepdims = keep_dims, axis = axes)

        # Numpy returns scalar value for full reduction. To match DALI, wrap it with an array
        if type(sample_sum) != np.ndarray:
            sample_sum = np.asarray([sample_sum])
        
        result.append(sample_sum)
    return result


def compare(dali_res, np_res):
    for dali_sample, np_sample in zip(dali_res, np_res):
        assert np.array_equal(dali_sample, np_sample)


def run_reduce(keep_dims, reduce_fns, batch_gen, data_type):
    batch_fn = batch_gen(data_type)
    dali_reduce_fn, numpy_reduce_fn = reduce_fns

    for axes in batch_fn.valid_axes():
        dali_res = run_dali(
            dali_reduce_fn, batch_fn, keep_dims = keep_dims, axes = axes)
        np_res = run_numpy(
            numpy_reduce_fn, batch_fn, keep_dims = keep_dims, axes = axes)

        compare(dali_res, np_res)


def test_reduce():
    reductions = [
        (fn.sum, np.sum),
        (fn.min, np.min),
        (fn.max, np.max)]

    batch_gens = [Batch1D, Batch2D, Batch3D]
    types = [np.float32, np.int32, np.int16]

    for keep_dims in [True, False]:
        for reduce_fns in reductions:
            for batch_gen in batch_gens:
                for type_id in types:
                    yield run_reduce, keep_dims, reduce_fns, batch_gen, type_id
