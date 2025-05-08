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

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline, pipeline_def
from nose_utils import nottest, assert_raises
from nose2.tools import params

import numpy as np

from test_utils import np_type_to_dali


class Batch:
    def __init__(self, data_type):
        self._data_type = data_type
        self._index = 0

    def __call__(self):
        batch = self._data[self._index]
        self._index = (self._index + 1) % self.batch_size()
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
                np.array([1, 2, 3, 4], dtype=self._data_type),
                np.array([33, 2, 10, 10], dtype=self._data_type),
            ],
            [
                np.array([10, 20, 30, 20], dtype=self._data_type),
                np.array([33, 2, 15, 19], dtype=self._data_type),
            ],
        ]

    def valid_axes(self):
        return [None, (), 0]


class Batch2D(Batch):
    def __init__(self, data_type):
        super().__init__(data_type)
        self._data = [
            [
                np.array([[1, 0, 2], [3, 1, 4]], dtype=self._data_type),
                np.array([[5, 0, 6], [7, 0, 8]], dtype=self._data_type),
            ],
            [
                np.array([[13, 23, 22], [23, 21, 14]], dtype=self._data_type),
                np.array([[23, 3, 6], [7, 0, 20]], dtype=self._data_type),
            ],
        ]

    def valid_axes(self):
        return [None, (), 0, 1, (0, 1)]


class Batch3D(Batch):
    def __init__(self, data_type):
        super().__init__(data_type)
        self._data = [
            [
                np.array([[[1, 0, 1], [2, 3, 1]], [[0, 4, 1], [0, 4, 1]]], dtype=self._data_type),
                np.array([[[5, 0, 1], [6, 7, 1]], [[0, 8, 1], [0, 4, 1]]], dtype=self._data_type),
            ],
            [
                np.array([[[9, 0, 3], [3, 3, 3]], [[7, 0, 3], [0, 6, 8]]], dtype=self._data_type),
                np.array([[[7, 2, 3], [7, 8, 2]], [[3, 9, 2], [2, 6, 2]]], dtype=self._data_type),
            ],
        ]

    def valid_axes(self):
        return [None, (), 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]


class Batch3DOverflow(Batch3D):
    def __init__(self, data_type):
        super().__init__(data_type)

        for batch in self._data:
            for sample in batch:
                sample *= 100000


class Batch3DNegativeAxes(Batch3D):
    def valid_axes(self):
        return [-3, -2, -1, (-3, 1), (0, -1), (-2, 2), (-3, -2, -1)]


def get_expected_layout(in_layout, axes, keep_dims):
    in_layout = in_layout or ""
    if keep_dims or not in_layout:
        return in_layout
    if axes is None:
        return ""
    if isinstance(axes, int):
        axes = [axes]
    ndim = len(in_layout)
    axes = [(axis + ndim) % ndim for axis in axes]
    return "".join(c for i, c in enumerate(in_layout) if i not in axes)


def check_layout(tensor, in_layout, axes, keep_dims):
    expected_layout = get_expected_layout(in_layout, axes, keep_dims)
    assert (
        tensor.layout() == expected_layout
    ), f"Layout mismatch. Got: `{tensor.layout()}`, expected `{expected_layout}` (axes: {axes})"


def run_dali(
    reduce_fn, batch_fn, keep_dims, axes, output_type, add_mean_input=False, ddof=0, layout=None
):
    batch_size = batch_fn.batch_size()

    # Needed due to how ExternalSource API works. It fails on methods, partials.
    def get_batch():
        return batch_fn()

    result_cpu = []
    result_gpu = []

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)

    args = {"keep_dims": keep_dims, "axes": axes}
    if output_type is not None:
        args["dtype"] = np_type_to_dali(output_type)

    with pipe:
        input = fn.external_source(source=get_batch, layout=layout)
        if not add_mean_input:
            reduced_cpu = reduce_fn(input, **args)
            reduced_gpu = reduce_fn(input.gpu(), **args)
        else:
            mean = fn.reductions.mean(input, **args)
            args["ddof"] = ddof
            reduced_cpu = reduce_fn(input, mean, **args)
            reduced_gpu = reduce_fn(input.gpu(), mean.gpu(), **args)
        pipe.set_outputs(reduced_cpu, reduced_gpu)

    for _ in range(batch_fn.num_iter()):
        output = pipe.run()
        check_layout(output[0], layout, axes, keep_dims)
        check_layout(output[1], layout, axes, keep_dims)
        reduced_cpu = output[0].as_array()
        reduced_gpu = output[1].as_cpu().as_array()
        result_cpu.append(reduced_cpu)
        result_gpu.append(reduced_gpu)

    return result_cpu, result_gpu


def run_numpy(reduce_fn, batch_fn, keep_dims, axes, output_type, ddof=None):
    result = []
    args = {"keepdims": keep_dims, "axis": axes}
    if output_type is not None:
        args["dtype"] = output_type

    if ddof is not None:
        args["ddof"] = ddof

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
            if not np.array_equal(dali_sample, np_sample):
                print(dali_sample)
                print(np_sample)
                assert np.array_equal(dali_sample, np_sample)


def np_mean_square(input, keepdims=False, axis=None, dtype=None):
    return np.mean(np.square(input), keepdims=keepdims, axis=axis, dtype=dtype)


def np_root_mean_square(input, keepdims=False, axis=None, dtype=None):
    return np.sqrt(np_mean_square(input, keepdims=keepdims, axis=axis, dtype=dtype))


reduce_fns = {
    "sum": (fn.reductions.sum, np.sum),
    "min": (fn.reductions.min, np.min),
    "max": (fn.reductions.max, np.max),
    "mean": (fn.reductions.mean, np.mean),
    "mean_square": (fn.reductions.mean_square, np_mean_square),
    "rms": (fn.reductions.rms, np_root_mean_square),
    "std_dev": (fn.reductions.std_dev, np.std),
    "variance": (fn.reductions.variance, np.var),
}


def run_reduce(keep_dims, reduction_name, batch_gen, input_type, output_type=None, layout=None):
    batch_fn = batch_gen(input_type)
    dali_reduce_fn, numpy_reduce_fn = reduce_fns[reduction_name]

    for axes in batch_fn.valid_axes():
        dali_res_cpu, dali_res_gpu = run_dali(
            dali_reduce_fn,
            batch_fn,
            keep_dims=keep_dims,
            axes=axes,
            output_type=output_type,
            layout=layout,
        )

        batch_fn.reset()

        np_res = run_numpy(
            numpy_reduce_fn, batch_fn, keep_dims=keep_dims, axes=axes, output_type=output_type
        )

        for iteration in range(batch_fn.num_iter()):
            compare(dali_res_cpu[iteration], np_res[iteration])
            compare(dali_res_gpu[iteration], np_res[iteration])


def test_reduce():
    reductions = ["sum", "min", "max"]
    batch_gens = [Batch1D, Batch2D, Batch3D]
    types = [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float32,
    ]

    rng = np.random.default_rng(1000)
    for keep_dims in [False, True]:
        for reduction_name in reductions:
            for ndim, batch_gen in enumerate(batch_gens, start=1):
                type_id = rng.choice(types)
                layout = rng.choice([None, "XYZ"[:ndim]])
                yield run_reduce, keep_dims, reduction_name, batch_gen, type_id, None, layout


def test_reduce_negative_axes():
    reductions = ["sum", "max"]
    type = np.uint8

    for layout in ["FGH", None]:
        for keep_dims in [False, True]:
            for reduction_name in reductions:
                yield run_reduce, keep_dims, reduction_name, Batch3DNegativeAxes, type, None, layout


def test_reduce_invalid_axes():
    class Batch3DInvalidAxes(Batch3D):
        def valid_axes(self):  # Invalid axes
            return [-100, (100, 0)]

    batch_fn = Batch3DInvalidAxes(np.uint8)
    dali_reduce_fn, numpy_reduce_fn = reduce_fns["sum"]

    for axes in batch_fn.valid_axes():
        with assert_raises(IndexError, glob="Axis index out of range"):
            dali_res_cpu, dali_res_gpu = run_dali(
                dali_reduce_fn, batch_fn, keep_dims=False, axes=axes, output_type=np.uint8
            )


def test_reduce_with_promotion():
    reductions = ["rms", "mean_square"]

    batch_gens = [Batch3D]
    types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32]

    rng = np.random.default_rng(1041)
    for keep_dims in [False, True]:
        for reduction_name in reductions:
            for batch_gen in batch_gens:
                for type_id in types:
                    layout = rng.choice([None, "ABC"])
                    yield run_reduce, keep_dims, reduction_name, batch_gen, type_id, None, layout


def test_reduce_with_promotion_with_overflow():
    reductions = ["sum", "mean"]

    batch_gens = [Batch3DOverflow]
    types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32]

    rng = np.random.default_rng(1042)
    for keep_dims in [False, True]:
        for reduction_name in reductions:
            for batch_gen in batch_gens:
                for type_id in types:
                    layout = rng.choice([None, "ABC"])
                    yield run_reduce, keep_dims, reduction_name, batch_gen, type_id, None, layout


def test_sum_with_output_type():
    reductions = ["sum"]

    batch_gens = [Batch3DOverflow]
    types = [
        (np.uint8, [np.uint64, np.float32]),
        (np.int8, [np.int64, np.float32]),
        (np.uint16, [np.uint64, np.float32]),
        (np.int16, [np.int64, np.float32]),
        (np.uint32, [np.uint64, np.float32]),
        (np.int32, [np.int32, np.int64, np.float32]),
    ]

    rng = np.random.default_rng(1043)
    for reduction_name in reductions:
        for batch_gen in batch_gens:
            for type_map in types:
                input_type = type_map[0]
                keep_dims = np.random.choice([False, True])
                for output_type in type_map[1]:
                    layout = rng.choice([None, "RGB"])
                    yield (
                        run_reduce,
                        keep_dims,
                        reduction_name,
                        batch_gen,
                        input_type,
                        output_type,
                        layout,
                    )


def run_reduce_with_mean_input(
    keep_dims, reduction_name, batch_gen, input_type, output_type=None, layout=None
):
    batch_fn = batch_gen(input_type)
    dali_reduce_fn, numpy_reduce_fn = reduce_fns[reduction_name]

    for axes in batch_fn.valid_axes():
        if axes == ():
            valid_ddofs = [0]
        elif axes is None:
            valid_ddofs = [0, 1, 2, 3]
        else:
            valid_ddofs = [0, 1]
        for ddof in valid_ddofs:
            dali_res_cpu, dali_res_gpu = run_dali(
                dali_reduce_fn,
                batch_fn,
                keep_dims=keep_dims,
                axes=axes,
                output_type=output_type,
                add_mean_input=True,
                ddof=ddof,
                layout=layout,
            )

            batch_fn.reset()

            np_res = run_numpy(
                numpy_reduce_fn,
                batch_fn,
                keep_dims=keep_dims,
                axes=axes,
                output_type=output_type,
                ddof=ddof,
            )

            for iteration in range(batch_fn.num_iter()):
                compare(dali_res_cpu[iteration], np_res[iteration])
                compare(dali_res_gpu[iteration], np_res[iteration])


def test_reduce_with_mean_input():
    reductions = ["std_dev", "variance"]

    batch_gens = [Batch1D, Batch2D, Batch3D]
    types = [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float32,
    ]

    rng = np.random.default_rng(1044)
    for keep_dims in [False, True]:
        for reduction_name in reductions:
            for ndim, batch_gen in enumerate(batch_gens, start=1):
                type_id = np.random.choice(types)
                layout = rng.choice([None, "CDE"[:ndim]])
                yield (
                    run_reduce_with_mean_input,
                    keep_dims,
                    reduction_name,
                    batch_gen,
                    type_id,
                    None,
                    layout,
                )


def run_and_compare_with_layout(batch_gen, pipe):
    for _ in range(batch_gen.num_iter()):
        output = pipe.run()
        assert (
            output[0].layout() == output[1].layout()
        ), f"{output[0].layout()} vs {output[1].layout()}"
        reduced = output[0].as_array()
        reduced_by_name = output[1].as_array()

        assert np.array_equal(reduced, reduced_by_name)


def run_reduce_with_layout(batch_size, get_batch, reduction, axes, axis_names, batch_fn):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    with pipe:
        input = fn.external_source(source=get_batch, layout="ABC")
        reduced = reduction(input, keep_dims=False, axes=axes)
        reduced_by_name = reduction(input, keep_dims=False, axis_names=axis_names)

    pipe.set_outputs(reduced, reduced_by_name)

    run_and_compare_with_layout(batch_fn, pipe)


def run_reduce_with_layout_with_mean_input(
    batch_size, get_batch, reduction, axes, axis_names, batch_fn
):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    with pipe:
        input = fn.external_source(source=get_batch, layout="ABC")
        mean = fn.reductions.mean(input, axes=axes)
        reduced = reduction(input, mean, keep_dims=False, axes=axes)
        reduced_by_name = reduction(input, mean, keep_dims=False, axis_names=axis_names)

    pipe.set_outputs(reduced, reduced_by_name)

    run_and_compare_with_layout(batch_fn, pipe)


def test_reduce_axis_names():
    reductions = [
        fn.reductions.max,
        fn.reductions.min,
        fn.reductions.mean,
        fn.reductions.mean_square,
        fn.reductions.sum,
        fn.reductions.rms,
    ]

    reductions_with_mean_input = [fn.reductions.std_dev, fn.reductions.variance]

    batch_fn = Batch3D(np.float32)
    batch_size = batch_fn.batch_size()

    def get_batch():
        return batch_fn()

    axes_and_names = [
        ((), ""),
        (0, "A"),
        (1, "B"),
        (2, "C"),
        ((0, 1), "AB"),
        ((0, 2), "AC"),
        ((1, 2), "BC"),
        ((0, 1, 2), "ABC"),
    ]

    for axes, axis_names in axes_and_names:
        for reduction in reductions:
            yield (
                run_reduce_with_layout,
                batch_size,
                get_batch,
                reduction,
                axes,
                axis_names,
                batch_fn,
            )
        for reduction in reductions_with_mean_input:
            yield (
                run_reduce_with_layout_with_mean_input,
                batch_size,
                get_batch,
                reduction,
                axes,
                axis_names,
                batch_fn,
            )


_random_buf = None
_random_lo = 0
_random_hi = 1


def fast_large_random_batches(rank, batch_size, num_batches, lo=0, hi=1):
    max_vol = 10000000
    max_extent = min(65536, int(np.floor(max_vol ** (1 / rank))))

    # generate a maximum size buffer pre-filled with random numbers
    global _random_buf
    global _random_lo
    global _random_hi
    should_generate = (
        _random_buf is None
        or _random_buf.size < max_extent**rank
        or _random_lo != lo
        or _random_hi != hi
    )
    if should_generate:
        _random_lo = lo
        _random_hi = hi
        _random_buf = np.random.uniform(low=lo, high=hi, size=max_vol).astype(np.float32)

    data = []
    for _ in range(num_batches):
        batch = []
        for _ in range(batch_size):
            size = np.random.randint(1, max_extent, size=rank)
            vol = np.prod(size)
            # now that we know the actual volume of the sample, we can pick a random
            # location in the pre-filled buffer
            offset = np.random.randint(0, (_random_buf.size - vol) + 1)
            # take a slice and reshape it to the desired shape - these are constant time operations
            sample = _random_buf[offset : offset + vol].reshape(size)
            batch.append(sample)
        data.append(batch)
    return data


@nottest
def _test_reduce_large_data(rank, axes, device, in_layout):
    batch_size = 16
    num_batches = 2
    data = fast_large_random_batches(rank, batch_size, num_batches)

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0 if device == "gpu" else None)
    input = fn.external_source(data, cycle=True, device=device, layout=in_layout)
    reduced = fn.reductions.sum(input, axes=axes)
    pipe.set_outputs(reduced)

    for b, batch in enumerate(data):
        (out,) = pipe.run()
        check_layout(out, in_layout, axes, False)
        if device == "gpu":
            out = out.as_cpu()
        for i in range(batch_size):
            ref = np.sum(batch[i].astype(np.float64), axis=axes)
            assert np.allclose(out[i], ref, 1e-5, 1e-5)


def test_reduce_large_data():
    np.random.seed(12344)
    for device in ["cpu", "gpu"]:
        for rank in [1, 2, 3]:
            for axis_mask in range(1, 2**rank):
                layout = np.random.choice([None, "DALI"[:rank]])
                axes = tuple(
                    filter(
                        lambda x: x >= 0, (i if axis_mask & (1 << i) else -1 for i in range(rank))
                    )
                )
                yield _test_reduce_large_data, rank, axes, device, layout


def empty_batches(rank, axes, batch_size, num_batches):
    data = []
    for _ in range(num_batches):
        batch = []
        for _ in range(batch_size):
            shape = np.random.randint(1, 10, size=rank)
            for a in axes:
                shape[a] = 0
            sample = np.empty(shape, dtype=np.float32)
            batch.append(sample)
        data.append(batch)
    return data


@nottest
def _test_reduce_empty_data(rank, axes, device, in_layout):
    batch_size = 16
    num_batches = 2
    data = empty_batches(rank, axes, batch_size, num_batches)

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0 if device == "gpu" else None)
    input = fn.external_source(data, cycle=True, device=device, layout=in_layout)
    reduced = fn.reductions.sum(input, axes=axes)
    pipe.set_outputs(reduced)

    for b, batch in enumerate(data):
        (out,) = pipe.run()
        check_layout(out, in_layout, axes, False)
        if device == "gpu":
            out = out.as_cpu()
        for i in range(batch_size):
            ref = np.sum(batch[i].astype(np.float64), axis=axes)
            assert np.allclose(out[i], ref, 1e-5, 1e-5)


def test_reduce_empty_data():
    np.random.seed(12344)
    for device in ["cpu", "gpu"]:
        for rank in [1, 2, 3]:
            for axis_mask in range(1, 2**rank):
                layout = np.random.choice([None, "DALI"[:rank]])
                axes = tuple(
                    filter(
                        lambda x: x >= 0, (i if axis_mask & (1 << i) else -1 for i in range(rank))
                    )
                )
                yield _test_reduce_empty_data, rank, axes, device, layout


@nottest
def _test_std_dev_large_data(rank, axes, device, in_layout):
    batch_size = 16
    num_batches = 2
    data = fast_large_random_batches(rank, batch_size, num_batches)

    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0 if device == "gpu" else None)
    input = fn.external_source(data, cycle=True, device=device, layout=in_layout)
    mean = fn.reductions.mean(input, axes=axes)
    reduced = fn.reductions.std_dev(input, mean, axes=axes, ddof=0)
    pipe.set_outputs(reduced)

    for b, batch in enumerate(data):
        (out,) = pipe.run()
        check_layout(out, in_layout, axes, False)
        if device == "gpu":
            out = out.as_cpu()
        for i in range(batch_size):
            ref = np.std(batch[i].astype(np.float64), axis=axes, ddof=0)
            assert np.allclose(out[i], ref, 1e-5, 1e-5)


def test_std_dev_large_data():
    np.random.seed(12344)
    for device in ["cpu", "gpu"]:
        for rank in [1, 2, 3, 4]:
            for axis_mask in range(1, 2**rank):
                layout = np.random.choice([None, "DALI"[:rank]])
                axes = tuple(
                    filter(
                        lambda x: x >= 0, (i if axis_mask & (1 << i) else -1 for i in range(rank))
                    )
                )
                yield _test_std_dev_large_data, rank, axes, device, layout


@params(
    ("cpu", (0,)),
    ("cpu", (1,)),
    ("cpu", (0, 1)),
    ("gpu", (0,)),
    ("gpu", (1,)),
    ("gpu", (0, 1)),
)
def test_sum_degenerate_data(device, axes):
    data = [
        np.ones(shape=(5, 3), dtype=np.int32),
        np.ones(shape=(3, 4), dtype=np.int32),
    ]

    @pipeline_def(batch_size=2, num_threads=2, prefetch_queue_depth=1)
    def pipe():
        d = fn.external_source(data, cycle=True, batch=False, device=device)
        r = fn.reductions.sum(d, axes=axes)
        return r

    p = pipe()

    # This is a bugfix test.
    # Before the fix the data was not written at all, so we need to write something (nonzero) to the
    # outputs first, then switch to degenerate inputs and re-check if it's all zero now.

    # first run with normal inputs
    (o,) = p.run()
    o = o.as_cpu()
    assert np.array_equal(o[0], np.sum(data[0], axis=axes))
    assert np.array_equal(o[1], np.sum(data[1], axis=axes))

    # switch to degenerate inputs
    data[0] = np.ones(shape=(0, 2), dtype=np.int32)
    data[1] = np.ones(shape=(0, 5), dtype=np.int32)

    # ... and re-run
    (o,) = p.run()
    o = o.as_cpu()
    assert np.array_equal(o[0], np.sum(data[0], axis=axes))
    assert np.array_equal(o[1], np.sum(data[1], axis=axes))
