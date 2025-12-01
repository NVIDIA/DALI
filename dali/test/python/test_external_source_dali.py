# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import random
import functools
import inspect
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from nvidia.dali import Pipeline, pipeline_def
from test_utils import check_batch
from nose_utils import raises, assert_warns, assert_raises
from nvidia.dali.types import DALIDataType
from numpy.random import default_rng


def build_src_pipe(device, layout=None):
    if layout is None:
        layout = "XY"
    batches = [
        [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
        ],
        [
            np.array([[9, 10], [11, 12]], dtype=np.float32),
            np.array([[100, 200, 300, 400, 500]], dtype=np.float32),
        ],
    ]

    src_pipe = Pipeline(len(batches), 1, 0)
    out_batches = fn.external_source(source=batches, device=device, cycle=True, layout=layout)
    src_pipe.set_outputs(out_batches)
    return src_pipe, len(batches)


def _test_feed_input(device, is_serialized):
    src_pipe, batch_size = build_src_pipe(device)

    dst_pipe = Pipeline(batch_size, 1, 0, exec_async=False, exec_pipelined=False)
    dst_pipe.set_outputs(fn.external_source(name="ext", device=device))
    if is_serialized:
        serialized = dst_pipe.serialize()
        dst_pipe = None
        dst_pipe = Pipeline.deserialize(
            serialized_pipeline=serialized,
            batch_size=batch_size,
            num_threads=1,
            device_id=0,
            exec_async=False,
            exec_pipelined=False,
        )

    for _ in range(3):
        out1 = src_pipe.run()
        dst_pipe.feed_input("ext", out1[0])
        out2 = dst_pipe.run()
        check_batch(out2[0], out1[0], batch_size, 0, 0, "XY")


def test_feed_input():
    for device in ["cpu", "gpu"]:
        for is_serialized in [True, False]:
            yield _test_feed_input, device, is_serialized


def _test_callback(device, as_tensors, change_layout_to=None):
    src_pipe, batch_size = build_src_pipe(device)
    ref_pipe, batch_size = build_src_pipe(device, layout=change_layout_to)

    dst_pipe = Pipeline(batch_size, 1, 0)

    def get_from_src():
        tl = src_pipe.run()[0]
        return [tl[i] for i in range(len(tl))] if as_tensors else tl

    outs = fn.external_source(source=get_from_src, device=device, layout=change_layout_to)
    dst_pipe.set_outputs(outs)

    for iter in range(3):
        ref = ref_pipe.run()
        out = dst_pipe.run()
        check_batch(out[0], ref[0], batch_size, 0, 0)


def test_callback():
    for device in ["cpu", "gpu"]:
        for as_tensors in [False, True]:
            for change_layout in [None, "AB"]:
                yield _test_callback, device, as_tensors, change_layout


def _test_scalar(device, as_tensors):
    """Test propagation of scalars from external source"""
    batch_size = 4
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext = fn.external_source(
        source=lambda i: [np.float32(i * 10 + i + 1) for i in range(batch_size)], device=device
    )
    src_pipe.set_outputs(src_ext)

    dst_pipe = Pipeline(batch_size, 1, 0, exec_async=False, exec_pipelined=False)
    dst_pipe.set_outputs(fn.external_source(name="ext", device=device))

    for iter in range(3):
        src = src_pipe.run()
        data = src[0]
        if as_tensors:
            data = [data[i] for i in range(len(data))]
        dst_pipe.feed_input("ext", data)
        dst = dst_pipe.run()
        check_batch(src[0], dst[0], batch_size, 0, 0, "")


def test_scalar():
    for device in ["cpu", "gpu"]:
        for as_tensors in [False, True]:
            yield _test_scalar, device, as_tensors


class BatchCb:
    def __init__(self, batch_info, batch_size, epoch_size):
        self.batch_info = batch_info
        self.batch_size = batch_size
        self.epoch_size = epoch_size

    def __call__(self, arg):
        if self.batch_info:
            assert isinstance(
                arg, types.BatchInfo
            ), f"Expected BatchInfo instance as cb argument, got {arg}"
            iteration = arg.iteration
            epoch_idx = arg.epoch_idx
        else:
            assert isinstance(arg, int), "Expected integer as cb argument, got {}".format(arg)
            iteration = arg
            epoch_idx = -1
        if iteration >= self.epoch_size:
            raise StopIteration
        return [np.array([iteration, epoch_idx], dtype=np.int32) for _ in range(self.batch_size)]


class SampleCb:
    def __init__(self, batch_size, epoch_size):
        self.batch_size = batch_size
        self.epoch_size = epoch_size

    def __call__(self, sample_info):
        if sample_info.iteration >= self.epoch_size:
            raise StopIteration
        return np.array(
            [
                sample_info.idx_in_epoch,
                sample_info.idx_in_batch,
                sample_info.iteration,
                sample_info.epoch_idx,
            ],
            dtype=np.int32,
        )


def _test_batch_info_flag_default(cb, batch_size):
    pipe = Pipeline(batch_size, 1, 0)
    with pipe:
        ext = fn.external_source(source=cb)
        pipe.set_outputs(ext)
    pipe.run()


def test_batch_info_flag_default():
    batch_size = 5
    cb_int = BatchCb(False, batch_size, 1)
    yield _test_batch_info_flag_default, cb_int, batch_size
    cb_batch_info = BatchCb(True, batch_size, 1)
    yield raises(AssertionError, "Expected BatchInfo instance as cb argument")(
        _test_batch_info_flag_default
    ), cb_batch_info, batch_size


def _test_epoch_idx(batch_size, epoch_size, cb, batch_info, batch_mode):
    num_epochs = 3
    pipe = Pipeline(batch_size, 1, 0)
    with pipe:
        ext = fn.external_source(source=cb, batch_info=batch_info, batch=batch_mode)
        pipe.set_outputs(ext)
    for epoch_idx in range(num_epochs):
        for iteration in range(epoch_size):
            (batch,) = pipe.run()
            assert len(batch) == batch_size
            for sample_i, sample in enumerate(batch):
                if batch_mode:
                    expected = np.array([iteration, epoch_idx if batch_info else -1])
                else:
                    expected = np.array(
                        [iteration * batch_size + sample_i, sample_i, iteration, epoch_idx]
                    )
                np.testing.assert_array_equal(sample, expected)
        try:
            pipe.run()
        except StopIteration:
            pipe.reset()
        else:
            assert False, "expected StopIteration"


def test_epoch_idx():
    batch_size = 3
    epoch_size = 4
    for batch_info in (True, False):
        batch_cb = BatchCb(batch_info, batch_size, epoch_size)
        yield _test_epoch_idx, batch_size, epoch_size, batch_cb, batch_info, True
    sample_cb = SampleCb(batch_size, epoch_size)
    yield _test_epoch_idx, batch_size, epoch_size, sample_cb, None, False


def test_dtype_arg():
    batch_size = 2
    src_data = [[np.ones((120, 120, 3), dtype=np.uint8)] * batch_size]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext = fn.external_source(source=src_data, dtype=DALIDataType.UINT8)
    src_pipe.set_outputs(src_ext)
    (out,) = src_pipe.run()
    for i in range(batch_size):
        t = out.at(i)
        assert t.dtype == np.uint8
        np.array_equal(t, np.ones((120, 120, 3), dtype=np.uint8))


def test_dtype_arg_multioutput():
    batch_size = 2
    src_data = [
        [
            [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
            [np.ones((120, 120, 3), dtype=np.float32)] * batch_size,
        ]
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext, src_ext2 = fn.external_source(
        source=src_data, num_outputs=2, dtype=[DALIDataType.UINT8, DALIDataType.FLOAT]
    )
    src_pipe.set_outputs(src_ext, src_ext2)
    out1, out2 = src_pipe.run()
    for i in range(batch_size):
        t1 = out1.at(i)
        t2 = out2.at(i)
        assert t1.dtype == np.uint8
        assert np.array_equal(t1, np.ones((120, 120, 3), dtype=np.uint8))
        assert t2.dtype == np.float32
        assert np.allclose(t2, [np.ones((120, 120, 3), dtype=np.float32)])


@raises(RuntimeError, glob="ExternalSource expected data of type uint8 and got: float")
def test_incorrect_dtype_arg():
    batch_size = 2
    src_data = [[np.ones((120, 120, 3), dtype=np.float32)] * batch_size]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext = fn.external_source(source=src_data, dtype=DALIDataType.UINT8)
    src_pipe.set_outputs(src_ext)
    src_pipe.run()


@raises(
    RuntimeError,
    glob="Type of the data fed to the external source has changed from the "
    "previous iteration. Type in the previous iteration was float and "
    "the current type is uint8.",
)
def test_changing_dtype():
    batch_size = 2
    src_data = [
        [np.ones((120, 120, 3), dtype=np.float32)] * batch_size,
        [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext = fn.external_source(source=src_data)
    src_pipe.set_outputs(src_ext)
    src_pipe.run()
    src_pipe.run()


def test_ndim_arg():
    batch_size = 2
    src_data = [[np.ones((120, 120, 3), dtype=np.uint8)] * batch_size]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext1 = fn.external_source(source=src_data, dtype=DALIDataType.UINT8, ndim=3)
    src_ext2 = fn.external_source(source=src_data, dtype=DALIDataType.UINT8, layout="HWC")
    src_pipe.set_outputs(src_ext1, src_ext2)
    out1, out2 = src_pipe.run()
    for i in range(batch_size):
        t1 = out1.at(i)
        t2 = out2.at(i)
        assert np.array_equal(t1, np.ones((120, 120, 3), dtype=np.uint8))
        assert np.array_equal(t2, np.ones((120, 120, 3), dtype=np.uint8))


def test_ndim_arg_multioutput():
    batch_size = 2
    src_data = [
        [
            [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
            [np.ones((120, 120), dtype=np.float32)] * batch_size,
        ]
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src1_ext, src1_ext2 = fn.external_source(
        source=src_data, num_outputs=2, dtype=[DALIDataType.UINT8, DALIDataType.FLOAT], ndim=[3, 2]
    )

    src2_ext, src2_ext2 = fn.external_source(
        source=src_data,
        num_outputs=2,
        dtype=[DALIDataType.UINT8, DALIDataType.FLOAT],
        layout=["HWC", "HW"],
    )

    src_pipe.set_outputs(src1_ext, src1_ext2, src2_ext, src2_ext2)
    out11, out12, out21, out22 = src_pipe.run()
    for i in range(batch_size):
        t1 = out11.at(i)
        t2 = out12.at(i)
        assert np.array_equal(t1, np.ones((120, 120, 3), dtype=np.uint8))
        assert np.allclose(t2, [np.ones((120, 120), dtype=np.float32)])
        t3 = out21.at(i)
        t4 = out22.at(i)
        assert np.array_equal(t3, np.ones((120, 120, 3), dtype=np.uint8))
        assert np.allclose(t4, [np.ones((120, 120), dtype=np.float32)])


def test_layout_ndim_match():
    batch_size = 2
    src_data = [
        [
            [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
            [np.ones((120, 120), dtype=np.uint8)] * batch_size,
        ]
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext1, src_ext2 = fn.external_source(
        source=src_data, num_outputs=2, dtype=DALIDataType.UINT8, layout=["HWC", "HW"], ndim=[3, 2]
    )
    src_pipe.set_outputs(src_ext1, src_ext2)
    out1, out2 = src_pipe.run()
    for i in range(batch_size):
        t1 = out1.at(i)
        t2 = out2.at(i)
        assert np.array_equal(t1, np.ones((120, 120, 3), dtype=np.uint8))
        assert np.allclose(t2, [np.ones((120, 120), dtype=np.uint8)])


@raises(
    RuntimeError,
    glob="Number of dimensions in the provided layout does not match the ndim "
    'argument. The arguments provided:\n ndim = 2,\n layout: "HWC".',
)
def test_ndim_layout_mismatch():
    src_pipe = Pipeline(1, 1, 0)
    src_ext = fn.external_source(layout="HWC", ndim=2)
    src_pipe.set_outputs(src_ext)
    src_pipe.run()


@raises(RuntimeError, glob="ExternalSource expected data with 3 dimensions and got 2 dimensions")
def test_ndim_data_mismatch():
    batch_size = 2
    src_data = [
        [
            [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
            [np.ones((120, 120), dtype=np.uint8)] * batch_size,
        ]
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext1, src_ext2 = fn.external_source(
        source=src_data, num_outputs=2, dtype=DALIDataType.UINT8, ndim=3
    )
    src_pipe.set_outputs(src_ext1, src_ext2)
    src_pipe.run()


@raises(
    RuntimeError,
    glob="Number of dimensions of the data fed to the external source has "
    "changed from previous iteration. Dimensionality in the previous "
    "iteration was 3 and the current is 2.",
)
def test_ndim_changing():
    batch_size = 2
    src_data = [
        [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
        [np.ones((120, 120), dtype=np.uint8)] * batch_size,
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext1 = fn.external_source(source=src_data, dtype=DALIDataType.UINT8)
    src_pipe.set_outputs(src_ext1)
    src_pipe.run()
    src_pipe.run()


@raises(RuntimeError, glob='Expected data with layout: "H" and got: "W"')
def test_layout_data_mismatch():
    src_pipe = Pipeline(1, 1, 0, prefetch_queue_depth=1)
    src_pipe.set_outputs(fn.external_source(name="input", layout="H"))
    src_pipe.feed_input("input", [np.zeros((1))], layout="W")
    src_pipe.run()


@raises(
    RuntimeError,
    glob="Layout of the data fed to the external source has changed from "
    'previous iteration. Layout in the previous iteration was "W" '
    'and the current is "H".',
)
def test_layout_changing():
    src_pipe = Pipeline(1, 1, 0)
    src_pipe.set_outputs(fn.external_source(name="input"))
    src_pipe.feed_input("input", [np.zeros((1))], layout="W")
    src_pipe.feed_input("input", [np.zeros((1))], layout="H")
    src_pipe.run()
    src_pipe.run()


def test_layout_set_as_arg():
    src_pipe = Pipeline(1, 1, 0, prefetch_queue_depth=1)
    src_pipe.set_outputs(fn.external_source(name="input", layout="H"))
    src_pipe.feed_input("input", [np.zeros((1))])
    (out,) = src_pipe.run()
    assert out.layout() == "H"


def _test_partially_utilized_external_source_warning(usage_mask, source_type):
    np_rng = np.random.default_rng(12345)
    max_batch_size = 8
    num_outputs = len(usage_mask)

    def rand_batch():
        batch_size = np.int32(np_rng.uniform(1, max_batch_size + 1))
        return np.float32(np_rng.uniform(-1, 1, shape=(batch_size, 10, 100, 3)))

    def rand_tuple(rand_elem):
        return tuple(rand_elem() for _ in range(num_outputs))

    def sample_cb_source(sample_info):
        def rand_sample():
            return np.float32(np_rng.uniform(-1, 1, shape=(10, 100, 3)))

        return rand_tuple(rand_sample)

    def batch_cb_source():
        return rand_tuple(rand_batch)

    def gen_fun_source():
        while True:
            yield rand_tuple(rand_batch)

    class IteratorSource:
        def __iter__(self):
            return self

        def __next__(self):
            return rand_tuple(rand_batch)

    sources = {
        "sample_cb_source": sample_cb_source,
        "batch_cb_source": batch_cb_source,
        "gen_fun_source": gen_fun_source,
        "generator": gen_fun_source(),
        "IteratorSource": IteratorSource(),
    }

    @pipeline_def
    def pipeline():
        outputs = fn.external_source(
            source=sources[source_type],
            num_outputs=num_outputs,
            batch=source_type != "sample_cb_source",
        )
        assert len(outputs) == num_outputs
        utilized_outputs = (out for out, is_used in zip(outputs, usage_mask) if is_used)
        return tuple(fn.gaussian_blur(out, window_size=3) for out in utilized_outputs)

    pipe = pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
    unused_output_idxs = [i for i, is_used in enumerate(usage_mask) if not is_used]
    assert len(unused_output_idxs) > 0
    pruned_idx_str = ", ".join(str(idx) for idx in unused_output_idxs)
    if len(unused_output_idxs) == 1:
        pruned_str = f"output at the index {pruned_idx_str} is"
    else:
        pruned_str = f"outputs at the indices {pruned_idx_str} are"
    expected_error_msg = (
        f"The external source node '*{source_type}*' produces {num_outputs} outputs, "
        f"but the {pruned_str} not used."
    )
    with assert_warns(Warning, glob=expected_error_msg):
        pipe.build()


def test_partially_utilized_external_source_warning():
    rng = random.Random(42)

    def sources():
        while True:
            for source in (
                "sample_cb_source",
                "batch_cb_source",
                "gen_fun_source",
                "generator",
                "IteratorSource",
            ):
                yield source

    source_type = sources()

    for num_outputs in (2, 3, 4):
        for num_unused in range(1, num_outputs):
            unused = rng.sample(list(range(num_outputs)), num_unused)
            usage_mask = [i not in unused for i in range(num_outputs)]
            yield _test_partially_utilized_external_source_warning, usage_mask, next(source_type)


def _test_partially_utilized_es_old_style(usage_mask):
    # check that the build time error on unused external source does not interfere
    # with external sources that are manually fed by user provided code

    num_outputs = len(usage_mask)
    batch_size = 16
    batch = np.array(list(range(batch_size * 1024))).reshape(batch_size, 16, 16, 4)

    class OldStylePipe(Pipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.inp = ops.ExternalSource(num_outputs=num_outputs)
            self.gb = ops.GaussianBlur(window_size=3)

        def define_graph(self):
            self.all_inputs = self.inp()
            assert len(self.all_inputs) == num_outputs
            self.utilized_inputs = [
                inp for inp, is_used in zip(self.all_inputs, usage_mask) if is_used
            ]
            return tuple(self.gb(inp) for inp in self.utilized_inputs)

        def iter_setup(self):
            assert len(self.utilized_inputs) == sum(usage_mask)
            for out in self.utilized_inputs:
                self.feed_input(out, batch)

    pipe = OldStylePipe(batch_size=batch_size, num_threads=4, device_id=0)
    pipe.run()


def test_partially_utilized_es_old_style():
    rng = random.Random(42)
    for num_outputs in (2, 3, 4):
        for num_unused in range(1, num_outputs):
            unused = rng.sample(list(range(num_outputs)), num_unused)
            usage_mask = [i not in unused for i in range(num_outputs)]
            yield _test_partially_utilized_es_old_style, usage_mask


def _test_non_utilized_external_source_pruning(num_outputs):
    max_batch_size = 16

    def sample_cb_source(sample_info):
        return None

    @pipeline_def
    def pipeline():
        outputs = fn.external_source(  # noqa F841
            source=sample_cb_source, batch=False, num_outputs=num_outputs
        )
        data = fn.random.uniform(range=(0, 255), shape=(300, 100, 3))
        img = fn.reshape(data, layout="HWC")
        return fn.gaussian_blur(img, window_size=3)

    pipe = pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
    pipe.run()


def test_non_utilized_external_source_pruning():
    # if all outputs are unused, ES should simply be pruned not preventing pipeline from operation
    for num_outputs in (None, 1, 2, 3, 4):
        yield _test_non_utilized_external_source_pruning, num_outputs


def __test_empty_es():
    max_batch_size = 16

    @pipeline_def
    def pipeline():
        return fn.external_source(source=lambda: [])

    # Providing an empty batch was legal, but it failed on MakeContiguous node.
    # This checks proper validation in External Source which is the only way that could provide
    # empty batch as input into DALI graph.
    with assert_raises(RuntimeError, glob="*ExternalSource expects non-empty batches*"):
        pipe = pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        pipe.run()


def to_tensor_list_gpu(data):
    @pipeline_def(batch_size=len(data), num_threads=4, device_id=0, prefetch_queue_depth=1)
    def convert_pipe():
        return fn.external_source(source=[data], device="gpu")

    pipe = convert_pipe()
    (out,) = pipe.run()
    return out


def test_repeat_last():
    @pipeline_def
    def pipeline():
        cpu = fn.external_source(name="es_cpu", repeat_last=True)
        gpu = fn.external_source(name="es_gpu", repeat_last=True, device="gpu", no_copy=True)
        return cpu, gpu

    pipe = pipeline(batch_size=4, num_threads=4, device_id=0, prefetch_queue_depth=1)
    data1 = [
        np.array([1], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([42], dtype=np.int32),
        np.array([666], dtype=np.int32),
    ]
    data2 = [
        np.array([11], dtype=np.int32),
        np.array([33], dtype=np.int32),
        np.array([422], dtype=np.int32),
        np.array([6666], dtype=np.int32),
    ]
    data1_gpu = to_tensor_list_gpu(data1)
    data2_gpu = to_tensor_list_gpu(data2)
    pipe.feed_input("es_cpu", data1)
    pipe.feed_input("es_gpu", data1_gpu)
    a, b = pipe.run()
    check_batch(a, data1)
    check_batch(b, data1)
    a, b = pipe.run()
    check_batch(a, data1)
    check_batch(b, data1)

    pipe.feed_input("es_cpu", data2)
    a, b = pipe.run()
    check_batch(a, data2)
    check_batch(b, data1)

    pipe.feed_input("es_gpu", data2_gpu)
    a, b = pipe.run()
    check_batch(a, data2)
    check_batch(b, data2)

    pipe.feed_input("es_cpu", data1)
    a, b = pipe.run()
    check_batch(a, data1)
    check_batch(b, data2)


def test_repeat_last_queue():
    @pipeline_def
    def pipeline():
        cpu = fn.external_source(name="es_cpu", repeat_last=True)
        gpu = fn.external_source(name="es_gpu", repeat_last=True, device="gpu")
        return cpu, gpu

    pipe = pipeline(batch_size=4, num_threads=4, device_id=0, prefetch_queue_depth=2)
    data1 = [
        np.array([1], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([42], dtype=np.int32),
        np.array([666], dtype=np.int32),
    ]
    data2 = [
        np.array([11], dtype=np.int32),
        np.array([33], dtype=np.int32),
        np.array([422], dtype=np.int32),
        np.array([6666], dtype=np.int32),
    ]
    data3 = data1

    pipe.feed_input("es_cpu", data1)
    pipe.feed_input("es_gpu", data1)
    a, b = pipe.run()
    check_batch(a, data1)
    check_batch(b, data1)
    a, b = pipe.run()
    check_batch(a, data1)
    check_batch(b, data1)

    pipe.feed_input("es_cpu", data2)
    a, b = pipe.run()
    check_batch(a, data1)  # <- still the old value
    check_batch(b, data1)

    pipe.feed_input("es_gpu", data3)
    a, b = pipe.run()
    check_batch(a, data2)  # <- new value visible
    check_batch(b, data1)  # <- still old

    pipe.feed_input("es_cpu", data3)
    a, b = pipe.run()
    check_batch(a, data2)  # <- still 2, the most recent change not visible
    check_batch(b, data3)  # <- new


def _check_repeat_last_var_batch(device):
    @pipeline_def
    def pipeline():
        es = fn.external_source(name="es", repeat_last=True, device=device)
        u = fn.random.uniform(range=(0, 0.01))
        return es, u

    pipe = pipeline(batch_size=4, num_threads=4, device_id=0, prefetch_queue_depth=2)
    data1 = [
        np.array([1], dtype=np.int32),
        np.array([3], dtype=np.int32),
        np.array([42], dtype=np.int32),
        np.array([666], dtype=np.int32),
    ]
    data2 = [
        np.array([11], dtype=np.int32),
        np.array([33], dtype=np.int32),
        np.array([422], dtype=np.int32),
    ]
    pipe.feed_input("es", data1)

    a, b = pipe.run()
    check_batch(a, data1)
    assert len(b) == len(data1)

    a, b = pipe.run()
    check_batch(a, data1)
    assert len(b) == len(data1)

    pipe.feed_input("es", data2)
    a, b = pipe.run()
    check_batch(a, data1)  # <- still the old value
    assert len(b) == len(data1)

    a, b = pipe.run()  # <- new value visible
    check_batch(a, data2)
    assert len(b) == len(data2)

    pipe.feed_input("es", data1)
    a, b = pipe.run()
    check_batch(a, data2)  # <- still 2, the most recent change not visible
    assert len(b) == len(data2)

    a, b = pipe.run()  # <- new value visible
    check_batch(a, data1)
    assert len(b) == len(data1)


def test_repeat_last_var_batch():
    for device in ["cpu", "gpu"]:
        yield _check_repeat_last_var_batch, device


def _check_blocking(device):
    batch_size = 5
    prefetch_queue_depth = 10

    @pipeline_def
    def test_pipeline():
        data = fn.external_source(
            dtype=types.INT32, name="test_source", blocking=True, device=device
        )
        return data

    rng = default_rng()
    data_to_feed = rng.random(size=(batch_size, 4, 6, 2)).astype(dtype=np.int32)

    pipe = test_pipeline(
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
        seed=12,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    pipe.feed_input("test_source", data_to_feed)

    for _ in range(5):
        out = pipe.run()[0].as_tensor()
        if device == "gpu":
            out = out.as_cpu()
        assert np.all(np.equal(np.array(out), data_to_feed))
        data_to_feed = rng.random(size=(batch_size, 4, 6, 2)).astype(dtype=np.int32)
        pipe.feed_input("test_source", data_to_feed)

    # make sure that the pipeline is not waiting for data preventing it from being deleted
    for _ in range(prefetch_queue_depth):
        pipe.feed_input("test_source", data_to_feed)


def test_blocking():
    for device in ["cpu", "gpu"]:
        yield _check_blocking, device


def _blocking_destructor(device):
    batch_size = 5
    prefetch_queue_depth = 5

    @pipeline_def
    def test_pipeline():
        data = fn.external_source(
            dtype=types.INT32, name="test_source", blocking=True, device=device
        )
        return data

    rng = default_rng()
    data_to_feed = rng.random(size=(batch_size, 4, 6, 2)).astype(dtype=np.int32)

    pipe = test_pipeline(
        batch_size=batch_size,
        num_threads=2,
        device_id=0,
        seed=12,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    # feed one input to pipeline can return something
    pipe.feed_input("test_source", data_to_feed)

    # should not hang
    _ = pipe.run()


def test_blocking_destructor():
    for device in ["cpu", "gpu"]:
        yield _blocking_destructor, device


def test_decorated_external_source():
    def code_smashing_decorator(func=None):
        """Decorator that hides the original __code__.co_argcount"""
        if func is None:
            return code_smashing_decorator

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(
                f"Now `wrapper` signature: {inspect.signature(wrapper)} looks like "
                f"`func` signature: {inspect.signature(func)}, "
                f"but the __code__.co_argcount is different: {wrapper.__code__.co_argcount} "
                f"vs {func.__code__.co_argcount}."
            )
            return func(*args, **kwargs)

        return wrapper

    @code_smashing_decorator
    def my_source(sample_info):
        return np.array([sample_info.idx_in_epoch])

    class SourceClass:
        def __init__(self, offset):
            self.offset = offset

        @code_smashing_decorator
        def __call__(self, sample_info):
            return np.array([sample_info.idx_in_epoch + self.offset])

    class SourceClassWithoutInfo:
        def __init__(self, offset):
            self.offset = offset

        @code_smashing_decorator
        def __call__(self):
            return np.array([self.offset])

    @pipeline_def(batch_size=4, device_id=0, num_threads=4)
    def test_pipe():
        src_0 = fn.external_source(source=my_source, batch=False)
        src_1 = fn.external_source(source=SourceClass(2), batch=False)
        src_2 = fn.external_source(source=SourceClassWithoutInfo(42), batch=False)
        return src_0, src_1, src_2

    pipe = test_pipe()
    (out0, out1, out2) = pipe.run()
    np.array_equal(np.array(out0.as_tensor()), np.array([0, 1, 2, 3]))
    np.array_equal(np.array(out1.as_tensor()), np.array([2, 3, 4, 5]))
    np.array_equal(np.array(out2.as_tensor()), np.array([42, 42, 42, 42]))


@raises(TypeError, glob="Found var-positional argument `*args` which is not allowed")
def test_external_source_with_disallowed_var_args():
    def my_source(*args):
        return np.array([args[0].idx_in_epoch])

    @pipeline_def(batch_size=4, device_id=0, num_threads=4)
    def test_pipe():
        return fn.external_source(source=my_source, batch=False)

    pipe = test_pipe()
    pipe.build()


@raises(TypeError, glob="Found var-positional argument `*args` which is not allowed")
def test_external_source_with_disallowed_arg_and_var_args():
    def my_source(arg, *args):
        return np.array([arg.idx_in_epoch])

    @pipeline_def(batch_size=4, device_id=0, num_threads=4)
    def test_pipe():
        return fn.external_source(source=my_source, batch=False)

    pipe = test_pipe()
    pipe.build()


@raises(TypeError, glob="Found var-keyword argument `**kwargs` which is not allowed")
def test_external_source_with_disallowed_var_kwargs():
    def my_source(**kwargs):
        return np.array([kwargs["sample_info"].idx_in_epoch])

    @pipeline_def(batch_size=4, device_id=0, num_threads=4)
    def test_pipe():
        return fn.external_source(source=my_source, batch=False)

    pipe = test_pipe()
    pipe.build()


@raises(TypeError, glob="Found var-keyword argument `**kwargs` which is not allowed")
def test_external_source_with_disallowed_arg_and_var_kwargs():
    def my_source(arg, **kwargs):
        return np.array([arg.idx_in_epoch])

    @pipeline_def(batch_size=4, device_id=0, num_threads=4)
    def test_pipe():
        return fn.external_source(source=my_source, batch=False)

    pipe = test_pipe()
    pipe.build()


@raises(TypeError, glob="Found keyword-only argument `kw_only` which is not allowed.")
def test_external_source_with_disallowed_kwarg_only():
    def my_source(*, kw_only=10):
        return np.array([kw_only])

    @pipeline_def(batch_size=4, device_id=0, num_threads=4)
    def test_pipe():
        return fn.external_source(source=my_source, batch=False)

    pipe = test_pipe()
    pipe.build()


@raises(TypeError, glob="Found more than one positional argument, which is not allowed.")
def test_external_source_with_disallowed_too_many():
    def my_source(arg, a, b, /):
        return np.array([arg.idx_in_epoch])

    @pipeline_def(batch_size=4, device_id=0, num_threads=4)
    def test_pipe():
        return fn.external_source(source=my_source, batch=False)

    pipe = test_pipe()
    pipe.build()


def test_accepted_arg_count():
    from nvidia.dali._utils.external_source_impl import accepted_arg_count

    def fun_zero():
        pass

    def fun_one(a):
        pass

    class MethodZero:
        def __call__(self):
            pass

    class MethodOne:
        def __call__(self, a):
            pass

    assert accepted_arg_count(fun_zero) == 0
    assert accepted_arg_count(fun_one) == 1
    assert accepted_arg_count(MethodZero()) == 0
    assert accepted_arg_count(MethodOne()) == 1
