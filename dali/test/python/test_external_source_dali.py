# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from nvidia.dali import Pipeline, pipeline_def
from test_utils import check_batch
from nose_utils import raises, assert_warns
from nvidia.dali.types import DALIDataType


def build_src_pipe(device, layout=None):
    if layout is None:
        layout = "XY"
    batches = [[
        np.array([[1,2,3],[4,5,6]], dtype=np.float32),
        np.array([[10,20], [30,40], [50,60]], dtype=np.float32)
    ],
    [
        np.array([[9,10],[11,12]], dtype=np.float32),
        np.array([[100,200,300,400,500]], dtype=np.float32)
    ]]

    src_pipe = Pipeline(len(batches), 1, 0)
    src_pipe.set_outputs(fn.external_source(source=batches, device=device, cycle=True, layout=layout))
    src_pipe.build()
    return src_pipe, len(batches)


def _test_feed_input(device):
    src_pipe, batch_size = build_src_pipe(device)

    dst_pipe = Pipeline(batch_size, 1, 0, exec_async=False, exec_pipelined=False)
    dst_pipe.set_outputs(fn.external_source(name="ext", device=device))
    dst_pipe.build()
    for iter in range(3):
        out1 = src_pipe.run()
        dst_pipe.feed_input("ext", out1[0])
        out2 = dst_pipe.run()
        check_batch(out2[0], out1[0], batch_size, 0, 0, "XY")


def test_feed_input():
    for device in ["cpu", "gpu"]:
        yield _test_feed_input, device


def _test_callback(device, as_tensors, change_layout_to=None):
    src_pipe, batch_size = build_src_pipe(device)
    ref_pipe, batch_size = build_src_pipe(device, layout=change_layout_to)

    dst_pipe = Pipeline(batch_size, 1, 0)

    def get_from_src():
        tl = src_pipe.run()[0]
        return [tl[i] for i in range(len(tl))] if as_tensors else tl

    dst_pipe.set_outputs(fn.external_source(source=get_from_src, device=device, layout=change_layout_to))
    dst_pipe.build()

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
    src_ext = fn.external_source(source=lambda i: [np.float32(i * 10 + i + 1) for i in range(batch_size)], device=device)
    src_pipe.set_outputs(src_ext)

    src_pipe.build()
    dst_pipe = Pipeline(batch_size, 1, 0, exec_async=False, exec_pipelined=False)
    dst_pipe.set_outputs(fn.external_source(name="ext", device=device))
    dst_pipe.build()

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
            assert isinstance(arg, types.BatchInfo), "Expected BatchInfo instance as cb argument, got {}".format(arg)
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
        return np.array([
            sample_info.idx_in_epoch, sample_info.idx_in_batch,
            sample_info.iteration, sample_info.epoch_idx], dtype=np.int32)


def _test_batch_info_flag_default(cb, batch_size):
    pipe = Pipeline(batch_size, 1, 0)
    with pipe:
        ext = fn.external_source(source=cb)
        pipe.set_outputs(ext)
    pipe.build()
    pipe.run()


def test_batch_info_flag_default():
    batch_size = 5
    cb_int = BatchCb(False, batch_size, 1)
    yield _test_batch_info_flag_default, cb_int, batch_size
    cb_batch_info = BatchCb(True, batch_size, 1)
    yield raises(AssertionError, "Expected BatchInfo instance as cb argument")(_test_batch_info_flag_default), cb_batch_info, batch_size


def _test_epoch_idx(batch_size, epoch_size, cb, batch_info, batch_mode):
    num_epochs = 3
    pipe = Pipeline(batch_size, 1, 0)
    with pipe:
        ext = fn.external_source(source=cb, batch_info=batch_info, batch=batch_mode)
        pipe.set_outputs(ext)
    pipe.build()
    for epoch_idx in range(num_epochs):
        for iteration in range(epoch_size):
            (batch,) = pipe.run()
            assert len(batch) == batch_size
            for sample_i, sample in enumerate(batch):
                if batch_mode:
                    expected = np.array([iteration, epoch_idx if batch_info else -1])
                else:
                    expected = np.array([
                        iteration * batch_size + sample_i,
                        sample_i, iteration, epoch_idx])
                np.testing.assert_array_equal(sample, expected)
        try:
            pipe.run()
        except:
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
    src_data = [
        [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext = fn.external_source(source=src_data, dtype=DALIDataType.UINT8)
    src_pipe.set_outputs(src_ext)
    src_pipe.build()
    out, = src_pipe.run()
    for i in range(batch_size):
        t = out.at(i)
        assert t.dtype == np.uint8
        np.array_equal(t, np.ones((120, 120, 3), dtype=np.uint8))


def test_dtype_arg_multioutput():
    batch_size = 2
    src_data = [
        [[np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
         [np.ones((120, 120, 3), dtype=np.float32)] * batch_size]
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext, src_ext2 = fn.external_source(source=src_data, num_outputs=2,
                                           dtype=[DALIDataType.UINT8, DALIDataType.FLOAT])
    src_pipe.set_outputs(src_ext, src_ext2)
    src_pipe.build()
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
    src_data = [
        [np.ones((120, 120, 3), dtype=np.float32)] * batch_size
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext = fn.external_source(source=src_data, dtype=DALIDataType.UINT8)
    src_pipe.set_outputs(src_ext)
    src_pipe.build()
    src_pipe.run()


@raises(RuntimeError, glob="Type of the data fed to the external source has changed from the "
                           "previous iteration. Type in the previous iteration was float and "
                           "the current type is uint8.")
def test_changing_dtype():
    batch_size = 2
    src_data = [
        [np.ones((120, 120, 3), dtype=np.float32)] * batch_size,
        [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext = fn.external_source(source=src_data)
    src_pipe.set_outputs(src_ext)
    src_pipe.build()
    src_pipe.run()
    src_pipe.run()


def test_ndim_arg():
    batch_size = 2
    src_data = [
        [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext1 = fn.external_source(source=src_data, dtype=DALIDataType.UINT8, ndim=3)
    src_ext2 = fn.external_source(source=src_data, dtype=DALIDataType.UINT8, layout="HWC")
    src_pipe.set_outputs(src_ext1, src_ext2)
    src_pipe.build()
    out1, out2 = src_pipe.run()
    for i in range(batch_size):
        t1 = out1.at(i)
        t2 = out2.at(i)
        assert np.array_equal(t1, np.ones((120, 120, 3), dtype=np.uint8))
        assert np.array_equal(t2, np.ones((120, 120, 3), dtype=np.uint8))


def test_ndim_arg_multioutput():
    batch_size = 2
    src_data = [
        [[np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
         [np.ones((120, 120), dtype=np.float32)] * batch_size]
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src1_ext, src1_ext2 = fn.external_source(source=src_data, num_outputs=2,
                                           dtype=[DALIDataType.UINT8, DALIDataType.FLOAT],
                                           ndim=[3, 2])

    src2_ext, src2_ext2 = fn.external_source(source=src_data, num_outputs=2,
                                           dtype=[DALIDataType.UINT8, DALIDataType.FLOAT],
                                           layout=["HWC", "HW"])

    src_pipe.set_outputs(src1_ext, src1_ext2, src2_ext, src2_ext2)
    src_pipe.build()
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
        [[np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
         [np.ones((120, 120), dtype=np.uint8)] * batch_size]
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext1, src_ext2 = fn.external_source(source=src_data, num_outputs=2,
                                            dtype=DALIDataType.UINT8, layout=["HWC", "HW"],
                                            ndim=[3, 2])
    src_pipe.set_outputs(src_ext1, src_ext2)
    src_pipe.build()
    out1, out2 = src_pipe.run()
    for i in range(batch_size):
        t1 = out1.at(i)
        t2 = out2.at(i)
        assert np.array_equal(t1, np.ones((120, 120, 3), dtype=np.uint8))
        assert np.allclose(t2, [np.ones((120, 120), dtype=np.uint8)])


@raises(RuntimeError, glob="Number of dimensions in the provided layout does not match the ndim "
                           "argument. The arguments provided:\n ndim = 2,\n layout: \"HWC\".")
def test_ndim_layout_mismatch():
    src_pipe = Pipeline(1, 1, 0)
    src_ext = fn.external_source(layout="HWC", ndim=2)
    src_pipe.set_outputs(src_ext)
    src_pipe.build()


@raises(RuntimeError, glob="ExternalSource expected data with 3 dimensions and got 2 dimensions")
def test_ndim_data_mismatch():
    batch_size = 2
    src_data = [
        [[np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
         [np.ones((120, 120), dtype=np.uint8)] * batch_size]
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext1, src_ext2 = fn.external_source(source=src_data, num_outputs=2,
                                            dtype=DALIDataType.UINT8, ndim=3)
    src_pipe.set_outputs(src_ext1, src_ext2)
    src_pipe.build()
    src_pipe.run()


@raises(RuntimeError, glob="Number of dimensions of the data fed to the external source has changed "
                           "from previous iteration. Dimensionality in the previous iteration "
                           "was 3 and the current is 2.")
def test_ndim_changing():
    batch_size = 2
    src_data = [
        [np.ones((120, 120, 3), dtype=np.uint8)] * batch_size,
        [np.ones((120, 120), dtype=np.uint8)] * batch_size
    ]
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext1 = fn.external_source(source=src_data, dtype=DALIDataType.UINT8)
    src_pipe.set_outputs(src_ext1)
    src_pipe.build()
    src_pipe.run()
    src_pipe.run()


@raises(RuntimeError, glob="Expected data with layout: \"H\" and got: \"W\"")
def test_layout_data_mismatch():
    src_pipe = Pipeline(1, 1, 0)
    src_pipe.set_outputs(fn.external_source(name="input", layout="H"))
    src_pipe.build()
    src_pipe.feed_input("input", [np.zeros((1))], layout="W")


@raises(RuntimeError, glob="Layout of the data fed to the external source has changed from "
                            "previous iteration. Layout in the previous iteration was \"W\" "
                            "and the current is \"H\".")
def test_layout_changing():
    src_pipe = Pipeline(1, 1, 0)
    src_pipe.set_outputs(fn.external_source(name="input"))
    src_pipe.build()
    src_pipe.feed_input("input", [np.zeros((1))], layout="W")
    src_pipe.feed_input("input", [np.zeros((1))], layout="H")


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
        'sample_cb_source': sample_cb_source,
        'batch_cb_source': batch_cb_source,
        'gen_fun_source': gen_fun_source,
        'generator': gen_fun_source(),
        'IteratorSource': IteratorSource()
    }

    @pipeline_def
    def pipeline():
        outputs = fn.external_source(source=sources[source_type], num_outputs=num_outputs,
                                     batch=source_type != "sample_cb_source")
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
        f"but the {pruned_str} not used.")
    with assert_warns(Warning, glob=expected_error_msg):
        pipe.build()


def test_partially_utilized_external_source_warning():
    rng = random.Random(42)

    def sources():
        while True:
            for source in ('sample_cb_source', 'batch_cb_source', 'gen_fun_source', 'generator',
                           'IteratorSource'):
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
                inp for inp, is_used in zip(self.all_inputs, usage_mask) if is_used]
            return tuple(self.gb(inp) for inp in self.utilized_inputs)

        def iter_setup(self):
            assert len(self.utilized_inputs) == sum(usage_mask)
            for out in self.utilized_inputs:
                self.feed_input(out, batch)

    pipe = OldStylePipe(batch_size=batch_size, num_threads=4, device_id=0)
    pipe.build()
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
            source=sample_cb_source, batch=False,
            num_outputs=num_outputs)
        data = fn.random.uniform(range=(0, 255), shape=(300, 100, 3))
        img = fn.reshape(data, layout="HWC")
        return fn.gaussian_blur(img, window_size=3)

    pipe = pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
    pipe.build()
    pipe.run()


def test_non_utilized_external_source_pruning():
    # if all outputs are unused, ES should simply be pruned not preventing pipeline from operation
    for num_outputs in (None, 1, 2, 3, 4):
        yield _test_non_utilized_external_source_pruning, num_outputs
