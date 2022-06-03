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

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
from nvidia.dali.pipeline import Pipeline
from test_utils import check_batch
from nose_utils import raises
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
