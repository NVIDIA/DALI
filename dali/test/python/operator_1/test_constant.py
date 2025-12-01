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

import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os
from nvidia.dali import Pipeline

from test_utils import check_batch
from test_utils import get_dali_extra_path

jpeg_folder = os.path.join(get_dali_extra_path(), "db", "single", "jpeg")

array_interfaces = [(np.array, None)]

try:
    import torch

    array_interfaces.append((torch.tensor, lambda x: eval("torch." + x)))
    print("ConstantOp: PyTorch support enabled")
except ModuleNotFoundError:
    print("ConstantOp: PyTorch support disabled")
    pass

try:
    import mxnet

    array_interfaces.append((mxnet.ndarray.array, None))
    print("ConstantOp: MXNet support enabled")
except ModuleNotFoundError:
    print("ConstantOp: MXNet support disabled")
    pass


class ConstantPipeline(Pipeline):
    def __init__(self, device):
        super().__init__(10, 3, device_id=0, exec_async=True, exec_pipelined=True)
        self.const1 = ops.Constant(device=device, fdata=(1.25, 2.5, 3))
        self.const2 = ops.Constant(device=device, idata=(1, 2, 3, 4), shape=(2, 1, 2))
        self.const3 = ops.Constant(device=device, idata=(-1, 1, 2, 3, 4), dtype=types.UINT8)
        self.const4 = ops.Constant(
            device=device, fdata=(0.25, 1.25, 2.25, 3.25, 4.25), dtype=types.FLOAT16
        )
        self.const5 = ops.Constant(device=device, fdata=5.5, shape=(100, 100))
        self.const6 = ops.Constant(device=device, idata=-4, shape=(10, 20))
        self.const7 = ops.Constant(device=device, idata=[0, 1, 0], dtype=types.BOOL)

    def define_graph(self):
        return (
            self.const1(),
            self.const2(),
            self.const3(),
            self.const4(),
            self.const5(),
            self.const6(),
            self.const7(),
        )


class ConstantFnPipeline(Pipeline):
    def __init__(self, device, array_interface):
        super().__init__(10, 3, device_id=0, exec_async=True, exec_pipelined=True)
        self.device = device
        self.array = array_interface[0]
        self.dtype = array_interface[1]
        if self.dtype is None:
            self.dtype = lambda x: x

    def define_graph(self):
        device = self.device
        return [
            types.Constant(device=device, value=(1.25, 2.5, 3)),
            types.Constant(
                device=device, value=self.array([[[1, 2]], [[3, 4]]], dtype=self.dtype("int32"))
            ),
            types.Constant(
                device=device, value=self.array([0, 1, 2, 3, 4], dtype=self.dtype("uint8"))
            ),
            types.Constant(
                device=device,
                value=self.array([0.25, 1.25, 2.25, 3.25, 4.25], dtype=self.dtype("float16")),
            ),
            types.Constant(device=device, value=5.5, shape=(100, 100), name="large"),
            types.Constant(device=device, value=-4, shape=(10, 20)),
            types.Constant(device=device, value=[False, True, False]),
        ]


def check(a1, a2):
    if a1.dtype != a2.dtype:
        print(a1.dtype, a2.dtype)
    assert a1.dtype == a2.dtype
    if not np.array_equal(a1, a2):
        print("A1", a1)
        print("A2", a2)
    assert np.array_equal(a1, a2)


ref = [
    np.array([1.25, 2.5, 3], dtype=np.float32),
    np.array([[[1, 2]], [[3, 4]]], dtype=np.int32),
    np.array([0, 1, 2, 3, 4], dtype=np.uint8),
    np.array([0.25, 1.25, 2.25, 3.25, 4.25], dtype=np.float16),
    np.full([100, 100], 5.5, dtype=np.float32),
    np.full([10, 20], -4, dtype=np.int32),
    np.array([False, True, False], dtype=bool),
]


def _test_op(device):
    pipe = ConstantPipeline(device)
    for iter in range(3):
        out = pipe.run()
        if device == "gpu":
            out = [o.as_cpu() for o in out]

        for o in range(len(ref)):
            for i in range(len(out[o])):
                check(out[o].at(i), ref[o])


def _test_func(device, array_interface):
    pipe = ConstantFnPipeline(device, array_interface)
    for iter in range(3):
        out = pipe.run()
        if device == "gpu":
            out = [o.as_cpu() for o in out]

        for o in range(len(ref)):
            for i in range(len(out[o])):
                check(out[o].at(i), ref[o])


def _test_scalar_constant_promotion(device):
    @dali.pipeline_def(batch_size=1, device_id=0, num_threads=4)
    def scalar_constant_pipeline(device):
        constant = types.Constant(1)
        with_explicit_dev = types.Constant(4, device=device)
        p1 = fn.stack(constant, 2)
        if device == "gpu":
            p1 = p1.gpu()
        p2 = fn.stack(3, with_explicit_dev)
        return (fn.copy(1.25, device=device), fn.cat(p1, p2))

    pipe = scalar_constant_pipeline(device)
    ref = [np.array(1.25, dtype=np.float32), np.array([1, 2, 3, 4], dtype=np.int32)]
    for iter in range(3):
        out = pipe.run()
        if device == "gpu":
            out = [o.as_cpu() for o in out]

        for o in range(len(ref)):
            for i in range(len(out[o])):
                check(out[o].at(i), ref[o])


def test_constant_op():
    yield _test_op, "cpu"
    yield _test_op, "gpu"


def test_constant_fn():
    for device in ["cpu", "gpu"]:
        for array_interface in array_interfaces:
            yield _test_func, device, array_interface


def test_scalar_constant_promotion():
    yield _test_scalar_constant_promotion, "cpu"
    yield _test_scalar_constant_promotion, "gpu"


def test_variable_batch():
    pipe = Pipeline(6, 1, 0)
    batches = [
        [np.array(1), np.array(2)],
        [np.array(1)],
        [np.array(1), np.array(2), np.array(3), np.array(4), np.array(5), np.array(5)],
    ]
    dummy = fn.external_source(batches, cycle=True)
    val = np.float32([[1, 2], [3, 4]])
    pipe.set_outputs(types.Constant(val, device="cpu"), types.Constant(val, device="gpu"), dummy)
    for batch in batches:
        cpu, gpu, _ = pipe.run()
        assert len(cpu) == len(batch)
        assert len(gpu) == len(batch)
        gpu = gpu.as_cpu()
        for i in range(len(batch)):
            assert np.array_equal(cpu.at(i), val)
            assert np.array_equal(gpu.at(i), val)


def test_constant_promotion_mixed():
    filename = os.path.join(jpeg_folder, "241", "cute-4074304_1280.jpg")
    file_contents = np.fromfile(filename, dtype=np.uint8)
    pipe = Pipeline(1, 3, 0)
    with pipe:
        jpegs, _ = fn.readers.file(files=[filename])
        from_reader = fn.decoders.image(jpegs, device="mixed")
        from_constant = fn.decoders.image(file_contents, device="mixed")
        pipe.set_outputs(from_constant, from_reader)
    from_reader, from_constant = pipe.run()
    check_batch(from_reader, from_constant, 1)
