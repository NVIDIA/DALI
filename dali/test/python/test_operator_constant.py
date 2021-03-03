# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import os
from test_utils import get_dali_extra_path
from test_utils import check_batch

jpeg_folder = os.path.join(get_dali_extra_path(), 'db', 'single', 'jpeg')

array_interfaces = [(np.array, None)]
try:
    import torch
    array_interfaces.append( (torch.tensor, lambda x : eval('torch.' + x)) )
    print("ConstantOp: PyTorch support enabled");
except:
    print("ConstantOp: PyTorch support disabled");
    pass
try:
    import mxnet
    array_interfaces.append( (mxnet.ndarray.array, None) )
    print("ConstantOp: MXNet support enabled");
except:
    print("ConstantOp: MXNet support disabled");
    pass


class ConstantPipeline(Pipeline):
    def __init__(self, device):
        super(ConstantPipeline, self).__init__(10, 3, device_id = 0, exec_async=True, exec_pipelined=True)
        self.const1 = ops.Constant(device = device, fdata = (1.25,2.5,3))
        self.const2 = ops.Constant(device = device, idata = (1,2,3,4), shape=(2,1,2))
        self.const3 = ops.Constant(device = device, idata = (-1,1,2,3,4), dtype=types.UINT8)
        self.const4 = ops.Constant(device = device, fdata = (0.25,1.25,2.25,3.25,4.25), dtype=types.FLOAT16)
        self.const5 = ops.Constant(device = device, fdata = 5.5, shape=(100,100))
        self.const6 = ops.Constant(device = device, idata = -4, shape=(10,20))
        self.const7 = ops.Constant(device = device, idata = [0, 1, 0], dtype=types.BOOL)


    def define_graph(self):
        return (self.const1(), self.const2(), self.const3(), self.const4(),
                self.const5(), self.const6(), self.const7())

class ConstantFnPipeline(Pipeline):
    def __init__(self, device, array_interface):
        super(ConstantFnPipeline, self).__init__(10, 3, device_id = 0, exec_async=True, exec_pipelined=True)
        self.device = device
        self.array = array_interface[0]
        self.dtype = array_interface[1]
        if self.dtype is None:
            self.dtype = lambda x : x

    def define_graph(self):
        device = self.device
        return [
            types.Constant(device = device, value = (1.25,2.5,3)),
            types.Constant(device = device, value = self.array([[[1,2]],[[3,4]]], dtype=self.dtype('int32'))),
            types.Constant(device = device, value = self.array([0,1,2,3,4], dtype=self.dtype('uint8'))),
            types.Constant(device = device, value = self.array([0.25,1.25,2.25,3.25,4.25], dtype=self.dtype('float16'))),
            types.Constant(device = device, value = 5.5, shape=(100,100), name="large"),
            types.Constant(device = device, value = -4, shape=(10,20)),
            types.Constant(device = device, value = [False, True, False])
        ]


class ScalarConstantPipeline(Pipeline):
    def __init__(self, device):
        super(ScalarConstantPipeline, self).__init__(10, 3, device_id = 0, exec_async=True, exec_pipelined=True)
        self.device = device

    def define_graph(self):
        device = self.device
        return [
            # no-op
            ops.Reshape(device = device, shape=[1])(types.Constant(1.25)),
            # flatten with reshape op
            ops.Reshape(device = device)(types.Constant(np.array([[1,2],[3,4]],
                                                        dtype = np.uint16),
                                                        device = device),
                                         shape = types.Constant([4]))
        ]

def check(a1, a2):
    if a1.dtype != a2.dtype:
        print(a1.dtype, a2.dtype)
    assert(a1.dtype == a2.dtype)
    assert(np.array_equal(a1, a2))


ref = [
    np.array([1.25, 2.5, 3], dtype=np.float32),
    np.array([[[1,2]],[[3,4]]], dtype=np.int32),
    np.array([0,1,2,3,4], dtype=np.uint8),
    np.array([0.25,1.25,2.25,3.25,4.25], dtype=np.float16),
    np.full([100, 100], 5.5, dtype=np.float32),
    np.full([10, 20], -4, dtype=np.int32),
    np.array([False, True, False], dtype=np.bool)
]


def _test_op(device):
    pipe = ConstantPipeline(device)
    pipe.build()
    for iter in range(3):
        out = pipe.run()
        if device == "gpu":
            out = [o.as_cpu() for o in out]

        for o in range(len(ref)):
            for i in range(len(out[o])):
                check(out[o].at(i), ref[o])

def _test_func(device, array_interface):
    pipe = ConstantFnPipeline(device, array_interface)
    pipe.build()
    for iter in range(3):
        out = pipe.run()
        if device == "gpu":
            out = [o.as_cpu() for o in out]

        for o in range(len(ref)):
            for i in range(len(out[o])):
                check(out[o].at(i), ref[o])


def _test_scalar_constant_promotion(device):
    pipe = ScalarConstantPipeline(device)
    pipe.build()
    ref = [
        np.array([1.25], dtype=np.float32),
        np.array([1,2,3,4], dtype=np.uint16)
    ]
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
        [np.array(1), np.array(2), np.array(3), np.array(4), np.array(5), np.array(5)]
    ]
    dummy = fn.external_source(batches, cycle=True)
    val = np.float32([[1,2],[3,4]])
    pipe.set_outputs(types.Constant(val, device="cpu"), types.Constant(val, device="gpu"), dummy)
    pipe.build()
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
    pipe.build()
    from_reader, from_constant = pipe.run()
    check_batch(from_reader, from_constant, 1)
