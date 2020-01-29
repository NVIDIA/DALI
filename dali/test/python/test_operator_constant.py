# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import os

class ConstantPipeline(Pipeline):
    def __init__(self, device):
        super(ConstantPipeline, self).__init__(10, 3, device_id = 0, exec_async=True, exec_pipelined=True)
        self.const1 = ops.Constant(device = device, fdata = (1.25,2.5,3))
        self.const2 = ops.Constant(device = device, idata = (1,2,3,4), shape=(2,1,2))
        self.const3 = ops.Constant(device = device, idata = (-1,1,2,3,4), dtype=types.UINT8)
        self.const4 = ops.Constant(device = device, fdata = 5.5, shape=(100,100))
        self.const5 = ops.Constant(device = device, idata = -4, shape=(10,20))


    def define_graph(self):
        return (self.const1(), self.const2(), self.const3(), self.const4(), self.const5())

class ConstantFnPipeline(Pipeline):
    def __init__(self, device):
        super(ConstantFnPipeline, self).__init__(10, 3, device_id = 0, exec_async=True, exec_pipelined=True)
        self.device = device

    def define_graph(self):
        device = self.device
        return [
            types.Constant(device = device, value = (1.25,2.5,3)),
            types.Constant(device = device, value = np.array([[[1,2]],[[3,4]]], dtype=np.int32)),
            types.Constant(device = device, value = np.array([0,1,2,3,4], dtype=np.uint8)),
            types.Constant(device = device, value = 5.5, shape=(100,100)),
            types.Constant(device = device, value = -4, shape=(10,20))
        ]

def check(a1, a2):
    assert(a1.dtype == a2.dtype)
    assert(np.array_equal(a1, a2))


ref = [
    np.array([1.25, 2.5, 3], dtype=np.float32),
    np.array([[[1,2]],[[3,4]]], dtype=np.int32),
    np.array([0,1,2,3,4], dtype=np.uint8),
    np.full([100, 100], 5.5, dtype=np.float32),
    np.full([10, 20], -4, dtype=np.int32)
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

def _test_func(device):
    pipe = ConstantFnPipeline(device)
    pipe.build()
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
    yield _test_func, "cpu"
    yield _test_func, "gpu"

def main():
    _test_op("cpu")
    _test_op("gpu")
    _test_func("cpu")
    _test_func("gpu")

if __name__ == '__main__':
  main()
