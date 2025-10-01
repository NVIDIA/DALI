# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dali2 as dali2
import numpy as np
from nvidia.dali.experimental.dali2._invocation import Invocation


class MockTensor:
    def __init__(self, data, layout=None):
        self.shape = data.shape
        self.dtype = dali2.dtype(data.dtype)
        self.layout = layout
        self.device = dali2.Device("cpu")
        self.data = data

    @property
    def ndim(self):
        return len(self._shape)

    def __repr__(self):
        return f"MockTensor(data={self.data}, layout={self.layout})"

    def __eq__(self, other):
        return np.array_equal(self.data, other.data) and self.layout == other.layout


class MockBatch:
    def __init__(self, tensors):
        self._tensors = tensors

    @property
    def tensors(self):
        return self._tensors

    @property
    def ndim(self):
        return len(self._tensors[0].shape) if self._tensors else None

    @property
    def shape(self):
        return [t.shape for t in self._tensors]

    @property
    def dtype(self):
        return self._tensors[0].dtype if self._tensors else None

    @property
    def layout(self):
        return self._tensors[0].layout if self._tensors else None

    @property
    def num_tensors(self):
        return len(self._tensors)

    @property
    def device(self):
        return self._tensors[0].device if self._tensors else None

    def __repr__(self):
        return f"MockBatch(tensors={self._tensors})"

    def __eq__(self, other):
        return self.tensors == other.tensors


class MockOperator:
    def __init__(self, batch_size=None, addend=None):
        self._batch_size = batch_size
        self._addend = addend

    def run(self, ctx, *inputs, batch_size=None, addend=None):
        if addend is None:
            addend = self._addend
            if addend is None:
                addend = 1
        if batch_size is None:
            batch_size = self._batch_size
        if len(inputs) == 0:
            if isinstance(addend, MockBatch):
                assert batch_size is not None
                tensors = addend.tensors
            else:
                if not isinstance(addend, MockTensor):
                    addend = MockTensor(addend)
                tensors = [addend] * (batch_size or 1)
            return (tensors[0],) if batch_size is None else (MockBatch(tensors),)
        if batch_size is not None:
            return tuple(
                MockBatch(
                    [
                        MockTensor(t.data + a.data, t.layout)
                        for t, a in zip(input.tensors, addend.tensors)
                    ]
                )
                for input in inputs
            )
        else:
            return tuple(MockTensor(t.data + addend.data, t.layout) for t in inputs)

    def infer_num_outputs(self, *inputs, **args):
        return max(len(inputs), 1)

    def infer_output_devices(self, *inputs, **args):
        return [input.device for input in inputs]


def test_mock_operator_tensor():
    op = MockOperator(batch_size=None)
    b1 = MockTensor(np.int32([1, 2, 3]))
    b2 = MockTensor(np.int32([4, 5, 6]))
    b3 = MockTensor(np.int32([7, 8, 9]))
    a = MockTensor(np.int32([10, 20, 30]))
    assert op.infer_num_outputs() == 1
    assert op.infer_num_outputs(b1, addend=a) == 1
    assert op.infer_num_outputs(b1, b2, addend=a) == 2
    assert op.infer_output_devices(b1, b2, addend=a) == [b1.device, b2.device]
    assert op.infer_output_devices(b1, b2, b3, addend=a) == [b1.device, b2.device, b3.device]

    out = op.run(dali2.EvalContext().get(), addend=a)
    assert out == (MockTensor(np.int32([10, 20, 30])),)

    out = op.run(dali2.EvalContext().get(), b1, addend=a)
    assert out == (MockTensor(np.int32([11, 22, 33])),)

    out = op.run(dali2.EvalContext().get(), b1, b2, addend=a)
    assert out == (
        MockTensor(np.int32([11, 22, 33])),
        MockTensor(np.int32([14, 25, 36])),
    )

    out = op.run(dali2.EvalContext().get(), b1, b2, b3, addend=a)
    assert out == (
        MockTensor(np.int32([11, 22, 33])),
        MockTensor(np.int32([14, 25, 36])),
        MockTensor(np.int32([17, 28, 39])),
    )


def test_mock_operator_batch():
    op = MockOperator(batch_size=1)
    b1 = MockBatch([MockTensor(np.int32([1, 2, 3]))])
    b2 = MockBatch([MockTensor(np.int32([4, 5, 6]))])
    b3 = MockBatch([MockTensor(np.int32([7, 8, 9]))])
    a = MockBatch([MockTensor(np.int32([10, 20, 30]))])
    assert op.infer_num_outputs() == 1
    assert op.infer_num_outputs(b1, addend=a) == 1
    assert op.infer_num_outputs(b1, b2, addend=a) == 2
    assert op.infer_output_devices(b1, b2, addend=a) == [b1.device, b2.device]
    assert op.infer_output_devices(b1, b2, b3, addend=a) == [b1.device, b2.device, b3.device]

    out = op.run(dali2.EvalContext().get(), addend=a, batch_size=1)
    assert out == (MockBatch([MockTensor(np.int32([10, 20, 30]))]),)

    out = op.run(dali2.EvalContext().get(), b1, addend=a, batch_size=1)
    assert out == (MockBatch([MockTensor(np.int32([11, 22, 33]))]),)

    out = op.run(dali2.EvalContext().get(), b1, b2, addend=a, batch_size=1)
    assert out == (
        MockBatch([MockTensor(np.int32([11, 22, 33]))]),
        MockBatch([MockTensor(np.int32([14, 25, 36]))]),
    )

    out = op.run(dali2.EvalContext().get(), b1, b2, b3, addend=a, batch_size=1)
    assert out == (
        MockBatch([MockTensor(np.int32([11, 22, 33]))]),
        MockBatch([MockTensor(np.int32([14, 25, 36]))]),
        MockBatch([MockTensor(np.int32([17, 28, 39]))]),
    )


def test_invocation_tensor():
    op = MockOperator()
    b1 = MockTensor(np.int32([1, 2, 3]))
    b2 = MockTensor(np.int32([4, 5, 6]))
    b3 = MockTensor(np.int32([7, 8, 9]))
    a = MockTensor(np.int32([10, 20, 30]))

    inv = Invocation(op, 0, inputs=[b1, b2, b3], args={"addend": a}, is_batch=False)
    inv.run(dali2.EvalContext().get())
    assert inv.values(dali2.EvalContext().get()) == (
        MockTensor(np.int32([11, 22, 33])),
        MockTensor(np.int32([14, 25, 36])),
        MockTensor(np.int32([17, 28, 39])),
    )


def test_invocation_batch():
    op = MockOperator(batch_size=2)
    b1 = MockBatch([MockTensor(np.int32([1, 2, 3])), MockTensor(np.int32([33, 44]))])
    b2 = MockBatch([MockTensor(np.int32([4, 5, 6])), MockTensor(np.int32([55, 66]))])
    b3 = MockBatch([MockTensor(np.int32([7, 8, 9])), MockTensor(np.int32([77, 88]))])
    a = MockBatch([MockTensor(np.int32([10, 20, 30])), MockTensor(np.int32([100, 200]))])

    inv = Invocation(op, 0, inputs=[b1, b2, b3], args={"addend": a}, is_batch=True)
    inv.run(dali2.EvalContext().get())
    assert inv.values(dali2.EvalContext().get()) == (
        MockBatch([MockTensor(np.int32([11, 22, 33])), MockTensor(np.int32([133, 244]))]),
        MockBatch([MockTensor(np.int32([14, 25, 36])), MockTensor(np.int32([155, 266]))]),
        MockBatch([MockTensor(np.int32([17, 28, 39])), MockTensor(np.int32([177, 288]))]),
    )


def test_invocation_lazy_result():
    op = MockOperator(batch_size=2)
    b1 = MockBatch([MockTensor(np.int32([1, 2, 3])), MockTensor(np.int32([33, 44]))])
    b2 = MockBatch([MockTensor(np.int32([4, 5, 6])), MockTensor(np.int32([55, 66]))])
    b3 = MockBatch([MockTensor(np.int32([7, 8, 9])), MockTensor(np.int32([77, 88]))])
    a = MockBatch([MockTensor(np.int32([10, 20, 30])), MockTensor(np.int32([100, 200]))])

    inv = Invocation(op, 0, inputs=[b1, b2, b3], args={"addend": a}, is_batch=True)
    # no explicit .run call

    # automatic lazy evaluation by accessing the result
    assert inv[0].value(dali2.EvalContext().get()) == MockBatch(
        [MockTensor(np.int32([11, 22, 33])), MockTensor(np.int32([133, 244]))]
    )
    assert inv[1].value(dali2.EvalContext().get()) == MockBatch(
        [MockTensor(np.int32([14, 25, 36])), MockTensor(np.int32([155, 266]))]
    )
    assert inv[2].value(dali2.EvalContext().get()) == MockBatch(
        [MockTensor(np.int32([17, 28, 39])), MockTensor(np.int32([177, 288]))]
    )
