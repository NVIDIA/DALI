# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
from nvidia.dali.experimental.dynamic.compile._invariant import (
    is_invariant,
    unwrap_invariant,
    unwrap_invariants,
)


@params((None,), (1,), ("text",), ([1],), (int,))
def test_invariant_marker(value):
    marked = ndd.compile.invariant(value)
    assert marked == value
    assert is_invariant(marked)
    assert isinstance(marked, type(value))
    assert unwrap_invariant(marked) is value


def test_invariant_attributes():
    class Value:
        def __init__(self):
            self.item = []

        def get_item(self):
            return self.item

    value = Value()
    marked = ndd.compile.invariant(value)
    assert is_invariant(marked.item)
    assert is_invariant(marked.get_item())
    assert not is_invariant(value.item)

    replacement = ndd.compile.invariant([1])
    marked.item = replacement
    assert value.item is replacement
    del marked.item
    assert not hasattr(value, "item")


def test_invariant_magic_methods():
    assert ndd.compile.invariant(2) + ndd.compile.invariant(3) == 5
    assert 7 - ndd.compile.invariant(2) == 5
    assert ndd.compile.invariant([1, 2])[ndd.compile.invariant(0)] == 1
    assert list(ndd.compile.invariant([1, 2])) == [1, 2]

    value = ndd.compile.invariant(object())
    assert ndd.compile.invariant(lambda x: x)(value) is value
    np.testing.assert_array_equal(np.asarray(ndd.compile.invariant(np.asarray([1, 2]))), [1, 2])


def test_invariant_unwrap():
    marked = ndd.compile.invariant(1)
    original = [marked, (marked,)]
    assert unwrap_invariants(original) == [1, (1,)]
    assert original[0] is marked

    unchanged = [1, (2,)]
    assert unwrap_invariants(unchanged) is unchanged
    assert unwrap_invariants({marked: marked}) == {1: 1}

    class Iterable:
        def __iter__(self):
            raise AssertionError("unwrap must not iterate user values")

    value = Iterable()
    assert unwrap_invariants(ndd.compile.invariant(value)) is value


def test_invariant_api_inputs():
    data = ndd.compile.invariant([np.asarray([1, 2]), np.asarray([3, 4])])
    batch = ndd.as_batch(data)
    dense = ndd.as_tensor(batch, pad=ndd.compile.invariant(False))
    np.testing.assert_array_equal(dense, [[1, 2], [3, 4]])

    images = ndd.as_batch([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)], layout="HWC")
    output = ndd.rotate(
        images,
        batch_size=ndd.compile.invariant(2),
        device=ndd.compile.invariant("cpu"),
        angle=ndd.compile.invariant(0.0),
    )
    np.testing.assert_array_equal(ndd.as_tensor(output, pad=True), ndd.as_tensor(images, pad=True))

    source = ndd.ExternalSource(
        ndd.compile.invariant(iter([np.asarray([1, 2, 3], dtype=np.int32)])),
        device=ndd.compile.invariant("cpu"),
    )
    np.testing.assert_array_equal(source(), [1, 2, 3])
