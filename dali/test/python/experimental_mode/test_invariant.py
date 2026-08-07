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
from nose_utils import assert_raises
from nvidia.dali.experimental.dynamic.capture._invariant import (
    is_invariant,
    unwrap_invariant,
    unwrap_invariants,
)


@params((None,), (1,), ("text",), ([1],), (int,))
def test_invariant_marker(value):
    marked = ndd.capture.invariant(value)
    assert marked == value
    assert is_invariant(marked)
    assert isinstance(marked, type(value))
    assert unwrap_invariant(marked) is value
    assert ndd.capture.invariant(marked) is marked


def test_invariant_attributes():
    class Value:
        def __init__(self):
            self.item = []

        def get_item(self):
            return self.item

    value = Value()
    marked = ndd.capture.invariant(value)
    assert is_invariant(marked.item)
    assert is_invariant(marked.get_item())
    assert not is_invariant(value.item)

    replacement = ndd.capture.invariant([1])
    marked.item = replacement
    assert value.item is replacement
    del marked.item
    assert not hasattr(value, "item")


def test_invariant_magic_methods():
    assert ndd.capture.invariant(2) + ndd.capture.invariant(3) == 5
    assert 7 - ndd.capture.invariant(2) == 5
    assert ndd.capture.invariant([1, 2])[ndd.capture.invariant(0)] == 1
    assert list(ndd.capture.invariant([1, 2])) == [1, 2]

    value = ndd.capture.invariant(object())
    assert ndd.capture.invariant(lambda x: x)(value) is value
    np.testing.assert_array_equal(np.asarray(ndd.capture.invariant(np.asarray([1, 2]))), [1, 2])

    with assert_raises(TypeError, glob="unhashable type"):
        hash(ndd.capture.invariant([1, 2]))


def test_invariant_unwrap():
    marked = ndd.capture.invariant(1)
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
    assert unwrap_invariants(ndd.capture.invariant(value)) is value


def test_invariant_api_inputs():
    data = ndd.capture.invariant([np.asarray([1, 2]), np.asarray([3, 4])])
    batch = ndd.as_batch(data)
    dense = ndd.as_tensor(batch, pad=ndd.capture.invariant(False))
    np.testing.assert_array_equal(dense, [[1, 2], [3, 4]])

    images = ndd.as_batch([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)], layout="HWC")
    output = ndd.rotate(
        images,
        batch_size=ndd.capture.invariant(2),
        device=ndd.capture.invariant("cpu"),
        angle=ndd.capture.invariant(0.0),
    )
    np.testing.assert_array_equal(ndd.as_tensor(output, pad=True), ndd.as_tensor(images, pad=True))

    source = ndd.ExternalSource(
        ndd.capture.invariant(iter([np.asarray([1, 2, 3], dtype=np.int32)])),
        device=ndd.capture.invariant("cpu"),
    )
    np.testing.assert_array_equal(source(), [1, 2, 3])
