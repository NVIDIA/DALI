# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import nvidia.dali._multiproc.shared_batch as sb

from test_utils import RandomlyShapedDataIterator


def recursive_equals(left, right, top_level=True):
    if top_level:
        idx_l, left = left
        idx_r, right = right
        assert idx_l == idx_r, "Indexes of samples should be the same"
    if isinstance(left, tuple):
        assert isinstance(right, tuple), "Nesting should be the same"
        assert len(left) == len(right), "Nesting len should be the same"
        for i in range(len(left)):
            recursive_equals(left[i], right[i], False)
    np.testing.assert_array_equal(left, right)


def check_serialize_deserialize(indexed_batch):
    mem_chunk = sb.SharedMemChunk("chunk_0", 100)
    shared_batch_meta = sb.write_batch(mem_chunk, indexed_batch)
    deserlized_indexed_batch = sb.deserialize_batch(mem_chunk.shm_chunk, shared_batch_meta)
    assert len(indexed_batch) == len(
        deserlized_indexed_batch), "Lengths before and after should be the same"
    for i in range(len(deserlized_indexed_batch)):
        recursive_equals(indexed_batch[i], deserlized_indexed_batch[i])
    mem_chunk.close()


def test_serialize_deserialize():
    for shapes in [[(10)], [(10, 20)], [(10, 20, 3)], [(1), (2)], [(2), (2, 3)],
                   [(2, 3, 4), (2, 3, 5), (3, 4, 5)], []]:
        for dtype in [np.int8, np.float, np.int32]:
            yield check_serialize_deserialize, [(i, np.full(s, 42, dtype=dtype)) for i, s in enumerate(shapes)]


def test_serialize_deserialize_random():
    for max_shape in [(12, 200, 100, 3), (200, 300, 3), (300, 2)]:
        for dtype in [np.uint8, np.float]:
            rsdi = RandomlyShapedDataIterator(10, max_shape=max_shape, dtype=dtype)
            for i, batch in enumerate(rsdi):
                if i == 10:
                    break
                indexed_batch = [(i, sample) for i, sample in enumerate(batch)]
                yield check_serialize_deserialize, indexed_batch
