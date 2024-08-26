# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.ops as ops
from nvidia.dali.types import Constant
from nose_utils import assert_equals, assert_raises


def test_group_inputs():
    e0 = ops._DataNode("op0", "cpu")
    e1 = ops._DataNode("op1", "cpu")
    inputs = [e0, e1, 10.0, Constant(0).uint8(), 42]
    cat_idx, edges, integers, reals = ops._group_inputs(inputs)
    assert_equals([("edge", 0), ("edge", 1), ("real", 0), ("integer", 0), ("integer", 1)], cat_idx)
    assert_equals([e0, e1], edges)
    assert_equals([Constant(0).uint8(), 42], integers)
    assert_equals([10.0], reals)
    assert_raises(
        TypeError,
        ops._group_inputs,
        [complex()],
        glob="Expected scalar value of type 'bool', 'int' or 'float', got *.",
    )
    _, _, _, none_reals = ops._group_inputs([e0, 10])
    assert_equals(None, none_reals)


def test_generate_input_desc():
    desc0 = ops._generate_input_desc([("edge", 0)], [], [])
    desc1 = ops._generate_input_desc([("edge", 0), ("edge", 1), ("edge", 2)], [], [])
    assert_equals("&0", desc0)
    assert_equals("&0 &1 &2", desc1)

    desc2 = ops._generate_input_desc(
        [("integer", 1), ("integer", 0), ("edge", 0)], [Constant(42).uint8(), 42], []
    )
    assert_equals("$1:int32 $0:uint8 &0", desc2)

    c = Constant(42)
    desc3 = ops._generate_input_desc(
        [
            ("integer", 0),
            ("integer", 1),
            ("integer", 2),
            ("integer", 3),
            ("integer", 4),
            ("integer", 5),
            ("integer", 6),
            ("integer", 7),
            ("integer", 8),
        ],
        [
            int(),
            c.uint8(),
            c.uint16(),
            c.uint32(),
            c.uint64(),
            c.int8(),
            c.int16(),
            c.int32(),
            c.int64(),
        ],
        [],
    )
    assert_equals(
        "$0:int32 $1:uint8 $2:uint16 $3:uint32 $4:uint64 $5:int8 $6:int16 $7:int32 $8:int64", desc3
    )

    desc4 = ops._generate_input_desc(
        [("real", 0), ("real", 1), ("real", 2), ("real", 3)],
        [],
        [float(), c.float16(), c.float32(), c.float64()],
    )
    assert_equals("$0:float32 $1:float16 $2:float32 $3:float64", desc4)
