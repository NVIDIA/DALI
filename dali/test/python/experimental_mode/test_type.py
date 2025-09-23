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
import nvidia.dali.types as types


def test_type_ids():
    assert dali2.int8.type_id == types.INT8
    assert dali2.uint8.type_id == types.UINT8
    assert dali2.int16.type_id == types.INT16
    assert dali2.uint16.type_id == types.UINT16
    assert dali2.int32.type_id == types.INT32
    assert dali2.uint32.type_id == types.UINT32
    assert dali2.int64.type_id == types.INT64
    assert dali2.uint64.type_id == types.UINT64
    assert dali2.float16.type_id == types.FLOAT16
    assert dali2.float32.type_id == types.FLOAT
    assert dali2.float64.type_id == types.FLOAT64
    assert dali2.bool.type_id == types.BOOL
    assert dali2.InterpType.type_id == types.INTERP_TYPE
    assert dali2.DataType.type_id == types.DATA_TYPE
    assert dali2.ImageType.type_id == types.IMAGE_TYPE


def test_type_ids_to_type():
    assert dali2.dtype(types.INT8) == dali2.int8
    assert dali2.dtype(types.UINT8) == dali2.uint8
    assert dali2.dtype(types.INT16) == dali2.int16
    assert dali2.dtype(types.UINT16) == dali2.uint16
    assert dali2.dtype(types.INT32) == dali2.int32
    assert dali2.dtype(types.UINT32) == dali2.uint32
    assert dali2.dtype(types.INT64) == dali2.int64
    assert dali2.dtype(types.UINT64) == dali2.uint64
    assert dali2.dtype(types.FLOAT16) == dali2.float16
    assert dali2.dtype(types.FLOAT) == dali2.float32
    assert dali2.dtype(types.FLOAT64) == dali2.float64
    assert dali2.dtype(types.BOOL) == dali2.bool
    assert dali2.dtype(types.INTERP_TYPE) == dali2.InterpType
    assert dali2.dtype(types.DATA_TYPE) == dali2.DataType
    assert dali2.dtype(types.IMAGE_TYPE) == dali2.ImageType


def test_type_names():
    assert dali2.int8.name == "i8"
    assert dali2.uint8.name == "u8"
    assert dali2.int16.name == "i16"
    assert dali2.uint16.name == "u16"
    assert dali2.int32.name == "i32"
    assert dali2.uint32.name == "u32"
    assert dali2.int64.name == "i64"
    assert dali2.uint64.name == "u64"
    assert dali2.float16.name == "f16"
    assert dali2.float32.name == "f32"
    assert dali2.float64.name == "f64"
    assert dali2.bool.name == "bool"
    assert dali2.bfloat16.name == "bfloat16"
    assert dali2.DataType.name == "DataType"
    assert dali2.ImageType.name == "ImageType"
    assert dali2.InterpType.name == "InterpType"


def test_type_names_to_type():
    assert dali2.dtype("i8") is dali2.int8
    assert dali2.dtype("u8") is dali2.uint8
    assert dali2.dtype("i16") is dali2.int16
    assert dali2.dtype("u16") is dali2.uint16
    assert dali2.dtype("i32") is dali2.int32
    assert dali2.dtype("u32") is dali2.uint32
    assert dali2.dtype("i64") is dali2.int64
    assert dali2.dtype("u64") is dali2.uint64
    assert dali2.dtype("f16") is dali2.float16
    assert dali2.dtype("f32") is dali2.float32
    assert dali2.dtype("f64") is dali2.float64
    assert dali2.dtype("bool") is dali2.bool
    assert dali2.dtype("bfloat16") is dali2.bfloat16
    assert dali2.dtype("DataType") is dali2.DataType
    assert dali2.dtype("ImageType") is dali2.ImageType
    assert dali2.dtype("InterpType") is dali2.InterpType


def test_type_bytes():
    assert dali2.int8.bytes == 1
    assert dali2.uint8.bytes == 1
    assert dali2.int16.bytes == 2
    assert dali2.uint16.bytes == 2
    assert dali2.int32.bytes == 4
    assert dali2.uint32.bytes == 4
    assert dali2.int64.bytes == 8
    assert dali2.uint64.bytes == 8
    assert dali2.float16.bytes == 2
    assert dali2.float32.bytes == 4
    assert dali2.float64.bytes == 8
    assert dali2.bool.bytes == 1
    assert dali2.InterpType.bytes == 4
    assert dali2.DataType.bytes == 4
    assert dali2.ImageType.bytes == 4


def test_type_exponent_bits():
    assert dali2.int8.exponent_bits is None
    assert dali2.uint8.exponent_bits is None
    assert dali2.int16.exponent_bits is None
    assert dali2.uint16.exponent_bits is None
    assert dali2.int32.exponent_bits is None
    assert dali2.uint32.exponent_bits is None
    assert dali2.int64.exponent_bits is None
    assert dali2.uint64.exponent_bits is None
    assert dali2.float16.exponent_bits == 5
    assert dali2.float32.exponent_bits == 8
    assert dali2.float64.exponent_bits == 11


def test_type_significand_bits():
    assert dali2.int8.significand_bits is None
    assert dali2.uint8.significand_bits is None
    assert dali2.int16.significand_bits is None
    assert dali2.uint16.significand_bits is None
    assert dali2.int32.significand_bits is None
    assert dali2.uint32.significand_bits is None
    assert dali2.int64.significand_bits is None
    assert dali2.uint64.significand_bits is None
    assert dali2.float16.significand_bits == 10
    assert dali2.float32.significand_bits == 23
    assert dali2.float64.significand_bits == 52
