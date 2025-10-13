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

import nvidia.dali.experimental.dynamic as D
import nvidia.dali.types as types


def test_type_ids():
    assert D.int8.type_id == types.INT8
    assert D.uint8.type_id == types.UINT8
    assert D.int16.type_id == types.INT16
    assert D.uint16.type_id == types.UINT16
    assert D.int32.type_id == types.INT32
    assert D.uint32.type_id == types.UINT32
    assert D.int64.type_id == types.INT64
    assert D.uint64.type_id == types.UINT64
    assert D.float16.type_id == types.FLOAT16
    assert D.float32.type_id == types.FLOAT
    assert D.float64.type_id == types.FLOAT64
    assert D.bool.type_id == types.BOOL
    assert D.InterpType.type_id == types.INTERP_TYPE
    assert D.DataType.type_id == types.DATA_TYPE
    assert D.ImageType.type_id == types.IMAGE_TYPE


def test_type_ids_to_type():
    assert D.dtype(types.INT8) == D.int8
    assert D.dtype(types.UINT8) == D.uint8
    assert D.dtype(types.INT16) == D.int16
    assert D.dtype(types.UINT16) == D.uint16
    assert D.dtype(types.INT32) == D.int32
    assert D.dtype(types.UINT32) == D.uint32
    assert D.dtype(types.INT64) == D.int64
    assert D.dtype(types.UINT64) == D.uint64
    assert D.dtype(types.FLOAT16) == D.float16
    assert D.dtype(types.FLOAT) == D.float32
    assert D.dtype(types.FLOAT64) == D.float64
    assert D.dtype(types.BOOL) == D.bool
    assert D.dtype(types.INTERP_TYPE) == D.InterpType
    assert D.dtype(types.DATA_TYPE) == D.DataType
    assert D.dtype(types.IMAGE_TYPE) == D.ImageType


def test_type_names():
    assert D.int8.name == "i8"
    assert D.uint8.name == "u8"
    assert D.int16.name == "i16"
    assert D.uint16.name == "u16"
    assert D.int32.name == "i32"
    assert D.uint32.name == "u32"
    assert D.int64.name == "i64"
    assert D.uint64.name == "u64"
    assert D.float16.name == "f16"
    assert D.float32.name == "f32"
    assert D.float64.name == "f64"
    assert D.bool.name == "bool"
    assert D.bfloat16.name == "bfloat16"
    assert D.DataType.name == "DataType"
    assert D.ImageType.name == "ImageType"
    assert D.InterpType.name == "InterpType"


def test_type_names_to_type():
    assert D.dtype("i8") is D.int8
    assert D.dtype("u8") is D.uint8
    assert D.dtype("i16") is D.int16
    assert D.dtype("u16") is D.uint16
    assert D.dtype("i32") is D.int32
    assert D.dtype("u32") is D.uint32
    assert D.dtype("i64") is D.int64
    assert D.dtype("u64") is D.uint64
    assert D.dtype("f16") is D.float16
    assert D.dtype("f32") is D.float32
    assert D.dtype("f64") is D.float64
    assert D.dtype("bool") is D.bool
    assert D.dtype("bfloat16") is D.bfloat16
    assert D.dtype("DataType") is D.DataType
    assert D.dtype("ImageType") is D.ImageType
    assert D.dtype("InterpType") is D.InterpType


def test_type_bytes():
    assert D.int8.bytes == 1
    assert D.uint8.bytes == 1
    assert D.int16.bytes == 2
    assert D.uint16.bytes == 2
    assert D.int32.bytes == 4
    assert D.uint32.bytes == 4
    assert D.int64.bytes == 8
    assert D.uint64.bytes == 8
    assert D.float16.bytes == 2
    assert D.float32.bytes == 4
    assert D.float64.bytes == 8
    assert D.bool.bytes == 1
    assert D.InterpType.bytes == 4
    assert D.DataType.bytes == 4
    assert D.ImageType.bytes == 4


def test_type_exponent_bits():
    assert D.int8.exponent_bits is None
    assert D.uint8.exponent_bits is None
    assert D.int16.exponent_bits is None
    assert D.uint16.exponent_bits is None
    assert D.int32.exponent_bits is None
    assert D.uint32.exponent_bits is None
    assert D.int64.exponent_bits is None
    assert D.uint64.exponent_bits is None
    assert D.float16.exponent_bits == 5
    assert D.float32.exponent_bits == 8
    assert D.float64.exponent_bits == 11


def test_type_significand_bits():
    assert D.int8.significand_bits is None
    assert D.uint8.significand_bits is None
    assert D.int16.significand_bits is None
    assert D.uint16.significand_bits is None
    assert D.int32.significand_bits is None
    assert D.uint32.significand_bits is None
    assert D.int64.significand_bits is None
    assert D.uint64.significand_bits is None
    assert D.float16.significand_bits == 10
    assert D.float32.significand_bits == 23
    assert D.float64.significand_bits == 52
