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

import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali.types as types


def test_type_ids():
    assert ndd.int8.type_id == types.INT8
    assert ndd.uint8.type_id == types.UINT8
    assert ndd.int16.type_id == types.INT16
    assert ndd.uint16.type_id == types.UINT16
    assert ndd.int32.type_id == types.INT32
    assert ndd.uint32.type_id == types.UINT32
    assert ndd.int64.type_id == types.INT64
    assert ndd.uint64.type_id == types.UINT64
    assert ndd.float16.type_id == types.FLOAT16
    assert ndd.float32.type_id == types.FLOAT
    assert ndd.float64.type_id == types.FLOAT64
    assert ndd.bool.type_id == types.BOOL
    assert ndd.InterpType.type_id == types.INTERP_TYPE
    assert ndd.DataType.type_id == types.DATA_TYPE
    assert ndd.ImageType.type_id == types.IMAGE_TYPE


def test_type_ids_to_type():
    assert ndd.dtype(types.INT8) == ndd.int8
    assert ndd.dtype(types.UINT8) == ndd.uint8
    assert ndd.dtype(types.INT16) == ndd.int16
    assert ndd.dtype(types.UINT16) == ndd.uint16
    assert ndd.dtype(types.INT32) == ndd.int32
    assert ndd.dtype(types.UINT32) == ndd.uint32
    assert ndd.dtype(types.INT64) == ndd.int64
    assert ndd.dtype(types.UINT64) == ndd.uint64
    assert ndd.dtype(types.FLOAT16) == ndd.float16
    assert ndd.dtype(types.FLOAT) == ndd.float32
    assert ndd.dtype(types.FLOAT64) == ndd.float64
    assert ndd.dtype(types.BOOL) == ndd.bool
    assert ndd.dtype(types.INTERP_TYPE) == ndd.InterpType
    assert ndd.dtype(types.DATA_TYPE) == ndd.DataType
    assert ndd.dtype(types.IMAGE_TYPE) == ndd.ImageType


def test_type_names():
    assert ndd.int8.name == "i8"
    assert ndd.uint8.name == "u8"
    assert ndd.int16.name == "i16"
    assert ndd.uint16.name == "u16"
    assert ndd.int32.name == "i32"
    assert ndd.uint32.name == "u32"
    assert ndd.int64.name == "i64"
    assert ndd.uint64.name == "u64"
    assert ndd.float16.name == "f16"
    assert ndd.float32.name == "f32"
    assert ndd.float64.name == "f64"
    assert ndd.bool.name == "bool"
    assert ndd.bfloat16.name == "bfloat16"
    assert ndd.DataType.name == "DataType"
    assert ndd.ImageType.name == "ImageType"
    assert ndd.InterpType.name == "InterpType"


def test_type_names_to_type():
    assert ndd.dtype("i8") is ndd.int8
    assert ndd.dtype("u8") is ndd.uint8
    assert ndd.dtype("i16") is ndd.int16
    assert ndd.dtype("u16") is ndd.uint16
    assert ndd.dtype("i32") is ndd.int32
    assert ndd.dtype("u32") is ndd.uint32
    assert ndd.dtype("i64") is ndd.int64
    assert ndd.dtype("u64") is ndd.uint64
    assert ndd.dtype("f16") is ndd.float16
    assert ndd.dtype("f32") is ndd.float32
    assert ndd.dtype("f64") is ndd.float64
    assert ndd.dtype("bool") is ndd.bool
    assert ndd.dtype("bfloat16") is ndd.bfloat16
    assert ndd.dtype("DataType") is ndd.DataType
    assert ndd.dtype("ImageType") is ndd.ImageType
    assert ndd.dtype("InterpType") is ndd.InterpType


def test_type_bytes():
    assert ndd.int8.bytes == 1
    assert ndd.uint8.bytes == 1
    assert ndd.int16.bytes == 2
    assert ndd.uint16.bytes == 2
    assert ndd.int32.bytes == 4
    assert ndd.uint32.bytes == 4
    assert ndd.int64.bytes == 8
    assert ndd.uint64.bytes == 8
    assert ndd.float16.bytes == 2
    assert ndd.float32.bytes == 4
    assert ndd.float64.bytes == 8
    assert ndd.bool.bytes == 1
    assert ndd.InterpType.bytes == 4
    assert ndd.DataType.bytes == 4
    assert ndd.ImageType.bytes == 4


def test_type_exponent_bits():
    assert ndd.int8.exponent_bits is None
    assert ndd.uint8.exponent_bits is None
    assert ndd.int16.exponent_bits is None
    assert ndd.uint16.exponent_bits is None
    assert ndd.int32.exponent_bits is None
    assert ndd.uint32.exponent_bits is None
    assert ndd.int64.exponent_bits is None
    assert ndd.uint64.exponent_bits is None
    assert ndd.float16.exponent_bits == 5
    assert ndd.float32.exponent_bits == 8
    assert ndd.float64.exponent_bits == 11


def test_type_significand_bits():
    assert ndd.int8.significand_bits is None
    assert ndd.uint8.significand_bits is None
    assert ndd.int16.significand_bits is None
    assert ndd.uint16.significand_bits is None
    assert ndd.int32.significand_bits is None
    assert ndd.uint32.significand_bits is None
    assert ndd.int64.significand_bits is None
    assert ndd.uint64.significand_bits is None
    assert ndd.float16.significand_bits == 10
    assert ndd.float32.significand_bits == 23
    assert ndd.float64.significand_bits == 52
