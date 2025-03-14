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

from enum import Enum, auto
import nvidia.dali.types


class DType:
    class Kind(Enum):
        signed = auto()
        unsigned = auto()
        float = auto()
        bool = auto()

    @staticmethod
    def default_exponent_bits(bits: int) -> int:
        """
        Returns the default number of exponent bits for a given number of bits.
        """
        if bits == 16:
            return 5
        elif bits == 32:
            return 8
        elif bits == 64:
            return 11
        elif bits == 8:
            return 4
        else:
            raise ValueError(f"Unsupported number of bits: {bits}")

    @staticmethod
    def default_significand_bits(bits: int) -> int:
        """
        Returns the default number of significand bits for a given number of bits.
        """
        return bits - DType.default_exponent_bits(bits) - 1

    def __init__(
        self, kind: Kind, bits: int, exponent_bits: int = None, significand_bits: int = None
    ):
        self.kind = kind
        self.bits = bits
        if kind == DType.Kind.float:
            self.exponent_bits = exponent_bits or DType.default_exponent_bits(bits)
            self.significand_bits = significand_bits or DType.default_significand_bits(bits)
        else:
            self.exponent_bits = None
            self.significand_bits = None
        self.name = DType.make_name(kind, bits, exponent_bits, significand_bits)
        self.bytes = (bits + 7) // 8

    @staticmethod
    def make_name(kind: Kind, bits: int, exponent_bits: int, significand_bits: int) -> str:
        if kind == DType.Kind.signed:
            return f"i{bits}"
        elif kind == DType.Kind.unsigned:
            return f"u{bits}"
        elif kind == DType.Kind.float:
            if exponent_bits == DType.default_exponent_bits(
                bits
            ) and significand_bits == DType.default_significand_bits(bits):
                return f"f{bits}"
            elif bits == 16 and exponent_bits == 8 and significand_bits == 7:
                return "bfloat16"
            elif bits == 32 and exponent_bits == 8 and significand_bits == 10:
                return "tf32"
            else:
                return f"f{bits}e{exponent_bits}m{significand_bits}"
        elif kind == DType.Kind.bool:
            return "bool"

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Type(kind={self.kind}, bits={self.bits}, exponent_bits={self.exponent_bits}, significand_bits={self.significand_bits})"

    def __eq__(self, other):
        return self.kind == other.kind and self.bits == other.bits

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.kind, self.bits, self.exponent_bits, self.significand_bits))

    def type_id(self) -> int:
        return _type2id[self]

    @staticmethod
    def from_type_id(type_id: int) -> "DType":
        return _id2type[type_id]


int8 = DType(DType.Kind.signed, 8)
int16 = DType(DType.Kind.signed, 16)
int32 = DType(DType.Kind.signed, 32)
int64 = DType(DType.Kind.signed, 64)
uint8 = DType(DType.Kind.unsigned, 8)
uint16 = DType(DType.Kind.unsigned, 16)
uint32 = DType(DType.Kind.unsigned, 32)
uint64 = DType(DType.Kind.unsigned, 64)
float16 = DType(DType.Kind.float, 16)
float32 = DType(DType.Kind.float, 32)
float64 = DType(DType.Kind.float, 64)
bool = DType(DType.Kind.bool, 8)
bfloat16 = DType(DType.Kind.float, 16, 8, 7)
tf32 = DType(DType.Kind.float, 32, 8, 10)
f8e4m3 = DType(DType.Kind.float, 8, 4, 3)
f8e5m2 = DType(DType.Kind.float, 8, 5, 2)

_type2id = {
    int8: nvidia.dali.types.INT8,
    int16: nvidia.dali.types.INT16,
    int32: nvidia.dali.types.INT32,
    int64: nvidia.dali.types.INT64,
    uint8: nvidia.dali.types.UINT8,
    uint16: nvidia.dali.types.UINT16,
    uint32: nvidia.dali.types.UINT32,
    uint64: nvidia.dali.types.UINT64,
    float16: nvidia.dali.types.FLOAT16,
    float32: nvidia.dali.types.FLOAT,
    float64: nvidia.dali.types.FLOAT64,
    bool: nvidia.dali.types.BOOL,
}

_id2type = {v: k for k, v in _type2id.items()}
