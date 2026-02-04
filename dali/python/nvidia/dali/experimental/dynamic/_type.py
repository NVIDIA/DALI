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

_id2type = {}
_type2id = {}
_name2type = {}


class DType:
    """
    DALI Dynamic Mode data type.

    This class is used to represent the data type of a Tensor or Batch.
    Data types such as ``uint32`` are instances of this class.

    .. note::
        This class is used to represent the types supported by DALI.
        Creation of custom user types is not supported.

    Attributes:
        bits: Size of the type, in bits.
        bytes: Size of the type, in bytes.
        exponent_bits: Number of exponent bits in a floating-point number.
        kind: kind of the type.
        name: Name of the type.
        significand_bits: Number of significand bits in a floating-point number.
        type_id: Corresponding DALI data type
    """

    class Kind(Enum):
        """
        Enum defining type kind in DALI dynamic mode.

        Attributes:
            signed:     Signed integer
            unsigned:   Unsigned integer
            float:      Floating-point number
            bool:       Boolean
            enum:       Enumerator type
        """

        signed = auto()
        unsigned = auto()
        float = auto()
        bool = auto()
        enum = auto()

    bits: int
    bytes: int
    exponent_bits: int
    kind: Kind
    name: str
    significand_bits: int
    type_id: nvidia.dali.types.DALIDataType

    @staticmethod
    def _default_exponent_bits(bits: int, error_on_unknown: bool = False) -> int:
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
            if error_on_unknown:
                raise ValueError(f"Unsupported number of bits: {bits}")
            else:
                return None

    @staticmethod
    def _default_significand_bits(bits: int, error_on_unknown: bool = False) -> int:
        """
        Returns the default number of significand bits for a given number of bits.
        """
        exp = DType._default_exponent_bits(bits, error_on_unknown)
        if exp is None:
            return None
        return bits - exp - 1

    def __init__(
        self,
        kind: Kind,
        bits: int,
        exponent_bits: int = None,
        significand_bits: int = None,
        bytes: int = None,
        type_id: nvidia.dali.types.DALIDataType = None,
        name: str = None,
        docs: str = None,
    ):
        self.kind = kind
        self.bits = bits
        self.type_id = type_id

        if kind == DType.Kind.float:
            self.exponent_bits = exponent_bits or DType._default_exponent_bits(bits, True)
            self.significand_bits = significand_bits or DType._default_significand_bits(bits, True)
        else:
            self.exponent_bits = None
            self.significand_bits = None
        if name and docs:
            self.name, self.__doc__ = name, docs
        else:
            generated_name, generated_docs = DType._make_name_and_docs(
                kind, bits, self.exponent_bits, self.significand_bits
            )
            self.name = name or generated_name
            self.__doc__ = docs or generated_docs
        self.bytes = bytes or ((bits + 7) // 8)

        # Register the type with the id and name
        if type_id is not None:
            _id2type[type_id] = self
            _type2id[self] = type_id
        _name2type[self.name] = self

    @staticmethod
    def _make_name_and_docs(
        kind: Kind, bits: int, exponent_bits: int, significand_bits: int
    ) -> str:
        if kind == DType.Kind.signed:
            return f"i{bits}", f"{bits}-bit signed integer"
        elif kind == DType.Kind.unsigned:
            return f"u{bits}", f"{bits}-bit unsigned integer"
        elif kind == DType.Kind.float:
            if exponent_bits == DType._default_exponent_bits(
                bits
            ) and significand_bits == DType._default_significand_bits(bits):
                return f"f{bits}", f"{bits}-bit floating point number"
            elif bits == 16 and exponent_bits == 8 and significand_bits == 7:
                return (
                    "bfloat16",
                    "Brain Floating Point. A 16-bit floating point number with 8-bit exponent "
                    "and 7-bit mantissa",
                )
            elif bits == 19 and exponent_bits == 8 and significand_bits == 10:
                return (
                    "tf32",
                    "TensorFloat-32. A 19-bit floating point number with 8-bit exponent and "
                    "10-bit mantissa",
                )
            else:
                return (
                    f"f{bits}e{exponent_bits}m{significand_bits}",
                    f"{bits}-bit floating point number with {exponent_bits}-bit exponent and "
                    f"{significand_bits}-bit mantissa",
                )
        elif kind == DType.Kind.bool:
            return "bool", "Boolean value"
        else:
            raise ValueError(f"Cannot make name for type of kind: {kind}")

    def __str__(self):
        return self.name

    def __repr__(self):
        repr_str = f"DType(kind={self.kind}, bits={self.bits}"
        if self.exponent_bits is not None:
            repr_str += f", exponent_bits={self.exponent_bits}"
        if self.significand_bits is not None:
            repr_str += f", significand_bits={self.significand_bits}"
        repr_str += ")"
        return repr_str

    def __eq__(self, other):
        if not (
            self.kind == other.kind
            and self.bits == other.bits
            and self.significand_bits == other.significand_bits
            and self.exponent_bits == other.exponent_bits
        ):
            return False
        if self.kind == DType.Kind.enum:
            return self.type_id == other.type_id
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        h = hash((self.kind, self.bits, self.exponent_bits, self.significand_bits))
        if self.kind == DType.Kind.enum:
            h = hash((h, self.type_id))
        return h

    @staticmethod
    def from_type_id(type_id: nvidia.dali.types.DALIDataType) -> "DType":
        """
        Returns a :py:class:`DType` associated with given
        :py:class:`nvidia.dali.types.DALIDataType`.
        """
        return _id2type[type_id]

    @staticmethod
    def from_fw_type(numpy_type) -> "DType":
        """
        Returns a :py:class:`DType` associated with given framework datatype.
        """
        return nvidia.dali.types.to_dali_type(numpy_type)

    @staticmethod
    def parse(name: str) -> "DType":
        """
        Parses a type name into a :py:class:`DType`.
        """
        if _name2type.get(name) is not None:
            return _name2type[name]

        def parse_internal(name: str) -> "DType":
            if name.startswith("i"):
                return DType(DType.Kind.signed, int(name[1:]))
            elif name.startswith("u"):
                return DType(DType.Kind.unsigned, int(name[1:]))
            elif name == "bool":
                return DType(DType.Kind.bool, 8)
            elif name == "bfloat16":
                return DType(DType.Kind.float, 16, 8, 7)
            elif name == "tf32":
                return DType(DType.Kind.float, 19, 8, 10, bytes=4)
            elif name.startswith("f"):
                exp = name.find("e")
                sig = name.find("m")
                if exp != -1 or sig != -1:
                    if exp == -1 or sig == -1:
                        raise ValueError(f"Invalid type name: {name}")
                    if exp < 2 or sig < exp + 2:
                        raise ValueError(f"Invalid type name: {name}")
                    bits = int(name[1:exp])
                    exp_bits = int(name[exp + 1 : sig])
                    sig_bits = int(name[sig + 1 :])
                    return DType(DType.Kind.float, bits, exp_bits, sig_bits)
                else:
                    bits = int(name[1:])
                    return DType(DType.Kind.float, bits)
            elif name == "DataType":
                return DataType
            elif name == "ImageType":
                return ImageType
            elif name == "InterpType":
                return InterpType
            else:
                raise ValueError(f"Unsupported type name: {name}")

        t = parse_internal(name)
        if t.type_id is None:
            t.type_id = _type2id[t]
        if t.type_id is not None:
            t = _id2type[t.type_id]  # use the same DType instance as the one registered with the id
        _name2type[name] = t
        return t

    def __call__(self, *args, **kwargs):
        """
        Create a new Tensor with this type.
        """
        if "dtype" in kwargs:
            raise ValueError("dtype cannot be overridden")
        from . import _tensor

        return _tensor.tensor(self, *args, dtype=self, **kwargs)


int8 = DType(DType.Kind.signed, 8, type_id=nvidia.dali.types.INT8)
int16 = DType(DType.Kind.signed, 16, type_id=nvidia.dali.types.INT16)
int32 = DType(DType.Kind.signed, 32, type_id=nvidia.dali.types.INT32)
int64 = DType(DType.Kind.signed, 64, type_id=nvidia.dali.types.INT64)
uint8 = DType(DType.Kind.unsigned, 8, type_id=nvidia.dali.types.UINT8)
uint16 = DType(DType.Kind.unsigned, 16, type_id=nvidia.dali.types.UINT16)
uint32 = DType(DType.Kind.unsigned, 32, type_id=nvidia.dali.types.UINT32)
uint64 = DType(DType.Kind.unsigned, 64, type_id=nvidia.dali.types.UINT64)
float16 = DType(DType.Kind.float, 16, type_id=nvidia.dali.types.FLOAT16)
float32 = DType(DType.Kind.float, 32, type_id=nvidia.dali.types.FLOAT)
float64 = DType(DType.Kind.float, 64, type_id=nvidia.dali.types.FLOAT64)
bool = DType(DType.Kind.bool, 8, type_id=nvidia.dali.types.BOOL)
bfloat16 = DType(DType.Kind.float, 16, 8, 7)  # TODO(michalz): Add type_id for bfloat16
DataType = DType(
    DType.Kind.enum,
    32,
    type_id=nvidia.dali.types.DATA_TYPE,
    name="DataType",
    docs="DALI data type. See :py:class:`nvidia.dali.types.DALIDataType`.",
)
ImageType = DType(
    DType.Kind.enum,
    32,
    type_id=nvidia.dali.types.IMAGE_TYPE,
    name="ImageType",
    docs="Image type. See :py:class:`nvidia.dali.types.DALIImageType`",
)
InterpType = DType(
    DType.Kind.enum,
    32,
    type_id=nvidia.dali.types.INTERP_TYPE,
    name="InterpType",
    docs="Interpolation type. See :py:class:`nvidia.dali.types.DALIInterpType`",
)


def dtype(*args):
    """
    Returns a :py:class:`DType` associated with given :py:class:`nvidia.dali.types.DALIDataType`,
    string or framework datatype.
    """
    if len(args) == 1:
        if isinstance(args[0], DType):
            return args[0]
        elif isinstance(args[0], nvidia.dali.types.DALIDataType):
            return DType.from_type_id(args[0])
        elif isinstance(args[0], str):
            return DType.parse(args[0])
        else:
            return DType.from_fw_type(args[0])
    else:
        return DType(*args)


def type_id(dtype) -> nvidia.dali.types.DALIDataType:
    """
    Returns :py:class:`nvidia.dali.types.DALIDataType` for the given dtype.
    """
    if isinstance(dtype, DType):
        return dtype.type_id
    elif isinstance(dtype, nvidia.dali.types.DALIDataType):
        return dtype
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


__all__ = sum(
    [
        # types and functions
        ["DType", "dtype", "type_id"],
        # integral types
        ["int8", "int16", "int32", "int64"],
        # unsigned integral types
        ["uint8", "uint16", "uint32", "uint64"],
        # floating point types
        ["float16", "float32", "float64", "bfloat16"],
        # boolean and enum types
        ["bool", "DataType", "ImageType", "InterpType"],
    ],
    [],
)
