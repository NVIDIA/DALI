# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#pylint: disable=no-name-in-module,unused-import
from nvidia.dali.backend_impl.types import *
from nvidia.dali.backend_impl import CUDAEvent
try:
    from nvidia.dali import tfrecord as tfrec
    _tfrecord_support = True
except ImportError:
    _tfrecord_support = False

def _to_list(func):
    def _to_list_instance(val):
        if isinstance(val, (list, tuple)):
            return [func(v) for v in val]
        else:
            return [func(val)]
    return _to_list_instance

def _not_implemented(val):
    raise NotImplementedError()

_known_types = {
        DALIDataType.INT32 : ("int", int),
        DALIDataType.INT64 : ("int", int),
        DALIDataType.FLOAT : ("float", float),
        DALIDataType.BOOL : ("bool", bool),
        DALIDataType.STRING : ("str", str),
        DALIDataType._BOOL_VEC : ("bool or list of bool", _to_list(bool)),
        DALIDataType._INT32_VEC : ("int or list of int",_to_list(int)),
        DALIDataType._STRING_VEC : ("str or list of str", _to_list(str)),
        DALIDataType._FLOAT_VEC : ("float or list of float", _to_list(float)),
        DALIDataType.IMAGE_TYPE : ("nvidia.dali.types.DALIImageType", lambda x: DALIImageType(int(x))),
        DALIDataType.DATA_TYPE : ("nvidia.dali.types.DALIDataType", lambda x: DALIDataType(int(x))),
        DALIDataType.INTERP_TYPE : ("nvidia.dali.types.DALIInterpType", lambda x: DALIInterpType(int(x))),
        DALIDataType.TENSOR_LAYOUT : ("nvidia.dali.types.TensorLayout", lambda x: TensorLayout(str(x))),
        DALIDataType.PYTHON_OBJECT : ("object", lambda x: x)
        }
_vector_types = {
        DALIDataType._BOOL_VEC : DALIDataType.BOOL,
        DALIDataType._INT32_VEC : DALIDataType.INT32,
        DALIDataType._STRING_VEC : DALIDataType.STRING,
        DALIDataType._FLOAT_VEC : DALIDataType.FLOAT,
        }

if _tfrecord_support:
    _known_types[DALIDataType.FEATURE] = ("nvidia.dali.tfrecord.Feature", tfrec.Feature)
    _known_types[DALIDataType._FEATURE_VEC] = ("nvidia.dali.tfrecord.Feature or " \
                                            "list of nvidia.dali.tfrecord.Feature",
                                            _to_list(tfrec.Feature))
    _known_types[DALIDataType._FEATURE_DICT] = ("dict of (string, nvidia.dali.tfrecord.Feature)",
            _not_implemented)

def _type_name_convert_to_string(dtype, is_tensor):
    if dtype in _known_types:
        ret = _known_types[dtype][0]
        if is_tensor:
            ret += " or " + ret + " tensor"
        return ret
    else:
        raise RuntimeError(str(dtype) + " does not correspond to a known type.")

def _type_convert_value(dtype, val):
    if dtype not in _known_types:
        raise RuntimeError(str(dtype) + " does not correspond to a known type.")
    return _known_types[dtype][1](val)

def _vector_element_type(dtype):
    if dtype not in _vector_types:
        raise RuntimeError(str(dtype) + " is not a vector type.")
    return _vector_types[dtype]

class PipelineAPIType(object):
    """Pipeline API type
    """
    @staticmethod
    def _is_member(self):
        return PipelineAPIType.__dict__.keys()
    BASIC = 0
    ITERATOR = 1
    SCHEDULED = 2


class CUDAStream:
    """Wrapper class for a CUDA stream."""
    def __init__(self, ptr=0):
        self._ptr = ptr

    @property
    def ptr(self):
        """Raw CUDA stream pointer, stored as uint64."""
        return self._ptr



_float_types = [DALIDataType.FLOAT16, DALIDataType.FLOAT, DALIDataType.FLOAT64]
_int_types = [DALIDataType.INT8, DALIDataType.INT16, DALIDataType.INT32, DALIDataType.INT64,
              DALIDataType.UINT8, DALIDataType.UINT16, DALIDataType.UINT32, DALIDataType.UINT64]

class Constant(object):
    """Wrapper for a constant value that can be used in arithmetic operations
    with the results of DALI Operators in `define_graph()` step.

    It indicates what type it should be treated as. The integers values
    will be passed to DALI as `int32` and the floating point values as `float32`.
    Python builtin types `int` and `float` will also be treated as those types.
    """
    def __init__(self, value, dtype=None):
        if not isinstance(value, (int, float)):
            raise TypeError("Expected scalar value of type 'int' or 'float', got {}."
                    .format(str(type(value))))
        if dtype:
            self.dtype = dtype
            if self.dtype in _int_types:
                self.value = int(value)
            elif self.dtype in _float_types:
                self.value = float(value)
            else:
                raise TypeError("DALI Constant can only hold one of: {} types."
                        .format(_int_types + _float_types))
        elif isinstance(value, int):
            self.value = value
            self.dtype = DALIDataType.INT32
        elif isinstance(value, float):
            self.value = value
            self.dtype = DALIDataType.FLOAT

    def int8(self):
        return Constant(self.value, DALIDataType.INT8)

    def int16(self):
        return Constant(self.value, DALIDataType.INT16)

    def int32(self):
        return Constant(self.value, DALIDataType.INT32)

    def int64(self):
        return Constant(self.value, DALIDataType.INT64)

    def uint8(self):
        return Constant(self.value, DALIDataType.UINT8)

    def uint16(self):
        return Constant(self.value, DALIDataType.UINT16)

    def uint32(self):
        return Constant(self.value, DALIDataType.UINT32)

    def uint64(self):
        return Constant(self.value, DALIDataType.UINT64)

    def float16(self):
        return Constant(self.value, DALIDataType.FLOAT16)

    def float32(self):
        return Constant(self.value, DALIDataType.FLOAT)

    def float64(self):
        return Constant(self.value, DALIDataType.FLOAT64)

    def __eq__(self, other):
        return self.value == other.value and self.dtype == other.dtype

    def __int__(self):
        if self.dtype in _int_types:
            return self.value
        raise TypeError("DALI Constant must be converted to one of int types explicitly before "
                "casting to builtin 'int'.")

    def __float__(self):
        if self.dtype in _float_types:
            return self.value
        raise TypeError("DALI Constant must be converted to one of float types explicitly before "
                "casting to builtin 'float'.")

    def __str__(self):
        return "{}:{}".format(self.value, self.dtype)

    def __repr__(self):
        return "{}".format(self.value)
