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
from enum import Enum, unique

from nvidia.dali.backend_impl.types import *
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
        DALIDataType._BOOL_VEC : ("bool", _to_list(bool)),
        DALIDataType._INT32_VEC : ("int", _to_list(int)),
        DALIDataType._STRING_VEC : ("str", _to_list(str)),
        DALIDataType._FLOAT_VEC : ("float", _to_list(float)),
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
            ret = "TensorList of " + ret
        elif dtype in _vector_types:
            ret = ret + " or list of " + _known_types[dtype][0]
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

@unique
class PipelineAPIType(Enum):
    """Pipeline API type
    """
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

_bool_types = [DALIDataType.BOOL]
_int_types = [DALIDataType.INT8, DALIDataType.INT16, DALIDataType.INT32, DALIDataType.INT64,
              DALIDataType.UINT8, DALIDataType.UINT16, DALIDataType.UINT32, DALIDataType.UINT64]
_float_types = [DALIDataType.FLOAT16, DALIDataType.FLOAT, DALIDataType.FLOAT64]

_int_like_types = _bool_types + _int_types
_all_types = _bool_types + _int_types + _float_types


class ScalarConstant(object):
    """Wrapper for a constant value that can be used in DALI arithmetic expressions
    and applied element-wise to the results of DALI Operators representing Tensors in
    :meth:`nvidia.dali.pipeline.Pipeline.define_graph` step.

    ScalarConstant indicates what type should the value be treated as with respect
    to type promotions. The actual values passed to the backend from python
    would be `int32` for integer values and `float32` for floating point values.
    Python builtin types `bool`, `int` and `float` will be marked to indicate
    :meth:`nvidia.dali.types.DALIDataType.BOOL`, :meth:`nvidia.dali.types.DALIDataType.INT32`,
    and :meth:`nvidia.dali.types.DALIDataType.FLOAT` respectively.

    Args
    ----
    value: bool or int or float
        The constant value to be passed to DALI expression.
    `dtype`: DALIDataType, optional
        Target type of the constant to be used in types promotions.
    """
    def __init__(self, value, dtype=None):
        if not isinstance(value, (bool, int, float)):
            raise TypeError(
                "Expected scalar value of type 'bool', 'int' or 'float', got {}."
                .format(str(type(value))))

        if dtype:
            self.dtype = dtype
            if self.dtype in _bool_types:
                self.value = bool(value)
            elif self.dtype in _int_types:
                self.value = int(value)
            elif self.dtype in _float_types:
                self.value = float(value)
            else:
                raise TypeError("DALI ScalarConstant can only hold one of: {} types."
                        .format(_all_types))
        elif isinstance(value, bool):
            self.value = value
            self.dtype = DALIDataType.BOOL
        elif isinstance(value, int):
            self.value = value
            self.dtype = DALIDataType.INT32
        elif isinstance(value, float):
            self.value = value
            self.dtype = DALIDataType.FLOAT



    def bool(self):
        return ScalarConstant(self.value, DALIDataType.BOOL)

    def int8(self):
        return ScalarConstant(self.value, DALIDataType.INT8)

    def int16(self):
        return ScalarConstant(self.value, DALIDataType.INT16)

    def int32(self):
        return ScalarConstant(self.value, DALIDataType.INT32)

    def int64(self):
        return ScalarConstant(self.value, DALIDataType.INT64)

    def uint8(self):
        return ScalarConstant(self.value, DALIDataType.UINT8)

    def uint16(self):
        return ScalarConstant(self.value, DALIDataType.UINT16)

    def uint32(self):
        return ScalarConstant(self.value, DALIDataType.UINT32)

    def uint64(self):
        return ScalarConstant(self.value, DALIDataType.UINT64)

    def float16(self):
        return ScalarConstant(self.value, DALIDataType.FLOAT16)

    def float32(self):
        return ScalarConstant(self.value, DALIDataType.FLOAT)

    def float64(self):
        return ScalarConstant(self.value, DALIDataType.FLOAT64)

    def __eq__(self, other):
        if isinstance(other, ScalarConstant):
            return self.value == other.value and self.dtype == other.dtype
        # Delegate the call to the `__eq__` of other object, most probably an `_EdgeReference`
        return other.__eq__(self)

    def __ne__(self, other):
        if isinstance(other, ScalarConstant):
            return self.value != other.value or self.dtype != other.dtype
        # Delegate the call to the `__ne__` of other object, most probably an `_EdgeReference`
        return other.__ne__(self)

    def __bool__(self):
        if self.dtype in _int_like_types:
            return bool(self.value)
        raise TypeError(("DALI ScalarConstant must be converted to one of bool or int types: ({}) "
                "explicitly before casting to builtin `bool`.").format(_int_like_types))

    def __int__(self):
        if self.dtype in _int_like_types:
            return int(self.value)
        raise TypeError(("DALI ScalarConstant must be converted to one of bool or int types: ({}) "
                "explicitly before casting to builtin `int`.").format(_int_like_types))

    def __float__(self):
        if self.dtype in _float_types:
            return self.value
        raise TypeError(("DALI ScalarConstant must be converted to one of the float types: ({}) "
                "explicitly before casting to builtin `float`.").format(_float_types))

    def __str__(self):
        return "{}:{}".format(self.value, self.dtype)

    def __repr__(self):
        return "{}".format(self.value)

def _is_scalar_shape(shape):
    return shape is None or shape == 1 or shape == [1]

def _is_numpy_array(value):
    if 'numpy.ndarray' in str(type(value)):
        return True

def ConstantNode(device, value, dtype, shape, layout):
    data = value
    if _is_numpy_array(value):
        import numpy as np

        # 64-bit types are not supported - downgrade the input
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        if value.dtype == np.int64:
            value = value.astype(np.int32)
        if value.dtype == np.uint64:
            value = value.astype(np.uint32)

        def _numpy_to_dali_type(t):
            if t is None:
                return None
            import numpy as np
            if t == np.bool:
                return DALIDataType.BOOL

            if t == np.float16:
                return DALIDataType.FLOAT16
            if t == np.float32:
                return DALIDataType.FLOAT
            if t == np.float64:
                return DALIDataType.FLOAT64

            if t == np.uint8:
                return DALIDataType.UINT8
            if t == np.int8:
                return DALIDataType.INT8
            if t == np.uint16:
                return DALIDataType.UINT16
            if t == np.int16:
                return DALIDataType.INT16
            if t == np.uint32:
                return DALIDataType.UINT32
            if t == np.int32:
                return DALIDataType.INT32
            if t == np.uint64:
                return DALIDataType.UINT64
            if t == np.int64:
                return DALIDataType.INT64
            raise TypeError("Unsupported type: " + str(t))

        actual_type = _numpy_to_dali_type(value.dtype)
        if dtype is None:
            dtype = actual_type
        if shape is not None:
            value = np.copy(value.reshape(shape), order='C')
        else:
            value = np.copy(value, order='C')
            shape = value.shape
        data = value.flatten().tolist()
    else:
        def _type_from_value_or_list(v):
            if isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, float):
                        return DALIDataType.FLOAT
                return DALIDataType.INT32

            if isinstance(v, float):
                return DALIDataType.FLOAT
            return DALIDataType.INT32

        actual_type = _type_from_value_or_list(value)

    import nvidia.dali.ops as ops

    def _convert(x, type):
        if isinstance(x, (list, tuple)):
            return [type(y) for y in x]
        return type(x)

    isint = actual_type in _int_like_types
    idata = _convert(data, int) if isint else None
    fdata = None if isint else data
    if device is None:
        device = "cpu"

    op = ops.Constant(device = device, fdata = fdata, idata = idata,
                      shape = shape, dtype = dtype, layout = layout)
    return op()

def Constant(value, dtype = None, shape = None, layout = None, device = None):
    if device is not None or \
        _is_numpy_array(value) or \
        isinstance(value, (list, tuple)) or \
        not _is_scalar_shape(shape) or \
        layout is not None:
        return ConstantNode(device, value, dtype, shape, layout)
    else:
        return ScalarConstant(value, dtype)
