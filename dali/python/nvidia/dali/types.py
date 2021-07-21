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
import re

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
        DALIDataType.INT8 : ("int", int),
        DALIDataType.INT16 : ("int", int),
        DALIDataType.INT32 : ("int", int),
        DALIDataType.INT64 : ("int", int),
        DALIDataType.UINT8 : ("int", int),
        DALIDataType.UINT16 : ("int", int),
        DALIDataType.UINT32 : ("int", int),
        # DALIDataType.UINT64 : ("int", int), # everything else fits into the Python int
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
        DALIDataType.TENSOR_LAYOUT : (":ref:`layout str<layout_str_doc>`", lambda x: str(x)),
        DALIDataType.PYTHON_OBJECT : ("object", lambda x: x),
        DALIDataType._TENSOR_LAYOUT_VEC : (":ref:`layout str<layout_str_doc>`", _to_list(lambda x: str(x))),
        DALIDataType._DATA_TYPE_VEC : ("nvidia.dali.types.DALIDataType", _to_list(lambda x: DALIDataType(int(x))))
        }

_vector_types = {
        DALIDataType._BOOL_VEC : DALIDataType.BOOL,
        DALIDataType._INT32_VEC : DALIDataType.INT32,
        DALIDataType._STRING_VEC : DALIDataType.STRING,
        DALIDataType._FLOAT_VEC : DALIDataType.FLOAT,
        DALIDataType._TENSOR_LAYOUT_VEC : DALIDataType.TENSOR_LAYOUT,
        DALIDataType._DATA_TYPE_VEC : DALIDataType.DATA_TYPE
        }

if _tfrecord_support:
    _known_types[DALIDataType.FEATURE] = ("nvidia.dali.tfrecord.Feature", tfrec.Feature)
    _known_types[DALIDataType._FEATURE_VEC] = ("nvidia.dali.tfrecord.Feature or " \
                                            "list of nvidia.dali.tfrecord.Feature",
                                            _to_list(tfrec.Feature))
    _known_types[DALIDataType._FEATURE_DICT] = ("dict of (string, nvidia.dali.tfrecord.Feature)",
            _not_implemented)

def _type_name_convert_to_string(dtype, allow_tensors):
    if dtype in _known_types:
        type_name = _known_types[dtype][0]
        ret = type_name
        if dtype in _vector_types:
            ret += " or list of " + type_name
        if allow_tensors:
            ret += " or TensorList of " + type_name
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

def _default_converter(dtype, default_value):
    if dtype in _enum_types:
        return str(_type_convert_value(dtype, default_value))
    else:
        return repr(_type_convert_value(dtype, default_value))


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

_enum_types = [DALIDataType.IMAGE_TYPE, DALIDataType.DATA_TYPE, DALIDataType.INTERP_TYPE]


class ScalarConstant(object):
    """
.. note::
    This class should not be instantiated directly; use :func:`Constant` function
    with appropriate arguments to create instances of this class.

Wrapper for a constant value that can be used in DALI :ref:`mathematical expressions`
and applied element-wise to the results of DALI Operators representing Tensors in
:meth:`nvidia.dali.Pipeline.define_graph` step.

ScalarConstant indicates what type should the value be treated as with respect
to type promotions. The actual values passed to the backend from python
would be `int32` for integer values and `float32` for floating point values.
Python builtin types `bool`, `int` and `float` will be marked to indicate
:const:`nvidia.dali.types.DALIDataType.BOOL`, :const:`nvidia.dali.types.DALIDataType.INT32`,
and :const:`nvidia.dali.types.DALIDataType.FLOAT` respectively.

Args
----
value: bool or int or float
    The constant value to be passed to DALI expression.
dtype: DALIDataType, optional
    Target type of the constant to be used in types promotions.
    """
    def __init__(self, value, dtype=None):
        self.shape = []
        value_dtype = getattr(value, "dtype", None)  # handle 0D tensors and numpy scalars
        if value_dtype is not None:
            dali_type = to_dali_type(value.dtype)
            if dali_type in _int_types:
                value = int(value)
            elif dali_type in _float_types:
                value = float(value)
            elif dali_type in _bool_types:
                value = bool(value)
            if dtype is None:
                dtype = dali_type

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
        # Delegate the call to the `__eq__` of other object, most probably a `DataNode`
        return other.__eq__(self)

    def __ne__(self, other):
        if isinstance(other, ScalarConstant):
            return self.value != other.value or self.dtype != other.dtype
        # Delegate the call to the `__ne__` of other object, most probably a `DataNode`
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
    return shape is None or shape == () or shape == [] or shape == 1 or \
           shape == [1] or shape == (1,)  # legacy pseudo-scalars

def _is_true_scalar(value):
    return len(getattr(value, "shape", ())) == 0

def _is_mxnet_array(value):
    return 'mxnet.ndarray.ndarray.NDArray' in str(type(value))

def _is_torch_tensor(value):
    return 'torch.Tensor' in str(type(value))

def _is_numpy_array(value):
    type_name = str(type(value))
    return  'numpy.ndarray' in type_name or \
            'numpy.int' in type_name or \
            'numpy.uint' in type_name or \
            'numpy.float' in type_name

def _raw_cuda_stream(stream_obj):
    if stream_obj is None:
        return None
    elif hasattr(stream_obj, "cuda_stream"):  # torch
        return stream_obj.cuda_stream
    elif hasattr(stream_obj, "ptr"):  # cupy
        return stream_obj.ptr
    else:
        return stream_obj

_cupy_array_type_regex = re.compile('.*cupy\..*\.ndarray.*')

def _is_cupy_array(value):
    return _cupy_array_type_regex.match(str(type(value)))

# common type names used by numpy, torch and possibly
_type_name_to_dali_type = {
    'bool'    : DALIDataType.BOOL,
    'boolean' : DALIDataType.BOOL,

    'int8'    : DALIDataType.INT8,
    'sbyte'   : DALIDataType.INT8,
    'uint8'   : DALIDataType.UINT8,
    'byte'    : DALIDataType.UINT8,
    'ubyte'   : DALIDataType.UINT8,

    'int16'   : DALIDataType.INT16,
    'short'   : DALIDataType.INT16,
    'uint16'  : DALIDataType.UINT16,
    'ushort'  : DALIDataType.UINT16,

    'int32'   : DALIDataType.INT32,
    'uint32'  : DALIDataType.UINT32,

    'int64'   : DALIDataType.INT64,
    'long'    : DALIDataType.INT64,
    'uint64'  : DALIDataType.UINT64,
    'ulong'   : DALIDataType.UINT64,

    'half'    : DALIDataType.FLOAT16,
    'float16' : DALIDataType.FLOAT16,
    'float'   : DALIDataType.FLOAT,
    'float32' : DALIDataType.FLOAT,
    'float64' : DALIDataType.FLOAT64,
    'double'  : DALIDataType.FLOAT64,
}

dali_type_converters = []

def to_dali_type(framework_type):
    t = str(framework_type)
    if t.startswith('torch.'):
        t = t[6:]
    t = _type_name_to_dali_type.get(t)
    if t is None:
        raise TypeError("'{}' could not be converted into any known DALIDataType.".format(framework_type));
    return t


def _is_compatible_array_type(value):
    return _is_numpy_array(value) or _is_mxnet_array(value) or _is_torch_tensor(value)

def ConstantNode(device, value, dtype, shape, layout, **kwargs):
    data = value
    if _is_mxnet_array(value):
        # mxnet ndarray is not directly compatible with numpy.ndarray, but provides conversion
        value = value.asnumpy()
    if _is_numpy_array(value):
        import numpy as np

        # 64-bit types require explicit dtype
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        if value.dtype == np.int64:
            value = value.astype(np.int32)
        if value.dtype == np.uint64:
            value = value.astype(np.uint32)

    if _is_numpy_array(value) or _is_torch_tensor(value):
        # torch tensor and numpy array have very similar API
        actual_type = to_dali_type(value.dtype)
        if dtype is None:
            dtype = actual_type
        if shape is not None:
            value = value.reshape(shape)
        else:
            shape = list(value.shape)  # torch uses torch.Size instead of list
        data = value.flatten().tolist()
    else:
        def _type_from_value_or_list(v):
            if not isinstance(v, (list, tuple)):
                v = [v]

            has_floats = False
            has_ints = False
            has_bools = False
            for x in v:
                if isinstance(x, float):
                    has_floats = True
                elif isinstance(x, bool):
                    has_bools = True
                elif isinstance(x, int):
                    has_ints = True
                else:
                    raise TypeError("Unexpected type: " + str(type(x)))

            if has_floats:
                return DALIDataType.FLOAT
            if has_ints:
                return DALIDataType.INT32
            if has_bools:
                return DALIDataType.BOOL
            # empty list defaults to float
            return DALIDataType.FLOAT

        actual_type = _type_from_value_or_list(value)
        if dtype is None:
            dtype = actual_type

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

    # 'name' argument is pased to call, not the constructor
    constructor_args = kwargs
    call_args = {}
    name = constructor_args.get("name")
    if name is not None:
        call_args["name"] = name
        del constructor_args["name"]

    op = ops.Constant(device = device, fdata = fdata, idata = idata,
                      shape = shape, dtype = dtype, layout = layout,
                      **constructor_args)
    return op(**call_args)

def _is_scalar_value(value):
    if value is None:
        return True
    if isinstance(value, (bool, int, float)):
        return True
    return not _is_compatible_array_type(value) or _is_scalar_shape(value.shape)

def Constant(value, dtype = None, shape = None, layout = None, device = None, **kwargs):
    """Wraps a constant value which can then be used in
:meth:`nvidia.dali.Pipeline.define_graph` pipeline definition step.

If the `value` argument is a scalar and neither `shape`, `layout` nor
`device` is provided, the function will return a :class:`ScalarConstant`
wrapper object, which receives special, optimized treatment when used in
:ref:`mathematical expressions`.

Otherwise, the function creates a `dali.ops.Constant` node, which produces
a batch of constant tensors.

Args
----
value: `bool`, `int`, `float`, a `list` or `tuple` thereof or a `numpy.ndarray`
    The constant value to wrap. If it is a scalar, it can be used as scalar
    value in mathematical expressions. Otherwise, it will produce a constant
    tensor node (optionally reshaped according to `shape` argument).
    If this argument is is a numpy array, a PyTorch tensor or an MXNet array,
    the values of `shape` and `dtype` will default to `value.shape` and `value.dtype`,
    respectively.
dtype: DALIDataType, optional
    Target type of the constant.
shape: list or tuple of int, optional
    Requested shape of the output. If `value` is a scalar, it is broadcast
    as to fill the requested shape. Otherwise, the number of elements in
    `value` must match the volume of the shape.
layout: string, optional
    A string descirbing the layout of the constant tensor, e.g. "HWC"
device: string, optional, "cpu" or "gpu"
    The device to place the constant tensor in. If specified, it forces
    the value to become a constant tensor node on given device,
    regardless of `value` type or `shape`.
**kwargs: additional keyword arguments
    If present, it forces the constant to become a Constant tensor node
    and the arguments are passed to the `dali.ops.Constant` operator
    """

    if device is not None or \
        (_is_compatible_array_type(value) and not _is_true_scalar(value)) or \
        isinstance(value, (list, tuple)) or \
        not _is_scalar_shape(shape) or \
        kwargs or \
        layout is not None:
        return ConstantNode(device, value, dtype, shape, layout, **kwargs)
    else:
        return ScalarConstant(value, dtype)

class SampleInfo:
    """
    Describes the indices of a sample requested from :meth:`nvidia.dali.fn.external_source`

    :ivar idx_in_epoch: 0-based index of the sample witin epoch
    :ivar idx_in_batch: 0-based index of the sample within batch
    :ivar iteration:    number of current batch within epoch
    """
    def __init__(self, idx_in_epoch, idx_in_batch, iteration):
        self.idx_in_epoch = idx_in_epoch
        self.idx_in_batch = idx_in_batch
        self.iteration = iteration
