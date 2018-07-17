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
        DALIDataType.TENSOR_LAYOUT : ("nvidia.dali.types.DALITensorLayout", lambda x: DALITensorLayout(int(x))),
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

