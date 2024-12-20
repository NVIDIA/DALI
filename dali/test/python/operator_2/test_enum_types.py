# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import fn, pipeline_def, types

import numpy as np
import tree

from nose_utils import assert_raises
from nose2.tools import params


@params(
    *[
        # Automatic promotion
        lambda value, dtype: fn.copy(value),
        # Explicit conversion to constant op
        lambda value, dtype: types.Constant(value=value, dtype=dtype),
        # Detection of type from value
        lambda value, dtype: types.Constant(value=value),
        # Explicit type when passed the underlying numeric value of the enum
        lambda value, dtype: types.Constant(
            value=tree.map_structure(lambda v: v.value, value), dtype=dtype
        ),
    ]
)
def test_enum_constant_capture(converter):
    batch_size = 2

    scalar_v = types.DALIDataType.INT16
    list_v = [
        types.DALIInterpType.INTERP_CUBIC,
        types.DALIInterpType.INTERP_GAUSSIAN,
        types.DALIInterpType.INTERP_LANCZOS3,
    ]

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def enum_constant_pipe():
        scalar = converter(scalar_v, types.DALIDataType.DATA_TYPE)
        tensor = converter(list_v, types.DALIDataType.INTERP_TYPE)

        scalar_as_int = fn.cast(scalar, dtype=types.DALIDataType.INT32)
        tensor_as_int = fn.cast(tensor, dtype=types.DALIDataType.INT32)
        return scalar, tensor, scalar_as_int, tensor_as_int

    pipe = enum_constant_pipe()
    scalar, tensor, scalar_as_int, tensor_as_int = pipe.run()
    assert scalar.dtype == types.DALIDataType.DATA_TYPE
    assert scalar.shape() == [()] * batch_size, f"{scalar.shape}"
    assert tensor.dtype == types.DALIDataType.INTERP_TYPE
    assert tensor.shape() == [(3,)] * batch_size
    # Compare the cast values with Python values
    for i in range(batch_size):
        assert np.array_equal(np.array(scalar_as_int[i]), np.array(scalar_v.value))
        assert np.array_equal(np.array(tensor_as_int[i]), np.array([elem.value for elem in list_v]))
    with assert_raises(
        TypeError,
        glob="DALI enum types cannot be used with buffer protocol*"
        "use `nvidia.dali.fn.cast` to convert",
    ):
        scalar.as_array()


def test_scalar_constant():
    with assert_raises(
        TypeError, glob="Expected scalar value of type 'bool', 'int' or 'float', got *.DALIDataType"
    ):
        types.ScalarConstant(types.DALIDataType.INT16)


@params(*[(1.0, types.DALIDataType.DATA_TYPE), (types.DALIImageType.RGB, types.DALIDataType.FLOAT)])
def test_prohibited_cast(param, dtype):
    @pipeline_def(batch_size=2, device_id=0, num_threads=4)
    def pipeline():
        return fn.cast(param, dtype=dtype)

    with assert_raises(
        RuntimeError,
        glob="Cannot cast from *float*. Enums can only participate "
        "in casts with integral types, but not floating point types.",
    ):
        p = pipeline()
        p.run()
