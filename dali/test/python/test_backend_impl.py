# Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import warnings
from numpy.testing import assert_array_equal
from nvidia.dali import pipeline_def
from nvidia.dali.backend_impl import TensorCPU, TensorListCPU, TensorListGPU
from nvidia.dali.backend_impl import types as types_

from nose_utils import assert_raises
from test_utils import dali_type_to_np, py_buffer_from_address


def test_create_tensor():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr, "NHWC")
    assert_array_equal(arr, np.array(tensor))


def test_create_tensorlist():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    assert_array_equal(arr, tensorlist.as_array())


def test_create_tensorlist_as_tensor():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    tensor = tensorlist.as_tensor()
    assert_array_equal(np.array(tensor), tensorlist.as_array())


def test_empty_tensor_tensorlist():
    arr = np.array([], dtype=np.float32)
    tensor = TensorCPU(arr, "NHWC")
    tensorlist = TensorListCPU(arr, "NHWC")
    assert_array_equal(np.array(tensor), tensorlist.as_array())
    assert (np.array(tensor).shape == (0,))
    assert (tensorlist.as_array().shape == (0,))


def test_tensorlist_getitem_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    list_of_tensors = [x for x in tensorlist]

    assert type(tensorlist.at(0)) == np.ndarray
    assert type(tensorlist[0]) != np.ndarray
    assert type(tensorlist[0]) == TensorCPU
    assert type(tensorlist[-3]) == TensorCPU
    assert len(list_of_tensors) == len(tensorlist)
    with assert_raises(IndexError, glob="out of range"):
        tensorlist[len(tensorlist)]
    with assert_raises(IndexError, glob="out of range"):
        tensorlist[-len(tensorlist) - 1]


def test_data_ptr_tensor_cpu():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr, "NHWC")
    from_tensor = py_buffer_from_address(tensor.data_ptr(), tensor.shape(),
                                         types.to_numpy_type(tensor.dtype))
    assert np.array_equal(arr, from_tensor)


def test_data_ptr_tensor_list_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    tensor = tensorlist.as_tensor()
    from_tensor_list = py_buffer_from_address(tensorlist.data_ptr(), tensor.shape(),
                                              types.to_numpy_type(tensor.dtype))
    assert (np.array_equal(arr, from_tensor_list))


def test_array_interface_tensor_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    assert tensorlist[0].__array_interface__['data'][0] == tensorlist[0].data_ptr()
    assert tensorlist[0].__array_interface__['data'][1]
    assert np.array_equal(tensorlist[0].__array_interface__['shape'], tensorlist[0].shape())
    assert np.dtype(tensorlist[0].__array_interface__['typestr']) == np.dtype(
        types.to_numpy_type(tensorlist[0].dtype))


def check_transfer(dali_type):
    arr = np.random.rand(3, 5, 6)
    data = dali_type(arr)
    data_gpu = data._as_gpu()
    data_cpu = data_gpu.as_cpu()
    if dali_type is TensorListCPU:
        np.testing.assert_array_equal(arr, data_cpu.as_array())
    else:
        np.testing.assert_array_equal(arr, np.array(data_cpu))


def test_transfer_cpu_gpu():
    for dali_type in [TensorCPU, TensorListCPU]:
        yield check_transfer, dali_type


def check_array_types(t):
    arr = np.array([[-0.39, 1.5], [-1.5, 0.33]], dtype=t)
    tensor = TensorCPU(arr, "NHWC")
    assert np.allclose(np.array(arr), np.asanyarray(tensor))


def test_array_interface_types():
    for t in [np.bool_, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
              np.uint8, np.uint16, np.uint32, np.uint64, np.float_, np.float32, np.float16,
              np.short, np.long, np.longlong, np.ushort, np.ulonglong]:
        yield check_array_types, t


# TODO(spanev): figure out which return_value_policy to choose
# def test_tensorlist_getitem_slice():
#    arr = np.random.rand(3, 5, 6)
#    tensorlist = TensorListCPU(arr, "NHWC")
#    two_first_tensors = tensorlist[0:2]
#    assert(type(two_first_tensors) == tuple)
#    assert(type(two_first_tensors[0]) == TensorCPU)


def test_tensor_cpu_squeeze():
    def check_squeeze(shape, dim, in_layout, expected_out_layout):
        arr = np.random.rand(*shape)
        t = TensorCPU(arr, in_layout)
        is_squeezed = t.squeeze(dim)
        should_squeeze = (len(expected_out_layout) < len(in_layout))
        arr_squeeze = arr.squeeze(dim)
        t_shape = tuple(t.shape())
        assert t_shape == arr_squeeze.shape, f"{t_shape} != {arr_squeeze.shape}"
        assert t.layout() == expected_out_layout, f"{t.layout()} != {expected_out_layout}"
        assert np.allclose(arr_squeeze, np.array(t))
        assert is_squeezed == should_squeeze, f"{is_squeezed} != {should_squeeze}"

    for dim, shape, in_layout, expected_out_layout in [
        (None, (3, 5, 6), "ABC", "ABC"),
        (None, (3, 1, 6), "ABC", "AC"),
        (1, (3, 1, 6), "ABC", "AC"),
        (-2, (3, 1, 6), "ABC", "AC"),
        (None, (1, 1, 6), "ABC", "C"),
        (1, (1, 1, 6), "ABC", "AC"),
        (None, (1, 1, 1), "ABC", ""),
        (None, (1, 5, 1), "ABC", "B"),
        (-1, (1, 5, 1), "ABC", "AB"),
        (0, (1, 5, 1), "ABC", "BC"),
        (None, (3, 5, 1), "ABC", "AB")
    ]:
        yield check_squeeze, shape, dim, in_layout, expected_out_layout


def test_tensorlist_shape():
    shapes = [(3, 4, 5, 6), (1, 8, 7, 6, 5), (1,), (1, 1)]
    for shape in shapes:
        arr = np.empty(shape)
        tl = TensorListCPU(arr)
        tl_gpu = tl._as_gpu()
        assert tl.shape() == [shape[1:]] * shape[0]
        assert tl_gpu.shape() == [shape[1:]] * shape[0]


def test_tl_from_list_of_tensors_same_shape():
    for shape in [(10, 1), (4, 5, 6), (13, 1), (1, 1)]:
        arr = np.random.rand(*shape)

        tl_cpu_from_np = TensorListCPU(arr)
        tl_cpu_from_tensors = TensorListCPU([TensorCPU(a) for a in arr])
        np.testing.assert_array_equal(tl_cpu_from_np.as_array(), tl_cpu_from_tensors.as_array())

        tl_gpu_from_np = tl_cpu_from_np._as_gpu()
        tl_gpu_from_tensors = TensorListGPU([TensorCPU(a)._as_gpu() for a in arr])
        np.testing.assert_array_equal(tl_gpu_from_np.as_cpu().as_array(),
                                      tl_gpu_from_tensors.as_cpu().as_array())


def test_tl_from_list_of_tensors_different_shapes():
    shapes = [(1, 2, 3), (4, 5, 6), (128, 128, 128), (8, 8, 8), (13, 47, 131)]
    for size in [10, 5, 36, 1]:
        np_arrays = [np.random.rand(*shapes[i])
                     for i in np.random.choice(range(len(shapes)), size=size)]

        tl_cpu = TensorListCPU([TensorCPU(a) for a in np_arrays])
        tl_gpu = TensorListGPU([TensorCPU(a)._as_gpu() for a in np_arrays])

        for arr, tensor_cpu, tensor_gpu in zip(np_arrays, tl_cpu, tl_gpu):
            np.testing.assert_array_equal(arr, tensor_cpu)
            np.testing.assert_array_equal(arr, tensor_gpu.as_cpu())


def test_tl_from_list_of_tensors_empty():
    with assert_raises(RuntimeError, glob='Cannot create TensorList from an empty list.'):
        TensorListCPU([])
    with assert_raises(RuntimeError, glob='Cannot create TensorList from an empty list.'):
        TensorListGPU([])


def test_tl_from_list_of_tensors_different_backends():
    t1 = TensorCPU(np.zeros((1)))
    t2 = TensorCPU(np.zeros((1)))._as_gpu()
    with assert_raises(TypeError, glob='Object at position 1 cannot be converted to TensorCPU'):
        TensorListCPU([t1, t2])
    with assert_raises(TypeError, glob='Object at position 1 cannot be converted to TensorGPU'):
        TensorListGPU([t2, t1])


def test_tl_from_list_of_tensors_different_dtypes():
    np_types = [np.float32, np.float16, np.int16, np.int8, np.uint16, np.uint8]
    for dtypes in np.random.choice(np_types, size=(3, 2), replace=False):
        t1 = TensorCPU(np.zeros((1), dtype=dtypes[0]))
        t2 = TensorCPU(np.zeros((1), dtype=dtypes[1]))
        with assert_raises(TypeError,
                           glob="Tensors cannot have different data types. Tensor at position 1 has type '*' expected to have type '*'."):  # noqa: E501
            TensorListCPU([t1, t2])


def test_dtype_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TensorCPU(np.empty((0))).dtype()
        assert "Calling '.dtype()' is deprecated, please use '.dtype' instead" == str(w[-1].message)


def test_dtype_placeholder_equivalence():
    dali_types = types._all_types
    np_types = list(map(dali_type_to_np, dali_types))

    for dali_type, np_type in zip(dali_types, np_types):
        assert TensorCPU(np.zeros((1), dtype=np_type)).dtype == dali_type


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def dtype_pipeline(np_type, placeholder_dali_type):
    res = fn.external_source(source=np.zeros((8, 1), dtype=np_type), dtype=placeholder_dali_type)

    return res


def test_dtype_converion():
    dali_types = [types_._DALIDataType.INT8,
                  types_._DALIDataType.UINT64,
                  types_._DALIDataType.FLOAT16]
    np_types = list(map(dali_type_to_np, dali_types))
    for dali_type, np_type in zip(dali_types, np_types):
        pipe = dtype_pipeline(np_type, dali_type)
        pipe.build()
        assert pipe.run()[0].dtype == dali_type


def test_tensorlist_dtype():
    dali_types = types._all_types
    np_types = list(map(dali_type_to_np, dali_types))

    for dali_type, np_type in zip(dali_types, np_types):
        tl = TensorListCPU([TensorCPU(np.zeros((1), dtype=np_type))])

        assert tl.dtype == dali_type
        assert tl._as_gpu().dtype == dali_type


def _expected_tensorlist_str(device, data, dtype, num_samples, shape, layout=None):
    return '\n    '.join([f'TensorList{device.upper()}(', f'{data},', f'dtype={dtype},'] +
                         ([f'layout={layout}'] if layout is not None else []) +
                         [f'num_samples={num_samples},', f'shape={shape})'])


def _expected_tensor_str(device, data, dtype, shape, layout=None):
    return '\n    '.join([f'Tensor{device.upper()}(', f'{data},', f'dtype={dtype},'] +
                         ([f'layout={layout}'] if layout is not None else []) +
                         [f'shape={shape})'])


def _test_str(tl, expected_params, expected_func):
    assert str(tl) == expected_func('cpu', *expected_params)
    assert str(tl._as_gpu()) == expected_func('gpu', *expected_params)


def test_tensorlist_str_empty():
    tl = TensorListCPU(np.empty(0))
    params = [[], 'DALIDataType.FLOAT64', 0, []]
    _test_str(tl, params, _expected_tensorlist_str)


def test_tensorlist_str_scalars():
    arr = np.arange(10)
    tl = TensorListCPU(arr)
    params = [arr, 'DALIDataType.INT64', 10, '[(), (), (), (), (), (), (), (), (), ()]']
    _test_str(tl, params, _expected_tensorlist_str)


def test_tensor_str_empty():
    t = TensorCPU(np.empty(0))
    params = [[], 'DALIDataType.FLOAT64', [0]]
    _test_str(t, params, _expected_tensor_str)


def test_tensor_str_sample():
    arr = np.arange(16)
    t = TensorCPU(arr)
    params = [arr, 'DALIDataType.INT64', [16]]
    _test_str(t, params, _expected_tensor_str)
