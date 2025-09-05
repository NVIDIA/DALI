# Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali.backend_impl import TensorCPU, TensorGPU, TensorListCPU, TensorListGPU, GetSchema
from nvidia.dali.backend_impl import types as types_
import nvidia.dali as dali

from nose2.tools import params
from nose_utils import assert_raises, SkipTest
from test_utils import dali_type_to_np, py_buffer_from_address, get_device_memory_info


def test_preallocation():
    dali.backend.PreallocateDeviceMemory(0, 0)  # initialize the context
    dali.backend.ReleaseUnusedMemory()
    mem_info = get_device_memory_info()
    if mem_info is None:
        raise SkipTest("Python bindings for NVML not found, skipping")
    free_before_prealloc = mem_info.free
    size = 256 << 20
    dali.backend.PreallocateDeviceMemory(size, 0)
    free_after_prealloc = get_device_memory_info().free
    assert free_after_prealloc < free_before_prealloc  # check that something was allocated
    dali.backend.ReleaseUnusedMemory()
    free_after_release = get_device_memory_info().free
    assert free_after_release > free_after_prealloc  # check that something was freed


def test_create_tensor():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr, "NHWC")
    assert_array_equal(arr, np.array(tensor))


def test_create_tensor_and_make_it_release_memory():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr, "NHWC")
    assert_array_equal(arr, np.array(tensor))
    arr = None
    tensor = None


def test_create_tensorlist():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    assert_array_equal(arr, tensorlist.as_array())


def test_create_tensorlist_list():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU([arr], "NHWC")
    assert_array_equal(arr.reshape(tuple([1]) + arr.shape), tensorlist.as_array())


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
    assert np.array(tensor).shape == (0,)
    assert tensorlist.as_array().shape == (0,)


def test_tensorlist_getitem_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    list_of_tensors = [x for x in tensorlist]

    assert type(tensorlist.at(0)) is np.ndarray
    assert type(tensorlist[0]) is not np.ndarray
    assert type(tensorlist[0]) is TensorCPU
    assert type(tensorlist[-3]) is TensorCPU
    assert len(list_of_tensors) == len(tensorlist)
    with assert_raises(IndexError, glob="out of range"):
        tensorlist[len(tensorlist)]
    with assert_raises(IndexError, glob="out of range"):
        tensorlist[-len(tensorlist) - 1]


def test_data_ptr_tensor_cpu():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr, "NHWC")
    from_tensor = py_buffer_from_address(
        tensor.data_ptr(), tensor.shape(), types.to_numpy_type(tensor.dtype)
    )
    assert np.array_equal(arr, from_tensor)


def test_data_ptr_tensor_list_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    tensor = tensorlist.as_tensor()
    from_tensor_list = py_buffer_from_address(
        tensorlist.data_ptr(), tensor.shape(), types.to_numpy_type(tensor.dtype)
    )
    assert np.array_equal(arr, from_tensor_list)


def test_array_interface_tensor_cpu():
    arr = np.random.rand(3, 5, 6)
    tensorlist = TensorListCPU(arr, "NHWC")
    assert tensorlist[0].__array_interface__["data"][0] == tensorlist[0].data_ptr()
    assert not tensorlist[0].__array_interface__["data"][1]
    assert np.array_equal(tensorlist[0].__array_interface__["shape"], tensorlist[0].shape())
    assert np.dtype(tensorlist[0].__array_interface__["typestr"]) == np.dtype(
        types.to_numpy_type(tensorlist[0].dtype)
    )


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
    for t in [
        np.bool_,
        np.int_,
        np.intc,
        np.intp,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float16,
        np.short,
        int,
        np.longlong,
        np.ushort,
        np.ulonglong,
    ]:
        yield check_array_types, t


def layout_compatible(a, b):
    if a is None:
        a = ""
    if b is None:
        b = ""
    return a == b


# TODO(spanev): figure out which return_value_policy to choose
# def test_tensorlist_getitem_slice():
#    arr = np.random.rand(3, 5, 6)
#    tensorlist = TensorListCPU(arr, "NHWC")
#    two_first_tensors = tensorlist[0:2]
#    assert type(two_first_tensors) == tuple
#    assert type(two_first_tensors[0]) == TensorCPU


def test_tensor_cpu_squeeze():
    def check_squeeze(shape, dim, in_layout, expected_out_layout):
        arr = np.random.rand(*shape)
        t = TensorCPU(arr, in_layout)
        is_squeezed = t.squeeze(dim)
        should_squeeze = len(expected_out_layout) < len(in_layout)
        arr_squeeze = arr.squeeze(dim)
        t_shape = tuple(t.shape())
        assert t_shape == arr_squeeze.shape, f"{t_shape} != {arr_squeeze.shape}"
        assert t.layout() == expected_out_layout, f"{t.layout()} != {expected_out_layout}"
        assert layout_compatible(
            t.get_property("layout"), expected_out_layout
        ), f'{t.get_property("layout")} doesn\'t match {expected_out_layout}'
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
        (None, (3, 5, 1), "ABC", "AB"),
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
        np.testing.assert_array_equal(
            tl_gpu_from_np.as_cpu().as_array(), tl_gpu_from_tensors.as_cpu().as_array()
        )


def test_tl_from_list_of_tensors_different_shapes():
    shapes = [(1, 2, 3), (4, 5, 6), (128, 128, 128), (8, 8, 8), (13, 47, 131)]
    for size in [10, 5, 36, 1]:
        np_arrays = [
            np.random.rand(*shapes[i]) for i in np.random.choice(range(len(shapes)), size=size)
        ]

        tl_cpu = TensorListCPU([TensorCPU(a) for a in np_arrays])
        tl_gpu = TensorListGPU([TensorCPU(a)._as_gpu() for a in np_arrays])

        for arr, tensor_cpu, tensor_gpu in zip(np_arrays, tl_cpu, tl_gpu):
            np.testing.assert_array_equal(arr, tensor_cpu)
            np.testing.assert_array_equal(arr, tensor_gpu.as_cpu())


def test_tl_from_list_of_tensors_different_backends():
    t1 = TensorCPU(np.zeros((1)))
    t2 = TensorCPU(np.zeros((1)))._as_gpu()
    with assert_raises(TypeError, glob="Object at position 1 cannot be converted to TensorCPU"):
        TensorListCPU([t1, t2])
    with assert_raises(TypeError, glob="Object at position 1 cannot be converted to TensorGPU"):
        TensorListGPU([t2, t1])


def test_tl_from_list_of_tensors_different_dtypes():
    np_types = [np.float32, np.float16, np.int16, np.int8, np.uint16, np.uint8]
    for dtypes in np.random.choice(np_types, size=(3, 2), replace=False):
        t1 = TensorCPU(np.zeros((1), dtype=dtypes[0]))
        t2 = TensorCPU(np.zeros((1), dtype=dtypes[1]))
        with assert_raises(
            TypeError,
            glob=(
                "Tensors cannot have different data types."
                " Tensor at position 1 has type '*' expected to have type '*'."
            ),
        ):
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
    dali_types = [
        types_._DALIDataType.INT8,
        types_._DALIDataType.UINT64,
        types_._DALIDataType.FLOAT16,
    ]
    np_types = list(map(dali_type_to_np, dali_types))
    for dali_type, np_type in zip(dali_types, np_types):
        pipe = dtype_pipeline(np_type, dali_type)
        assert pipe.run()[0].dtype == dali_type


def test_tensorlist_dtype():
    dali_types = types._all_types
    np_types = list(map(dali_type_to_np, dali_types))

    for dali_type, np_type in zip(dali_types, np_types):
        tl = TensorListCPU([TensorCPU(np.zeros((1), dtype=np_type))])

        assert tl.dtype == dali_type
        assert tl._as_gpu().dtype == dali_type


def _expected_tensorlist_str(device, data, dtype, num_samples, shape, layout=None):
    return "\n    ".join(
        [f"TensorList{device.upper()}(", f"{data},", f"dtype={dtype},"]
        + ([f"layout={layout}"] if layout is not None else [])
        + [f"num_samples={num_samples},", f"shape={shape})"]
    )


def _expected_tensor_str(device, data, dtype, shape, layout=None):
    return "\n    ".join(
        [f"Tensor{device.upper()}(", f"{data},", f"dtype={dtype},"]
        + ([f"layout={layout}"] if layout is not None else [])
        + [f"shape={shape})"]
    )


def _test_str(tl, expected_params, expected_func):
    assert str(tl) == expected_func("cpu", *expected_params)
    assert str(tl._as_gpu()) == expected_func("gpu", *expected_params)


def test_tensorlist_str_empty():
    tl = TensorListCPU(np.empty(0))
    params = [[], "DALIDataType.FLOAT64", 0, []]
    _test_str(tl, params, _expected_tensorlist_str)


def test_tensorlist_str_scalars():
    arr = np.arange(10)
    tl = TensorListCPU(arr)
    params = [arr, "DALIDataType.INT64", 10, "[(), (), (), (), (), (), (), (), (), ()]"]
    _test_str(tl, params, _expected_tensorlist_str)


def test_tensor_str_empty():
    t = TensorCPU(np.empty(0))
    params = [[], "DALIDataType.FLOAT64", [0]]
    _test_str(t, params, _expected_tensor_str)


def test_tensor_str_sample():
    arr = np.arange(16)
    t = TensorCPU(arr)
    params = [arr, "DALIDataType.INT64", [16]]
    _test_str(t, params, _expected_tensor_str)


def test_tensor_dlpack_export():
    # TODO(awolant): Numpy versions for Python 3.6 and 3.7 do not
    # support from_dlpack. When we upgrade DLPack support for DALI
    # this test needs to be changed.
    if not hasattr(np, "from_dlpack"):
        raise SkipTest("Test requires Numpy DLPack support.")

    arr = np.arange(20)
    tensor = TensorCPU(arr, "NHWC")

    arr_from_dlapck = np.from_dlpack(tensor)

    assert np.array_equal(arr, arr_from_dlapck)


def test_tensor_from_numpy_dlpack():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t = TensorCPU(a.__dlpack__())
    assert tuple(t.shape()) == tuple(a.shape)
    assert a.ctypes.data == t.data_ptr()


@params((TensorCPU,), (TensorGPU,))
def test_dlpack_reimport(tensor_type):
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t = TensorCPU(a)
    if tensor_type is TensorGPU:
        t = t._as_gpu()
    t2 = tensor_type(t.__dlpack__())

    assert t.shape() == t2.shape()
    assert t.data_ptr() == t2.data_ptr()
    assert t.dtype == t2.dtype


def test_schema_is_stateful():
    def get_schema(fn):
        return GetSchema(fn._schema_name)

    # Special operators
    assert get_schema(fn.external_source).IsStateful()
    assert get_schema(fn.python_function).IsStateful()
    # Random number generators
    assert get_schema(fn.random.uniform).IsStateful()
    assert get_schema(fn.random.normal).IsStateful()
    assert get_schema(fn.random.coin_flip).IsStateful()
    # Readers
    assert get_schema(fn.readers.file).IsStateful()
    assert get_schema(fn.readers.tfrecord).IsStateful()
    # Generic processing operators
    assert not get_schema(fn.resize).IsStateful()
    assert not get_schema(fn.tensor_subscript).IsStateful()
    assert not get_schema(fn.slice).IsStateful()
    assert not get_schema(fn.decoders.image).IsStateful()


def test_schema_get_input_device():
    # Slice's input 0 always matches the backend, but the other inputs can be on CPU
    # even if the operator is on GPU
    schema = GetSchema("Slice")
    # if the operator's backend is not known, the MatchBackend input's device is not known
    assert schema.GetInputDevice(0, None, None) is None
    assert schema.GetInputDevice(0, "cpu", None) is None
    assert schema.GetInputDevice(0, "gpu", None) is None
    assert schema.GetInputDevice(0, "cpu", "cpu") == "cpu"
    assert schema.GetInputDevice(0, "gpu", "gpu") == "gpu"
    assert schema.GetInputDevice(0, "cpu", "gpu") == "gpu"
    assert schema.GetInputDevice(0, "gpu", "cpu") == "cpu"
    assert schema.GetInputDevice(0, None, "cpu") == "cpu"
    assert schema.GetInputDevice(0, None, "gpu") == "gpu"

    assert schema.GetInputDevice(1, None, None) is None
    assert schema.GetInputDevice(1, "cpu", None) == "cpu"
    assert schema.GetInputDevice(1, "gpu", None) == "gpu"
    assert schema.GetInputDevice(1, "cpu", "cpu") == "cpu"
    assert schema.GetInputDevice(1, "gpu", "gpu") == "gpu"
    # GPU op, CPU input -> no conversion
    assert schema.GetInputDevice(1, "cpu", "gpu") == "cpu"
    # CPU op, GPU input -> need to copy back to CPU
    assert schema.GetInputDevice(1, "gpu", "cpu") == "cpu"
    assert schema.GetInputDevice(1, None, "cpu") == "cpu"
    assert schema.GetInputDevice(1, None, "gpu") == "gpu"

    # metadata op
    schema = GetSchema("Shapes")
    assert schema.GetInputDevice(0, None, None) is None
    for input_device in [None, "cpu", "gpu"]:
        for operator_device in [None, "cpu", "gpu"]:
            # Metadata operators can take whatever input is given to them, no conversion is
            # required.
            # If the input's actual device is not known, the operator's device is used.
            expected_device = input_device or operator_device
            dev = schema.GetInputDevice(0, input_device, operator_device)
            assert dev == expected_device, (
                f"{dev} != {expected_device}, "
                f"input_device: {input_device}, "
                f"operator_device: {operator_device}"
            )

    # op with input device "Any"
    schema = GetSchema("Copy")
    assert schema.GetInputDevice(0, None, None) is None
    for input_device in [None, "cpu", "gpu"]:
        for operator_device in [None, "cpu", "gpu"]:
            # Copy can take any input, no conversion required (it _does_ the conversion).
            # If the input's actual device is not known, the operator's device is used.
            expected_device = input_device or operator_device
            dev = schema.GetInputDevice(0, input_device, operator_device)
            assert dev == expected_device, (
                f"{dev} != {expected_device}, "
                f"input_device: {input_device}, "
                f"operator_device: {operator_device}"
            )


@params((TensorCPU,), (TensorGPU,))
def test_reinterpret_tensor(TensorType):
    t = TensorCPU(np.array([0x3F800000, 0x3FC00000, 0x40000000], np.int32))
    if TensorType is TensorGPU:
        t = t._as_gpu()
    t.reinterpret(types.UINT32)
    assert t.dtype == types.UINT32
    assert np.array_equal(np.array(t.as_cpu()), np.uint32([0x3F800000, 0x3FC00000, 0x40000000]))
    t.reinterpret(types.FLOAT)
    assert t.dtype == types.FLOAT
    assert np.array_equal(np.array(t.as_cpu()), np.float32([1.0, 1.5, 2.0]))
    with assert_raises(Exception, glob="*different*size*"):
        t.reinterpret(types.UINT16)
    with assert_raises(Exception, glob="*different*size*"):
        t.reinterpret(types.FLOAT64)


@params((TensorListCPU,), (TensorListGPU,))
def test_reinterpret_tensor_list(TensorListType):
    t = TensorListCPU(np.array([[1, 2, 3], [5, 6, 7]], np.int32))
    if TensorListType is TensorListGPU:
        t = t._as_gpu()
    t.reinterpret(types.UINT32)
    assert t.dtype == types.UINT32
    t.reinterpret(types.FLOAT)
    assert t.dtype == types.FLOAT
    with assert_raises(Exception, glob="*different*size*"):
        t.reinterpret(types.UINT16)
    with assert_raises(Exception, glob="*different*size*"):
        t.reinterpret(types.FLOAT64)
