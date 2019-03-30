from nvidia.dali.backend_impl import *
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

def test_create_tensor():
    arr = np.random.rand(3, 5, 6)
    tensor = TensorCPU(arr)
    assert_array_equal(arr, np.array(tensor))

def test_create_tensor_list():
    arr = np.random.rand(3, 5, 6)
    tensor_list = TensorListCPU(arr)
    assert_array_equal(arr, tensor_list.as_array())

def test_create_tensor_list_as_tensor():
    arr = np.random.rand(3, 5, 6)
    tensor_list = TensorListCPU(arr)
    tensor = tensor_list.as_tensor()
    assert_array_equal(np.array(tensor), tensor_list.as_array())

def test_empty_tensor_tensor_list():
    arr = np.array([], dtype=np.float32)
    tensor = TensorCPU(arr)
    tensor_list = TensorListCPU(arr)
    assert_array_equal(np.array(tensor), tensor_list.as_array())
    assert(np.array(tensor).shape == (0,))
    assert(tensor_list.as_array().shape == (0,))