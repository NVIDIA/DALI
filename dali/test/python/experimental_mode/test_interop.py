import warnings

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
from nose_utils import SkipTest, assert_raises, attr
from packaging import version


def test_numpy_dlpack():
    ndd_tensor = ndd.tensor([1, 2, 3], device="cpu")
    np_array = np.from_dlpack(ndd_tensor)

    np.testing.assert_array_equal(np_array, [1, 2, 3])


def test_numpy_array_interface():
    ndd_tensor = ndd.tensor([1, 2, 3], device="cpu")

    # NumPy will raise a warning if we use the old __array__ signature (without copy and dtype)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")

        np_array = np.array(ndd_tensor)
        assert len(record) == 0

    np.testing.assert_array_equal(np_array, [1, 2, 3])


def test_numpy_nocopy():
    ndd_tensor = ndd.tensor([1, 2, 3], device="cpu")
    np_array1 = np.asarray(ndd_tensor)  # no copy
    np_array2 = np.array(ndd_tensor)  # copy

    np_array1[0] = 42
    np_array2[1] = 0
    result = np.array(ndd_tensor)

    np.testing.assert_array_equal(result, [42, 2, 3])


def test_cuda_array_interface_exists():
    cpu_tensor = ndd.tensor([1, 2, 3], device="cpu")
    gpu_tensor = ndd.tensor([1, 2, 3], device="gpu")

    assert not hasattr(cpu_tensor, "__cuda_array_interface__")
    assert hasattr(gpu_tensor, "__cuda_array_interface__")


@attr("pytorch")
def test_torch_dlpack():
    import torch

    ndd_tensor = ndd.tensor([1, 2, 3], device="cpu")
    torch_tensor = torch.from_dlpack(ndd_tensor)

    np.testing.assert_array_equal(torch_tensor, [1, 2, 3])


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_torch_nocopy(device: str):
    import torch

    if version.parse(torch.__version__) < version.parse("2.6.0"):
        raise SkipTest("Requires PyTorch >= 2.6.0")

    ndd_tensor = ndd.tensor([1, 2, 3], device=device)
    torch_nocopy = torch.as_tensor(ndd_tensor)
    torch_copy = torch.tensor(ndd_tensor)

    torch_nocopy[0] = 42
    torch_copy[1] = 0

    np.testing.assert_array_equal(ndd_tensor.cpu(), [42, 2, 3])


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_tensor_to_torch(device: str):
    ndd_tensor = ndd.ones(shape=3).to_device(device)
    torch_nocopy = ndd_tensor.torch()
    torch_copy = ndd_tensor.torch(copy=True)

    expected_device = "cuda" if device == "gpu" else "cpu"
    assert torch_nocopy.device.type == torch_copy.device.type == expected_device
    torch_nocopy[0] = 42
    torch_copy[1] = 0

    np.testing.assert_array_equal(ndd_tensor.cpu(), [42, 1, 1])


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_batch_to_torch(device: str):
    ndd_batch = ndd.batch(np.ones((2, 2)), device=device)
    torch_nocopy = ndd_batch.torch()
    torch_copy = ndd_batch.torch(copy=True)

    expected_device = "cuda" if device == "gpu" else "cpu"
    assert torch_nocopy.device.type == torch_copy.device.type == expected_device
    torch_nocopy[0, 0] = 42
    torch_copy[1, 1] = 0

    np.testing.assert_array_equal(ndd.as_tensor(ndd_batch.cpu()), [[42, 1], [1, 1]])


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_ragged_batch_to_torch(device: str):
    batch = ndd.batch([[1, 2, 3], [4, 5], [6]])
    with assert_raises(ValueError, glob="dense"):
        batch.torch()


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_ragged_batch_to_torch(device: str):
    batch = ndd.batch([[1, 2, 3], [4, 5], [6]])
    t = batch.torch(pad=True)
    np.testing.assert_array_equal(t, [[1, 2, 3], [4, 5, 0], [6, 0, 0]])
