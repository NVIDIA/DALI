import warnings

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from nose2.tools import params
from nose_utils import attr


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

    ndd_tensor = ndd.tensor([1, 2, 3], device=device)
    torch_tensor1 = torch.as_tensor(ndd_tensor)  # no copy
    torch_tensor2 = torch.tensor(ndd_tensor)  # copy

    torch_tensor1[0] = 42
    torch_tensor2[1] = 0
    result = torch.tensor(ndd_tensor, device="cpu")

    np.testing.assert_array_equal(result, [42, 2, 3])
