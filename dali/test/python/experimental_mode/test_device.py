# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dali2 as dali2
import nvidia.dali.backend as _backend
from nose_utils import SkipTest, assert_raises, attr


def test_default_device():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")
    assert dali2.Device.current() == dali2.Device("gpu")


def test_device():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")
    with dali2.Device("gpu"):
        assert dali2.Device.current().device_type == "gpu"
        assert dali2.Device.current().device_id == 0
    assert dali2.Device.current().device_type == "gpu"
    assert dali2.Device.current().device_id == 0


def test_device_with_id():
    other_device_id = 1 if _backend.GetCUDADeviceCount() > 1 else 0
    if other_device_id == 0:
        print("Warning: Only 1 GPU detected, weak test")
    with dali2.Device("gpu", other_device_id):
        assert dali2.Device.current().device_type == "gpu"
        assert dali2.Device.current().device_id == other_device_id
        with dali2.Device("gpu", 0):
            assert dali2.Device.current().device_type == "gpu"
            assert dali2.Device.current().device_id == 0
        assert dali2.Device.current().device_type == "gpu"
        assert dali2.Device.current().device_id == other_device_id
    assert dali2.Device.current().device_type == "gpu"
    assert dali2.Device.current().device_id == 0


def test_device_parse():
    if _backend.GetCUDADeviceCount() == 0:
        raise SkipTest("At least 1 device needed for the test")
    assert dali2.Device("gpu:0") == dali2.Device("gpu")
    assert dali2.Device("cpu") == dali2.Device("cpu:0")
    with assert_raises(ValueError, glob="Invalid device name"):
        dali2.Device("gpu:0", 0)  # double id specification
    with assert_raises(ValueError, glob="Invalid device id"):
        dali2.Device("cpu:99")  # invalid id
    with assert_raises(ValueError, glob="Invalid device id"):
        dali2.Device("gpu:10000000000")  # CUDA device id is 32-bit, so 10B is surely invalid


@attr("multi_gpu")
def test_device_parse_multi_gpu():
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    assert dali2.Device("gpu:1") == dali2.Device("gpu", 1)


@attr("pytorch")
@attr("multi_gpu")
def test_device_same_as_in_torch_multi_gpu():
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    import torch

    torch.cuda.set_device(1)
    assert dali2.Device("gpu").device_id == 1

    torch.cuda.set_device(0)
    assert dali2.Device("gpu").device_id == 0

    with dali2.Device("gpu", 1):
        assert torch.cuda.current_device() == 1

    assert torch.cuda.current_device() == 0
