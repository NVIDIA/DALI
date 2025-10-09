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

import nvidia.dali.backend as _backend
from threading import local
from typing import Union, Optional

class Device:
    _thread_local = local()

    def __init__(self, name: str, device_id: int = None):
        device_type, name_device_id = Device.split_device_type_and_id(name)
        if name_device_id is not None and device_id is not None:
            raise ValueError(f"Invalid device name: {name}\n"
                             f"Ordinal ':{name_device_id}' should not appear "
                             "in device name when device_id is provided")
        if device_id is None:
            device_id = name_device_id

        if device_type == "cuda":
            device_type = "gpu"

        Device.validate_device_type(device_type)
        if device_id is not None:
            Device.validate_device_id(device_id, device_type)
        else:
            device_id = Device.default_device_id(device_type)

        self.device_type = device_type
        self.device_id = device_id

    @staticmethod
    def split_device_type_and_id(name: str) -> tuple[str, int]:
        type_and_id = name.split(":")
        if len(type_and_id) < 1 or len(type_and_id) > 2:
            raise ValueError(f"Invalid device name: {name}")
        device_type = type_and_id[0]
        device_id = None
        if len(type_and_id) == 2:
            device_id = int(type_and_id[1])
        return device_type, device_id

    @staticmethod
    def default_device_id(device_type: str) -> int:
        if device_type == "cpu":
            return 0
        elif device_type == "gpu" or device_type == "mixed":  # TODO(michalz): Remove mixed
            return _backend.GetCUDACurrentDevice()
        else:
            raise ValueError(f"Invalid device type: {device_type}")

    @staticmethod
    def validate_device_id(device_id: int, device_type: str):
        if device_id < 0:
            raise ValueError(f"Invalid device id: {device_id}")
        if device_type == "gpu" or device_type == "mixed":  # TODO(michalz): Remove mixed
            if device_id >= _backend.GetCUDADeviceCount():
                raise ValueError(f"Invalid device id: {device_id} for device type: {device_type}")
        elif device_type == "cpu":
            if device_id is not None and device_id != 0:
                raise ValueError(f"Invalid device id: {device_id} for device type: {device_type}")

    @staticmethod
    def validate_device_type(device_type: str):
        if device_type not in ["cpu", "gpu", "mixed"]:  # TODO(michalz): Remove mixed
            raise ValueError(f"Invalid device type: {device_type}")

    @staticmethod
    def type_from_dlpack(dev_type) -> str:
        dev_type_id = int(dev_type)
        if dev_type_id == 1:
            return "cpu"
        elif dev_type_id == 2:
            return "gpu"
        else:
            raise ValueError(f"Unsupported device type: {dev_type}")

    @staticmethod
    def from_dlpack(dlpack_device) -> "Device":
        dev_type, dev_id = dlpack_device.__dlpack_device__()
        return Device(Device.type_from_dlpack(dev_type), dev_id)

    def __str__(self):
        return f"{self.device_type}:{self.device_id}"

    def __repr__(self):
        return f"Device(device_type={self.device_type}, device_id={self.device_id})"

    def __eq__(self, other):
        return self.device_type == other.device_type and self.device_id == other.device_id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.device_type, self.device_id))

    @staticmethod
    def current():
        if not Device._thread_local.devices:
            return Device("gpu")
        return Device._thread_local.devices[-1]

    def __enter__(self):
        """Sets the device as current and stores the previous device on stack.

        If the device is GPU, then it sets the current CUDA device to the one identified
        by device_id. If the device is CPU, then it does nothing.
        """
        if self.device_type == "gpu" or self.device_type == "mixed":  # TODO(michalz): Remove mixed
            if Device._thread_local.previous_device_ids is None:
                Device._thread_local.previous_device_ids = []
            Device._thread_local.previous_device_ids.append(_backend.GetCUDACurrentDevice())
            _backend.SetCUDACurrentDevice(self.device_id)
        if Device._thread_local.devices is None:
            Device._thread_local.devices = []
        Device._thread_local.devices.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        """Pops the device from the stack ands it as current.

        If the device popped is GPU, then it sets the current CUDA device to the one identified
        by device_id. If the device is CPU, then it does nothing.
        """
        if self.device_type == "gpu" or self.device_type == "mixed":  # TODO(michalz): Remove mixed
            _backend.SetCUDACurrentDevice(Device._thread_local.previous_device_ids.pop())
        Device._thread_local.devices.pop()
        dev = Device.current()
        if dev.device_type == "gpu":
            _backend.SetCUDACurrentDevice(dev.device_id)


Device._thread_local.devices = None
Device._thread_local.previous_device_ids = None


def device(obj: Union[Device, str, "torch.device"], id: Optional[int] = None) -> Device:
    """
    Returns a Device object from various input types.

    - If `obj` is already a `Device`, returns it. In this case, `id` must be `None`.
    - If `obj` is a `str`, parses it as a device name (e.g., `"gpu"`, `"cpu:0"`, `"cuda:1"`). In this case, `id` can be specified.
      Note: If the string already contains a device id and `id` is also provided, a `ValueError` is raised.
    - If `obj` is a `torch.device`, converts it to a `Device`. In this case, `id` must be `None`.
    - If `obj` is None, returns it.
    - If `obj` is not a `Device`, `str`, or `torch.device` or None, raises a `TypeError`.
    """

    # None
    if obj is None:
        return obj

    # Device instance
    if isinstance(obj, Device):
        if id is not None:
            raise ValueError("Cannot specify id when passing a Device instance")
        return obj

    if isinstance(obj, str):
        return Device(obj, id)

    # torch.device detected by duck-typing
    is_torch_device = (
        obj.__class__.__module__ == "torch" and
        obj.__class__.__name__ == "device" and
        hasattr(obj, "type") and
        hasattr(obj, "index"))
    if is_torch_device:
        dev_type = "gpu" if obj.type == "cuda" else obj.type
        if id is not None:
            raise ValueError("Cannot specify id when passing a torch.device")
        return Device(dev_type, obj.index)

    raise TypeError(f"Cannot convert {type(obj)} to Device")
