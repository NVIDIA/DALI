# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.backend_impl as _b

from nvidia.dali.backend import Stream
from nvidia.dali.types import _raw_cuda_stream  # noqa: F401

_global_streams = None


def _get_stream_device(s):
    if s is None:
        return None
    dev_id = getattr(s, "device_id", None)
    if dev_id is not None:
        return dev_id
    dev = getattr(s, "device", None)
    if dev is not None:
        if isinstance(dev, int):
            return dev
        idx = getattr(dev, "index", None)
        if idx is not None:
            return idx
        id = getattr(dev, "id", None)
        if id is not None:
            return id
    return _b.GetCUDAStreamDevice(s)


def set_stream(stream, device_id=None):
    """
    Sets the default stream for a CUDA device. If the device id is not specified,
    the device associated with the stream will be used. If there's no device associated with the
    stream, current CUDA device is used.
    """
    global _global_streams
    if not _global_streams:
        _global_streams = [None] * _b.GetCUDADeviceCount()
    if stream is not None:
        stream_dev = _get_stream_device(stream)
        if device_id is not None and stream_dev is not None and device_id != stream_dev:
            raise ValueError(
                f"The device_id {device_id} doesn't match the device associated "
                f"with the stream {stream_dev}"
            )
        if device_id is None:
            device_id = stream_dev

    if device_id is None:
        device_id = _b.GetCUDACurrentDevice()

    _global_streams[device_id] = stream


def get_stream(device_id=None):
    if _global_streams is None:
        return None
    if device_id is None:
        device_id = _b.GetCUDACurrentDevice()
    return _global_streams[device_id]


__all__ = [
    "Stream",
    "get_stream",
    "set_stream",
]
