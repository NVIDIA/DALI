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


class Stream:
    create_new = object()

    def __init__(self, *, stream=create_new, device_id=None):
        if stream is None:
            raise ValueError(
                "The stream must not be None. To create a new stream, omit the stream parameter."
            )
        if stream is Stream.create_new:
            if device_id is None:
                device_id = _b.GetCUDACurrentDevice()
            self._device_id = device_id
            self._obj = _b.Stream(device_id)
            self._handle = self._obj.handle
        else:
            if isinstance(stream, Stream):
                self._handle = stream._handle
                self._device_id = stream._device_id
                self._obj = stream._obj
                if device_id is not None and device_id != stream._device_id:
                    raise ValueError(
                        f"The device_id {device_id} doesn't match the id {stream._device_id} "
                        f"associated with the stream object."
                    )

            else:
                self._handle = _raw_cuda_stream(stream)
                dev = _get_stream_device(stream)
                assert dev is not None
                if device_id is not None and device_id != dev:
                    raise ValueError(
                        f"The device_id {device_id} doesn't match the id {dev} "
                        f"inferred from the stream handle {self._handle}."
                    )
                if device_id is None:
                    device_id = dev
                self._device_id = device_id
                self._obj = stream

    @property
    def handle(self):
        return self._handle

    @property
    def device_id(self):
        return self._device_id

    def __cuda_stream__(self):
        return (0, self.handle)

    def __str__(self):
        return f"CUDA stream {self.handle}, device_id={self.device_id}"

    def __repr__(self):
        if isinstance(self._obj, _b.Stream):
            return repr(self._obj)
        return f"Stream(stream={repr(self._obj)}, device_id={self._device_id})"

    def __eq__(self, value):
        s = stream(stream=value)
        return self._handle == s._handle and self._device_id == s._device_id

    def __ne__(self, value):
        return not (self == value)

    # TODO(michalz): add synchronization functions


def stream(*, stream=Stream.create_new, device_id=None):
    if stream is None:
        return None
    if isinstance(stream, Stream):
        if device_id is not None and device_id != stream.device_id:
            raise ValueError(
                f"The device_id {device_id} doesn't match the id {stream.device_id} "
                f"found in the stream object."
            )
        return stream
    else:
        return Stream(stream=stream, device_id=device_id)


def set_default_stream(cuda_stream, /, device_id=None):
    """
    Sets the default stream for a CUDA device. If the device id is not specified,
    the device associated with the stream will be used. If there's no device associated with the
    stream, current CUDA device is used.


    """
    global _global_streams
    if not _global_streams:
        _global_streams = [None] * _b.GetCUDADeviceCount()
    if cuda_stream is not None:
        cuda_stream = stream(stream=cuda_stream, device_id=device_id)
        if device_id is None:
            device_id = cuda_stream.device_id

    if device_id is None:
        device_id = _b.GetCUDACurrentDevice()

    _global_streams[device_id] = cuda_stream


def get_default_stream(device_id=None):
    """Gets the default stream

    This stream is used when not overridden by thread's current stream (see
    :meth:`set_current_stream`) or an active non-default :class:`EvalContext`
    """
    if _global_streams is None:
        return None
    if device_id is None:
        device_id = _b.GetCUDACurrentDevice()
    return _global_streams[device_id]


def set_current_stream(cuda_stream, /):
    """Sets the stream associated with the calling thread's default context for the current device.

    The stream must match the current CUDA device. See :class:`Device`.
    Setting the current stream doesn't establish any synchronization between the work previously
    scheduled and new work.
    """
    from ._eval_context import EvalContext

    ctx = EvalContext.default()
    ctx._cuda_stream = stream(stream=cuda_stream, device_id=ctx.device_id)


def get_current_stream():
    """Gets the stream associated with the calling thread's default context for the current device."""
    from ._eval_context import EvalContext

    return EvalContext.default().cuda_stream


__all__ = [
    "Stream",
    "stream",
    "get_default_stream",
    "set_default_stream",
    "get_current_stream",
    "set_current_stream",
]
