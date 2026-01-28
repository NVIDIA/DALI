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
    """
    Wrapper for a CUDA stream object.

    This class wraps a CUDA stream object. It can be either a stream created by DALI or a
    compatible object created by a third-party library.
    """

    create_new = object()

    def __init__(self, *, stream=create_new, device_id=None):
        """
        Do not construct this class directly. Use :meth:`stream` instead.
        """
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
        """
        A raw CUDA stream handle, returned as an integer.
        """
        return self._handle

    @property
    def device_id(self):
        """
        The CUDA device ordinal associated with this stream.
        """
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
        """
        The stream objects are considered equal if they have the same handle and device.
        The wrapped object doesn't participate in the comparison.
        """
        if value is None:
            return False
        if value is self:
            return True
        try:
            s = stream(stream=value)
        except TypeError:
            return NotImplemented
        return self._handle == s._handle and self._device_id == s._device_id

    def __ne__(self, value):
        return not (self == value)

    # TODO(michalz): add synchronization functions


def stream(*, stream=Stream.create_new, device_id=None):
    """
    Wraps an existing object or creates a new stream.

    This function wraps a compatible stream object with a DALI Stream class or creates a new stream.

    Keyword Args
    ------------
    stream : a compatible stream object, None or `CreateNew` sentinel value, optional
        When this parameter contains a compatible stream object, the function returns a
        :class:`Stream` object wrapping it.
        If the value is not set and contains the ``Stream.create_new`` flag, a new stream on
        the specified device will be returned.
        If ``None`` is passed, the function returns ``None``.
        Compatible objects are:
          - objects exposing ``__cuda_stream__`` interface
          - PyTorch streams
          - raw stream handles
        If `stream` is not a raw handle but rather a stream object created by a third-party library,
        it is referenced by the wrapper object returned by this function, thereby prolonging its
        lifetime.
    device_id : int or None, optional
        If not ``None``, the function will create a new stream on the device specifed or, if
        `stream` contains a stream object, the function will verify that the ``stream`` is on
        the device specified in `device_id`.
        When `stream` is ``None``, this value is ignored.
    """
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

    Passing ``None`` clears the default stream.

    .. warning::
        This function is intended to be used once, at the beginning of the program, to set the
        default stream for DALI operations. Calling it affects all default contexts in all threads
        that haven't set their current streams with a call to :meth:`set_current_stream`.
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
    :meth:`set_current_stream`).
    """
    if _global_streams is None:
        return None
    if device_id is None:
        device_id = _b.GetCUDACurrentDevice()
    return _global_streams[device_id]


def set_current_stream(cuda_stream, /):
    """Sets the stream associated with the calling thread's default context for the current device.

    The stream must match the current CUDA device. See :class:`Device`.

    In addition to changing the stream of the current thread's default context, this method causes
    newly created :class:`EvalContext` objects with the current device to use this stream.

    Passing ``None`` resets the current thread's default context stream. After that, the value
    returned by :meth:`get_current_stream` will either point to the value returned by
    :meth:`get_default_stream` or a new stream.

    .. warning::
        Setting the current stream doesn't establish any synchronization between the work
        previously scheduled and new work.
    """
    from ._eval_context import EvalContext

    ctx = EvalContext.default()
    ctx._cuda_stream = stream(stream=cuda_stream, device_id=ctx.device_id)


def get_current_stream():
    """Gets the stream associated with the calling thread's default context.

    The value returned by this function is equivalent to ``EvalContext.default().cuda_stream``.
    """
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
