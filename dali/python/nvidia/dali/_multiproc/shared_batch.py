# Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import struct
from enum import IntEnum

from nvidia.dali._multiproc import shared_mem
from nvidia.dali._multiproc.messages import (
    CompletedTask,
    SampleRange,
    ScheduledTask,
    ShmMessageDesc,
    TaskArgs,
)
from nvidia.dali._utils.external_source_impl import (
    assert_cpu_sample_data_type as _assert_cpu_sample_data_type,
    sample_to_numpy as _sample_to_numpy,
)
from nvidia.dali.types import BatchInfo

np = None


def _div_ceil(a, b):
    """Calculate ceil of a/b without decaying to float."""
    return -(-a // b)


def _align_up(x, alignment):
    """Align x up to multiple of alignment"""
    return _div_ceil(x, alignment) * alignment


def import_numpy():
    global np
    if np is None:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "Could not import numpy. Please make sure you have numpy "
                "installed before you use parallel mode."
            )


_sample_error_msg = (
    "Unsupported callback return type. Expected NumPy array, PyTorch or MXNet cpu tensors, "
    "DALI TensorCPU, or list or tuple of them representing sample. Got `{}` instead."
)


class BufShmChunk:
    """Simple wrapper around shared memory chunks. Adds mem_chunk_id used
    to identify chunks in the communication between parent and worker process.
    """

    def __init__(self, shm_chunk_id, capacity, shm_chunk: shared_mem.SharedMem):
        self.shm_chunk_id = shm_chunk_id
        self.capacity = capacity
        self._shm_chunk = shm_chunk

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shm_chunk"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def allocate(cls, shm_chunk_id, initial_chunk_size):
        return cls(
            shm_chunk_id, initial_chunk_size, shared_mem.SharedMem.allocate(initial_chunk_size)
        )

    def open_shm(self, handle):
        # self._shm_chunk should be None only as a result of deserialization of the instance.
        # In that case it is not valid to call other methods until shared memory chunk is restored
        # with open_shm call
        assert self._shm_chunk is None
        try:
            self._shm_chunk = shared_mem.SharedMem.open(handle, self.capacity)
        except:  # noqa: E722
            if handle >= 0:
                os.close(handle)
            raise

    def resize(self, size, trunc=False):
        self._shm_chunk.resize(size, trunc)
        self.capacity = size

    def close(self):
        self._shm_chunk.close()

    def close_handle(self):
        self._shm_chunk.close_handle()

    @property
    def handle(self):
        return self._shm_chunk.handle

    @property
    def buf(self):
        return self._shm_chunk.buf


class SampleMeta:
    """Metadata describing serialized sample in a memory buffer.

    It is passed through memory, stored after sample it describes."""

    def __init__(self, offset, shape, dtype, nbytes):
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.nbytes = nbytes

    @classmethod
    def from_np(cls, offset, np_array):
        return cls(offset, np_array.shape, np_array.dtype, np_array.nbytes)


class SharedBatchMeta:
    """Describes offset within shared memory chunk and size of serialized list of
    `SampleMeta` instances"""

    def __init__(self, meta_offset, meta_size):
        self.meta_offset = meta_offset
        self.meta_size = meta_size

    @classmethod
    def from_writer(cls, writer):
        return cls(writer.data_size, writer.meta_data_size)


# Precompiled little-endian layouts of the shared memory messages.
_U8 = struct.Struct("<B")
_BOOL = struct.Struct("<?")
_U64 = struct.Struct("<Q")
_U64_PAIR = struct.Struct("<QQ")
_I64 = struct.Struct("<q")
_I64_PAIR = struct.Struct("<qq")
_I64_X4 = struct.Struct("<qqqq")
_I64_X6 = struct.Struct("<6q")
# array sample: kind tag, offset, nbytes, dtype code, ndim; followed by ndim u64 extents
_ARRAY_HEADER = struct.Struct("<BQQBI")


class _BinaryWriter:
    """Accumulates little-endian, fixed-layout binary encoding of the multiprocessing messages."""

    def __init__(self):
        self._parts = []

    def pack(self, layout, *values):
        self._parts.append(layout.pack(*values))

    def pack_u64_array(self, values):
        self._parts.append(struct.pack(f"<{len(values)}Q", *values))

    def write_bytes(self, data):
        self.pack(_U64, len(data))
        self._parts.append(bytes(data))

    def write_str(self, value):
        self.write_bytes(value.encode("utf-8", errors="backslashreplace"))

    def getvalue(self):
        return b"".join(self._parts)


class _BinaryReader:
    """Decodes data produced by `_BinaryWriter`, validating that reads stay within the buffer."""

    def __init__(self, buffer):
        self._buf = memoryview(buffer).cast("B")
        self._size = len(self._buf)
        self._offset = 0

    def _advance(self, size):
        offset = self._offset
        end = offset + size
        if end > self._size:
            raise RuntimeError("Malformed shared memory message: unexpected end of data.")
        self._offset = end
        return offset

    def peek_u8(self):
        if self._offset >= self._size:
            raise RuntimeError("Malformed shared memory message: unexpected end of data.")
        return self._buf[self._offset]

    def unpack(self, layout):
        return layout.unpack_from(self._buf, self._advance(layout.size))

    def unpack_u64_array(self, count):
        return struct.unpack_from(f"<{count}Q", self._buf, self._advance(8 * count))

    def read_bytes(self):
        (size,) = self.unpack(_U64)
        offset = self._advance(size)
        return bytes(self._buf[offset : offset + size])

    def read_str(self):
        return self.read_bytes().decode("utf-8", errors="replace")

    def read_tag(self, tag_type):
        (tag,) = self.unpack(_U8)
        member = _TAG_MEMBERS[tag_type].get(tag)
        if member is None:
            raise RuntimeError(
                f"Malformed shared memory message: unexpected tag {tag} ({tag_type.__name__})."
            )
        return member

    def assert_consumed(self):
        if self._offset != self._size:
            raise RuntimeError("Malformed shared memory message: unexpected trailing data.")


class _MessageKind(IntEnum):
    SCHEDULED_TASK = 1
    COMPLETED_TASK = 2


class _TaskKind(IntEnum):
    SAMPLE_RANGE = 0
    BATCH_ARGS = 1


class _BatchArgKind(IntEnum):
    NONE = 0
    INT = 1
    BATCH_INFO = 2


class _ExceptionKind(IntEnum):
    NONE = 0
    STOP_ITERATION = 1
    RUNTIME_ERROR = 2


class _SampleKind(IntEnum):
    ARRAY = 0
    TUPLE = 1
    LIST = 2


# Enum construction is slow compared to the rest of decoding, so tags are looked up directly
_TAG_MEMBERS = {
    tag_type: {member.value: member for member in tag_type}
    for tag_type in (_MessageKind, _TaskKind, _BatchArgKind, _ExceptionKind, _SampleKind)
}

_MAX_SAMPLE_NESTING = 64

# The only sample data types that can be passed; the wire format stores the index in this tuple.
# DALI cannot consume other data types (e.g. object, structured, complex or string ones) anyway.
_SAMPLE_DTYPES = (
    "|b1",
    "|i1",
    "|u1",
    "<i2",
    ">i2",
    "<u2",
    ">u2",
    "<i4",
    ">i4",
    "<u4",
    ">u4",
    "<i8",
    ">i8",
    "<u8",
    ">u8",
    "<f2",
    ">f2",
    "<f4",
    ">f4",
    "<f8",
    ">f8",
)
_code_to_dtype = None
_dtype_to_code = None


def _dtype_codes():
    global _code_to_dtype, _dtype_to_code
    if _code_to_dtype is None:
        import_numpy()
        code_to_dtype = tuple(np.dtype(dtype_str) for dtype_str in _SAMPLE_DTYPES)
        _dtype_to_code = {dtype: code for code, dtype in enumerate(code_to_dtype)}
        _code_to_dtype = code_to_dtype
    return _code_to_dtype, _dtype_to_code


def _write_optional_str(writer, value):
    writer.pack(_BOOL, value is not None)
    if value is not None:
        writer.write_str(value)


def _read_optional_str(reader):
    (present,) = reader.unpack(_BOOL)
    return reader.read_str() if present else None


def _write_batch_args(writer, batch_args):
    batch_args = tuple(batch_args)
    if len(batch_args) == 0:
        writer.pack(_U8, _BatchArgKind.NONE)
        return
    if len(batch_args) != 1:
        raise TypeError(f"Expected at most one batch argument, got {len(batch_args)}.")
    (arg,) = batch_args
    if isinstance(arg, BatchInfo):
        writer.pack(_U8, _BatchArgKind.BATCH_INFO)
        writer.pack(_I64_PAIR, arg.iteration, arg.epoch_idx)
    elif isinstance(arg, int):
        writer.pack(_U8, _BatchArgKind.INT)
        writer.pack(_I64, arg)
    else:
        raise TypeError(f"Unsupported batch argument type: `{type(arg)}`.")


def _read_batch_arg(reader):
    """Returns the batch callback argument, or None if the callback takes no argument."""
    kind = reader.read_tag(_BatchArgKind)
    if kind is _BatchArgKind.NONE:
        return None
    if kind is _BatchArgKind.INT:
        (iteration,) = reader.unpack(_I64)
        return iteration
    return BatchInfo(*reader.unpack(_I64_PAIR))


def _write_scheduled_task(writer, message: ScheduledTask):
    task = message.task
    writer.pack(_U8, _MessageKind.SCHEDULED_TASK)
    writer.pack(
        _I64_X4, message.context_i, message.scheduled_i, message.epoch_start, task.minibatch_i
    )
    if task.is_sample_mode():
        sr = task.sample_range
        writer.pack(_U8, _TaskKind.SAMPLE_RANGE)
        writer.pack(
            _I64_X6,
            sr.sample_start,
            sr.sample_end,
            sr.iteration,
            sr.epoch_idx,
            sr.slice_start,
            sr.slice_end,
        )
    else:
        writer.pack(_U8, _TaskKind.BATCH_ARGS)
        _write_batch_args(writer, task.batch_args)


def _read_scheduled_task(reader):
    context_i, scheduled_i, epoch_start, minibatch_i = reader.unpack(_I64_X4)
    kind = reader.read_tag(_TaskKind)
    if kind is _TaskKind.SAMPLE_RANGE:
        sample_start, sample_end, iteration, epoch_idx, slice_start, slice_end = reader.unpack(
            _I64_X6
        )
        # checked explicitly, as the asserts in `SampleRange` are skipped with `python -O`
        if not 0 <= slice_start <= slice_end <= sample_end - sample_start:
            raise RuntimeError("Malformed shared memory message: invalid sample range.")
        sample_range = SampleRange(
            sample_start,
            sample_end,
            iteration,
            epoch_idx,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        task = TaskArgs(minibatch_i, sample_range=sample_range)
    else:
        batch_arg = _read_batch_arg(reader)
        batch_args = () if batch_arg is None else (batch_arg,)
        task = TaskArgs(minibatch_i, batch_args=batch_args)
    return ScheduledTask(context_i, scheduled_i, epoch_start, task)


def _write_completed_task(writer, message: CompletedTask):
    writer.pack(_U8, _MessageKind.COMPLETED_TASK)
    writer.pack(
        _I64_X4, message.worker_id, message.context_i, message.scheduled_i, message.minibatch_i
    )
    batch_meta = message.batch_meta
    writer.pack(_BOOL, batch_meta is not None)
    if batch_meta is not None:
        writer.pack(_U64_PAIR, batch_meta.meta_offset, batch_meta.meta_size)
    exception = message.exception
    if exception is None:
        writer.pack(_U8, _ExceptionKind.NONE)
    else:
        if isinstance(exception, StopIteration):
            writer.pack(_U8, _ExceptionKind.STOP_ITERATION)
        else:
            writer.pack(_U8, _ExceptionKind.RUNTIME_ERROR)
        writer.write_str(str(exception))
    _write_optional_str(writer, message.traceback_str)


def _read_completed_task(reader):
    worker_id, context_i, scheduled_i, minibatch_i = reader.unpack(_I64_X4)
    (has_batch_meta,) = reader.unpack(_BOOL)
    batch_meta = SharedBatchMeta(*reader.unpack(_U64_PAIR)) if has_batch_meta else None
    exception_kind = reader.read_tag(_ExceptionKind)
    exception = None
    if exception_kind is _ExceptionKind.STOP_ITERATION:
        exception = StopIteration(reader.read_str())
    elif exception_kind is _ExceptionKind.RUNTIME_ERROR:
        exception = RuntimeError(reader.read_str())
    traceback_str = _read_optional_str(reader)
    return CompletedTask(
        worker_id,
        context_i,
        scheduled_i,
        minibatch_i,
        batch_meta=batch_meta,
        exception=exception,
        traceback_str=traceback_str,
    )


_MESSAGE_KINDS = {
    ScheduledTask: (_MessageKind.SCHEDULED_TASK, _read_scheduled_task),
    CompletedTask: (_MessageKind.COMPLETED_TASK, _read_completed_task),
}


def serialize_message(message):
    """Encodes `ScheduledTask` or `CompletedTask` into bytes that can be placed in shared memory."""
    writer = _BinaryWriter()
    if isinstance(message, ScheduledTask):
        _write_scheduled_task(writer, message)
    elif isinstance(message, CompletedTask):
        _write_completed_task(writer, message)
    else:
        raise TypeError(f"Unsupported shared memory message type: `{type(message)}`.")
    return writer.getvalue()


def deserialize_message(buffer, message_type):
    """Decodes a `message_type` (`ScheduledTask` or `CompletedTask`) encoded with
    `serialize_message`. Messages of the other type are rejected."""
    expected_kind, read_message = _MESSAGE_KINDS[message_type]
    reader = _BinaryReader(buffer)
    kind = reader.read_tag(_MessageKind)
    if kind is not expected_kind:
        raise RuntimeError(
            f"Malformed shared memory message: expected {expected_kind.name}, got {kind.name}."
        )
    message = read_message(reader)
    reader.assert_consumed()
    return message


def _write_sample_meta(writer, sample, dtype_to_code, depth=0):
    if depth > _MAX_SAMPLE_NESTING:
        raise TypeError(
            f"Samples nested deeper than {_MAX_SAMPLE_NESTING} levels are not supported."
        )
    if isinstance(sample, SampleMeta):
        dtype_code = dtype_to_code.get(sample.dtype)
        if dtype_code is None:
            raise TypeError(
                f"Unsupported sample data type: `{sample.dtype}`. Supported types are: bool, "
                "int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32 "
                "and float64."
            )
        shape = sample.shape
        writer.pack(
            _ARRAY_HEADER, _SampleKind.ARRAY, sample.offset, sample.nbytes, dtype_code, len(shape)
        )
        writer.pack_u64_array(shape)
    elif isinstance(sample, (tuple, list)):
        kind = _SampleKind.TUPLE if isinstance(sample, tuple) else _SampleKind.LIST
        writer.pack(_U8, kind)
        writer.pack(_U64, len(sample))
        for part in sample:
            _write_sample_meta(writer, part, dtype_to_code, depth + 1)
    else:
        raise TypeError(f"Unsupported sample meta-data type: `{type(sample)}`.")


def _read_sample_meta(reader, code_to_dtype, depth=0):
    if depth > _MAX_SAMPLE_NESTING:
        raise RuntimeError("Malformed shared memory message: sample nesting is too deep.")
    if reader.peek_u8() == _SampleKind.ARRAY:
        _, offset, nbytes, dtype_code, ndim = reader.unpack(_ARRAY_HEADER)
        if dtype_code >= len(code_to_dtype):
            raise RuntimeError("Malformed shared memory message: invalid sample data type.")
        shape = reader.unpack_u64_array(ndim)
        return SampleMeta(offset, shape, code_to_dtype[dtype_code], nbytes)
    kind = reader.read_tag(_SampleKind)
    (num_parts,) = reader.unpack(_U64)
    parts = [_read_sample_meta(reader, code_to_dtype, depth + 1) for _ in range(num_parts)]
    return tuple(parts) if kind is _SampleKind.TUPLE else parts


def serialize_samples_meta(samples_meta):
    """Encodes a list of (nested tuples/lists of) `SampleMeta` instances into bytes."""
    _, dtype_to_code = _dtype_codes()
    writer = _BinaryWriter()
    writer.pack(_U64, len(samples_meta))
    for sample in samples_meta:
        _write_sample_meta(writer, sample, dtype_to_code)
    return writer.getvalue()


def deserialize_samples_meta(buffer):
    """Decodes a list of `SampleMeta` instances encoded with `serialize_samples_meta`."""
    code_to_dtype, _ = _dtype_codes()
    reader = _BinaryReader(buffer)
    (num_samples,) = reader.unpack(_U64)
    samples_meta = [_read_sample_meta(reader, code_to_dtype) for _ in range(num_samples)]
    reader.assert_consumed()
    return samples_meta


def deserialize_sample(buffer: BufShmChunk, sample):
    if isinstance(sample, SampleMeta):
        offset = sample.offset
        assert offset % sample.dtype.itemsize == 0, "Sample offset is misaligned."
        buffer = buffer.buf[offset : offset + sample.nbytes]
        return np.ndarray(sample.shape, dtype=sample.dtype, buffer=buffer)
    if isinstance(
        sample,
        (
            tuple,
            list,
        ),
    ):
        return type(sample)(deserialize_sample(buffer, part) for part in sample)
    return sample


def deserialize_sample_meta(buffer: BufShmChunk, shared_batch_meta: SharedBatchMeta):
    """Helper to deserialize SampleMeta from memory based on SharedBatchMeta."""
    sbm = shared_batch_meta
    if sbm.meta_size == 0:
        return []
    serialized_meta = buffer.buf[sbm.meta_offset : sbm.meta_offset + sbm.meta_size]
    return deserialize_samples_meta(serialized_meta)


def deserialize_batch(buffer: BufShmChunk, shared_batch_meta: SharedBatchMeta):
    """Deserialize samples from the smem buffer and SampleMeta descriptions.

    Parameters
    ----------
    buffer : BufShmChunk
        Shared memory chunk with serialized sample data
    shared_batch_meta : SharedBatchMeta
        Metadata about serialized data in memory

    Returns
    -------
    List of (idx, numpy array) or (idx, tuple of numpy arrays)
        List of indexed deserialized samples
    """
    samples = deserialize_sample_meta(buffer, shared_batch_meta)
    return [deserialize_sample(buffer, sample) for sample in samples]


def assert_valid_data_type(sample):
    """Check if the output of the callback is type that can be serialized"""
    _apply_to_sample(lambda x: _assert_cpu_sample_data_type(x, _sample_error_msg), sample)


def _apply_to_sample(func, sample, *args, nest_with_sample=0):
    """Apply to a sample traversing the nesting of the data (tuple/list).

    Parameters
    ----------
    func : callable
        Function to be applied to every sample data object
    sample : sample object or any nesting of those in tuple/list
        Representation of sample
    nest_with_sample: int
        Specify how many consecutive (additional) arguments have the same level of nesting
        as the sample.
    """
    if isinstance(
        sample,
        (
            tuple,
            list,
        ),
    ):
        # Check that all the samples have common nesting
        for i in range(nest_with_sample):
            assert len(args[i]) == len(sample)
        nest_group = sample, *args[0:nest_with_sample]
        scalar_args = args[nest_with_sample:]
        return type(sample)(
            _apply_to_sample(func, *part, *scalar_args) for part in zip(*nest_group)
        )
    else:
        # we unpacked all nesting levels, now is actual data:
        return func(sample, *args)


class SharedBatchWriter:
    """SharedBatchWriter can serialize and write batch into given shared
    memory chunk (``shm_chunk``).
    """

    SAMPLE_ALIGNMENT = 128
    BUFFER_ALIGNMENT = 4096

    def __init__(self, shm_chunk: BufShmChunk, batch, min_trailing_offset=1024 * 1024):
        import_numpy()
        self.shm_chunk = shm_chunk
        self.data_size = 0
        self.meta_data_size = 0
        self.total_size = 0
        # hint how much space should be left in case of the resize at the end of the shm chunk
        # after batch data to accommodate meta data of the task
        self.min_trailing_offset = min_trailing_offset
        self._write_batch(batch)

    def _prepare_samples_meta(self, samples):
        """Calculate metadata and total size of data to be serialized"""
        data_size = 0

        def make_meta(np_array):
            nonlocal data_size
            offset = _align_up(data_size, self.SAMPLE_ALIGNMENT)
            data_size = offset + np_array.nbytes
            return SampleMeta(offset, np_array.shape, np_array.dtype, np_array.nbytes)

        meta = [_apply_to_sample(make_meta, sample) for sample in samples]
        return meta, data_size

    def _add_array_to_batch(self, np_array, meta, memview):
        sample_size = meta.nbytes
        offset = meta.offset
        buffer = memview[offset : (offset + sample_size)]
        shared_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=buffer)
        shared_array.ravel()[:] = np_array.ravel()[:]

    def _write_batch(self, batch):
        if not batch:
            return
        batch = [
            _apply_to_sample(lambda x: _sample_to_numpy(x, _sample_error_msg), sample)
            for sample in batch
        ]
        meta, data_size = self._prepare_samples_meta(batch)
        serialized_meta = serialize_samples_meta(meta)
        self.meta_data_size = len(serialized_meta)
        self.data_size = _align_up(data_size, self.SAMPLE_ALIGNMENT)
        self.total_size = _align_up(self.data_size + self.meta_data_size, self.SAMPLE_ALIGNMENT)
        if self.shm_chunk.capacity < self.total_size:
            resize_shm_chunk(self.shm_chunk, self.total_size + self.min_trailing_offset)
        memview = self.shm_chunk.buf
        for sample, sample_meta in zip(batch, meta):
            _apply_to_sample(
                self._add_array_to_batch, sample, sample_meta, memview, nest_with_sample=1
            )
        # copy meta data at the end of shared memory chunk
        buffer = memview[self.data_size : (self.data_size + self.meta_data_size)]
        buffer[:] = serialized_meta


def resize_shm_chunk(shm_chunk, needed_capacity):
    new_capacity = max(needed_capacity, 2 * shm_chunk.capacity)
    new_capacity = _align_up(new_capacity, SharedBatchWriter.BUFFER_ALIGNMENT)
    shm_chunk.resize(new_capacity, trunc=True)


def read_shm_message(shm_chunk: BufShmChunk, shm_message, message_type):
    if shm_message.shm_capacity != shm_chunk.capacity:
        shm_chunk.resize(shm_message.shm_capacity, trunc=False)
    buffer = shm_chunk.buf[shm_message.offset : shm_message.offset + shm_message.num_bytes]
    return deserialize_message(buffer, message_type)


def write_shm_message(worker_id, shm_chunk: BufShmChunk, message, offset, resize=True):
    """
    Serializes `message` instance, stores it in the provided `shm` chunk at given offset and returns
    `ShmMessageDesc` instance describing the placement of the `message`.
    Returned instance can be put into ShmQueue.
    """
    serialized_message = serialize_message(message)
    num_bytes = len(serialized_message)
    if num_bytes > shm_chunk.capacity - offset:
        if resize:
            resize_shm_chunk(shm_chunk, offset + num_bytes)
        else:
            # This should not happen, resize is False only when writing task description into memory
            # in the main process, and the description (ScheduledTask and its members) boils down
            # to bounded number of integers.
            raise RuntimeError(
                "Could not put message into shared memory region,"
                " not enough space in the buffer."
            )
    buffer = shm_chunk.buf[offset : offset + num_bytes]
    buffer[:] = serialized_message
    return ShmMessageDesc(worker_id, shm_chunk.shm_chunk_id, shm_chunk.capacity, offset, num_bytes)
