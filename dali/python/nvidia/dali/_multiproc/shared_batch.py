# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali._multiproc import shared_mem
from nvidia.dali._utils.external_source_impl import \
        assert_cpu_sample_data_type as _assert_cpu_sample_data_type, \
        sample_to_numpy as _sample_to_numpy
import pickle


np = None


def _div_ceil(a, b):
    """Calculate ceil of a/b without decaying to float."""
    return -(-a // b)

def _align_up(x, alignment):
    """ Align x up to multiple of alignment"""
    return _div_ceil(x, alignment) * alignment


def import_numpy():
    global np
    if np is None:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError('Could not import numpy. Please make sure you have numpy '
                               'installed before you use parallel mode.')


_sample_error_msg = (
    "Unsupported callback return type. Expected NumPy array, PyTorch or MXNet cpu tensors, "
    "DALI TensorCPU, or list or tuple of them representing sample. Got `{}` instead.")


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
    """Container for essential meta data about batch written into shared memory.
    Contains id of the memory chunk, size of the buffer, and offset of more detailed
    meta data passed through shared memory (SampleMeta).
    """

    def __init__(self, mem_chunk_id, capacity, meta_offset, meta_size):
        self.mem_chunk_id = mem_chunk_id
        self.capacity = capacity
        self.meta_offset = meta_offset
        self.meta_size = meta_size

    @classmethod
    def from_writer(cls, writer):
        return cls(writer.mem_batch.mem_chunk_id, writer.mem_batch.capacity, writer.data_size,
                   writer.meta_data_size)


def deserialize_sample(buffer: shared_mem.SharedMem, sample):
    if isinstance(sample, SampleMeta):
        offset = sample.offset
        assert offset % sample.dtype.itemsize == 0, "Sample offset is misaligned."
        buffer = buffer.buf[offset:offset + sample.nbytes]
        return np.ndarray(sample.shape, dtype=sample.dtype, buffer=buffer)
    if isinstance(sample, (tuple, list,)):
        return type(sample)(deserialize_sample(buffer, part) for part in sample)
    return sample


def deserialize_sample_meta(buffer: shared_mem.SharedMem, shared_batch_meta: SharedBatchMeta):
    """Helper to deserialize SampleMeta from memory based on SharedBatchMeta.
    """
    sbm = shared_batch_meta
    if sbm.meta_size == 0:
        return []
    pickled_meta = buffer.buf[sbm.meta_offset:sbm.meta_offset + sbm.meta_size]
    samples_meta = pickle.loads(pickled_meta)
    return samples_meta


def deserialize_batch(buffer: shared_mem.SharedMem, shared_batch_meta: SharedBatchMeta):
    """Deserialize samples from the smem buffer and SampleMeta descriptions.

    Parameters
    ----------
    buffer : shared_mem.SharedMem
        Buffer with serialized sample data
    shared_batch_meta : SharedBatchMeta
        Metadata about serialized data in memory

    Returns
    -------
    List of (idx, numpy array) or (idx, tuple of numpy arrays)
        List of indexed deserialized samples
    """
    samples = deserialize_sample_meta(buffer, shared_batch_meta)
    return [(idx, deserialize_sample(buffer, sample)) for (idx, sample) in samples]


class SharedMemChunk:
    """Simple wrapper around shared memory chunks. Most importantly adds mem_chunk_id used
    to identify chunks in the communication between parent and worker process
    (shared memory handles/file descriptors cannot serve this purpose easily as the same
    mapped memory chunk can have different handles in both processes).
    """

    def __init__(self, mem_chunk_id: str, capacity: int):
        # mem_chunk_id must be unique among all workers and callbacks in the pool,
        # used to identify shared memory chunks in the communication between processes
        self.mem_chunk_id = mem_chunk_id
        self.shm_chunk = shared_mem.SharedMem.allocate(capacity)
        self.capacity = capacity

    def resize(self, new_capacity):
        self.shm_chunk.resize(new_capacity, trunc=True)
        self.capacity = new_capacity

    def close(self):
        # Shared memory handle and mapping will be freed automatically when shm pybind wrapper gets
        # garbage collected anyway, but you can be so nice to free them as soon as you know
        # you won't use them
        self.shm_chunk.close()


def assert_valid_data_type(sample):
    """Check if the output of the callback is type that can be serialized"""
    _apply_to_sample(lambda x : _assert_cpu_sample_data_type(x, _sample_error_msg), sample)


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
    if isinstance(sample, (tuple, list,)):
        # Check that all the samples have common nesting
        for i in range(nest_with_sample):
            assert len(args[i]) == len(sample)
        nest_group = sample, *args[0:nest_with_sample]
        scalar_args = args[nest_with_sample:]
        return type(sample)(_apply_to_sample(func, *part, *scalar_args) for part in zip(*nest_group))
    else:
        # we unpacked all nesting levels, now is actual data:
        return func(sample, *args)


class SharedBatchWriter:
    """SharedBatchWriter can serialize and write batch into given shared
    memory chunk (``mem_batch``).
    """

    SAMPLE_ALIGNMENT = 128
    BUFFER_ALIGNMENT = 4096

    def __init__(self, mem_batch: SharedMemChunk, batch):
        import_numpy()
        self.mem_batch = mem_batch
        self.data_size = 0
        self.meta_data_size = 0
        self._write_batch(batch)

    def _prepare_samples_meta(self, indexed_samples):
        """Calculate metadata and total size of data to be serialized"""
        data_size = 0

        def make_meta(np_array):
            nonlocal data_size
            offset = _align_up(data_size, self.SAMPLE_ALIGNMENT)
            data_size = offset + np_array.nbytes
            return SampleMeta(offset, np_array.shape, np_array.dtype, np_array.nbytes)

        meta = []
        for idx, sample in indexed_samples:
            meta.append((idx, _apply_to_sample(make_meta, sample)))
        return meta, data_size

    def _add_array_to_batch(self, np_array, meta, memview):
        sample_size = meta.nbytes
        offset = meta.offset
        buffer = memview[offset:(offset + sample_size)]
        shared_array = np.ndarray(
            np_array.shape, dtype=np_array.dtype, buffer=buffer)
        shared_array.ravel()[:] = np_array.ravel()[:]

    def _write_batch(self, batch):
        if not batch:
            return
        batch = [(idx, _apply_to_sample(lambda x: _sample_to_numpy(x, _sample_error_msg), sample))
                 for idx, sample in batch]
        meta, data_size = self._prepare_samples_meta(batch)
        serialized_meta = pickle.dumps(meta)
        self.meta_data_size = len(serialized_meta)
        self.data_size = _align_up(data_size, self.SAMPLE_ALIGNMENT)
        needed_capacity = self.data_size + self.meta_data_size
        if self.mem_batch.capacity < needed_capacity:
            new_capacity = max(needed_capacity, 2 * self.mem_batch.capacity)
            new_capacity = _align_up(new_capacity, self.BUFFER_ALIGNMENT)
            self.mem_batch.resize(new_capacity)
        memview = self.mem_batch.shm_chunk.buf
        for (idx, sample), (meta_idx, sample_meta) in zip(batch, meta):
            assert idx == meta_idx
            _apply_to_sample(self._add_array_to_batch, sample, sample_meta, memview, nest_with_sample=1)
        # copy meta data at the end of shared memory chunk
        buffer = memview[self.data_size:(self.data_size + self.meta_data_size)]
        buffer[:] = serialized_meta



def write_batch(mem_batch: SharedMemChunk, batch):
    """Serialize and write the indexed data batch `batch` into the shared memory `mem_batch`.

    Returns description of serialized memory.

    Parameters
    ----------
    mem_batch : SharedMemChunk
        Target memory to write to.
    batch : List of (idx, Sample)
        Batch of data to be serialized

    Returns
    -------
        SharedBatchMeta
    """
    sbw = SharedBatchWriter(mem_batch, batch)
    return SharedBatchMeta.from_writer(sbw)
