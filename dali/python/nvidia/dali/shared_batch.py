# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali import shared_mem, tensors
from nvidia.dali import types
import pickle


np = None


def import_numpy():
    global np
    if np is None:
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError('Could not import numpy. Please make sure you have numpy '
                               'installed before you use parallel mode.')


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
        buffer = buffer.buf[offset:offset + sample.nbytes]
        return np.ndarray(sample.shape, dtype=sample.dtype, buffer=buffer)
    if isinstance(sample, (tuple, list,)):
        return type(sample)(deserialize_sample(buffer, part) for part in sample)
    return sample


def deserialize_sample_meta(buffer: shared_mem.SharedMem, shared_batch_meta: SharedBatchMeta):
    """Deserialize SampleMeta from memory based on SharedBatchMeta.
    Deserialized SampleMeta can be used to reconstruct the batch data by passing it
    to the `deserialize_batch`.
    """
    sbm = shared_batch_meta
    if sbm.meta_size == 0:
        return []
    pickled_meta = buffer.buf[sbm.meta_offset:sbm.meta_offset + sbm.meta_size]
    samples_meta = pickle.loads(pickled_meta)
    return samples_meta


def deserialize_batch(buffer: shared_mem.SharedMem, samples):
    """Deserialize samples from the smem buffer and SampleMeta descriptions.

    Parameters
    ----------
    buffer : shared_mem.SharedMem
        Buffer with serialized sample data
    samples : List of (idx, SampleMeta) or list of (idx, tuple of SampleMeta).
        Metadata describing the samples

    Returns
    -------
    List of (idx, numpy array) or (idx, tuple of numpy arrays)
        List of indexed deserialized samples
    """
    return [(idx, deserialize_sample(buffer, sample)) for (idx, sample) in samples]


class SharedMemChunk:
    """Simple wrapper around shared memory chunks. Most importantly adds mem_chunk_id used
    to identify chunks in the communication between parent and worker process
    (file descriptors cannot serve this purpose easily as the same memory chunk
    can have different fds in both processes).
    """

    def __init__(self, mem_chunk_id: str, capacity: int):
        # mem_chunk_id must be unique among all workers and callbacks in the pool,
        # used to identify shared memory chunks in the communication between processes
        self.mem_chunk_id = mem_chunk_id
        self.shm_chunk = shared_mem.SharedMem.allocate(capacity)
        self.capacity = capacity

    def resize(self, new_capacity=None):
        capacity = self.capacity
        new_capacity = new_capacity or 2 * capacity
        self.shm_chunk.resize(new_capacity, trunc=True)
        self.capacity = new_capacity

    def close(self):
        # Memory fd and mapping will be freed automatically when shm pybind wrapper gets
        # garbage collected anyway, but you can be so nice to free them as soon as you know
        # you won't use them
        self.shm_chunk.close()


def _to_numpy(sample):
    if isinstance(sample, np.ndarray):
        return sample
    if types._is_mxnet_array(sample):
        return sample.asnumpy()
    if types._is_torch_tensor(sample):
        if sample.device.type != 'cpu':
            raise TypeError("GPU tensors are not supported")
        return sample.numpy()
    elif isinstance(sample, tensors.TensorCPU):
        return np.array(sample)
    raise TypeError(
        "Unsupported callback return type. Expected numpy array, pytorch or mxnet cpu tensors, "
        "or list or tuple of them.")


def _apply_to_sample(func, sample, *args):
    """Apply to a sample traversing the nesting of the data (tuple/list).

    Parameters
    ----------
    func : callable
        Function to be applied to every sample data object
    sample : sample object or any nesting of those in tuple/list
        Representation of sample
    """
    if isinstance(sample, (tuple, list,)):
        return type(sample)(_apply_to_sample(func, part, *args) for part in sample)
    else:
        # we unpacked all nesting levels, now is actual data:
        return func(sample, *args)


class SharedBatchWriter:
    """SharedBatchWriter can serialize and write batch into given shared
    memory chunk (``mem_batch``).
    """

    def __init__(self, mem_batch: SharedMemChunk):
        import_numpy()
        self.mem_batch = mem_batch
        self.data_size = 0
        self.meta_data_size = 0

    def _prepare_samples_meta(self, indexed_samples):
        """Calculate metadata and total size of data to be serialized"""
        data_size = 0

        def make_meta(np_array):
            nonlocal data_size
            offset = data_size
            data_size += np_array.nbytes
            return SampleMeta(offset, np_array.shape, np_array.dtype, np_array.nbytes)

        meta = []
        for idx, sample in indexed_samples:
            meta.append((idx, _apply_to_sample(make_meta, sample)))
        return meta, data_size

    def _add_array_to_batch(self, np_array, memview):
        sample_size = np_array.nbytes
        offset = self.written_size
        self.written_size += sample_size
        buffer = memview[offset:(offset + sample_size)]
        shared_array = np.ndarray(
            np_array.shape, dtype=np_array.dtype, buffer=buffer)
        shared_array[:] = np_array[:]

    def write_batch(self, batch):
        if not batch:
            return
        batch = [(idx, _apply_to_sample(_to_numpy, sample)) for idx, sample in batch]
        meta, data_size = self._prepare_samples_meta(batch)
        serialized_meta = pickle.dumps(meta)
        self.meta_data_size = len(serialized_meta)
        self.data_size = data_size
        needed_capacity = self.data_size + self.meta_data_size
        if self.mem_batch.capacity < needed_capacity:
            self.mem_batch.resize(max([needed_capacity, 2 * self.mem_batch.capacity]))
        memview = self.mem_batch.shm_chunk.buf
        self.written_size = 0
        for idx, sample in batch:
            _apply_to_sample(self._add_array_to_batch, sample, memview)
        assert self.written_size == self.data_size, "Mismatch in written and precalculated size."
        # copy meta data at the end of shared memory chunk
        buffer = memview[self.data_size:(self.data_size + self.meta_data_size)]
        buffer[:] = serialized_meta
