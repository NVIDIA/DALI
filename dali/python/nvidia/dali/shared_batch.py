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


class NPSerialized:

    def __init__(self, offset, shape, dtype, nbytes):
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.nbytes = nbytes

    @classmethod
    def from_np(cls, offset, np_array):
        return cls(offset, np_array.shape, np_array.dtype, np_array.nbytes)


class SharedBatchSerialized:
    """Container for essential meta data about batch written into shared memory.
Contains id of the memory chunk, size of the buffer, and offset of more detailed
meta data passed through shared memory (such as offset of every single arrays).
"""

    def __init__(self, mem_chunk_id, capacity, meta_offset, meta_size):
        self.mem_chunk_id = mem_chunk_id
        self.capacity = capacity
        self.meta_offset = meta_offset
        self.meta_size = meta_size

    @classmethod
    def from_writer(cls, writer):
        return cls(writer.mem_batch.mem_chunk_id, writer.mem_batch.capacity, writer.size, writer.meta_data_size)


def deserialize_sample(buffer, sample):
    if isinstance(sample, NPSerialized):
        offset = sample.offset
        buffer = buffer[offset:offset + sample.nbytes]
        return np.ndarray(sample.shape, dtype=sample.dtype, buffer=buffer)
    if isinstance(sample, (tuple, list,)):
        return type(sample)(deserialize_sample(buffer, part) for part in sample)
    return sample


def deserialize_batch(buffer, samples):
    return [(idx, deserialize_sample(buffer, sample)) for (idx, sample) in samples]


class SharedMemChunk:
    """Simple wrapper around shared memory chunks. Most importantly adds mem_chunk_id used to identify chunks
in the communication between parent and worker process (file descriptors cannot serve this
purpose easily as the same memory chunk can have different fds in both processes).
"""

    def __init__(self, mem_chunk_id, capacity):
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
        # garbage collected anyway, but you can be so nice to free them as soon as you know you won't use them
        self.shm_chunk.close()


class SharedBatchWriter:
    """SharedBatchWriter can serialize and write batch into given shared
memory chunk (``mem_batch``).
"""

    def __init__(self, mem_batch):
        import_numpy()
        self.mem_batch = mem_batch
        self.meta_data_size = 0
        self.size = 0

    def _add_array_to_batch(self, memview, np_array):
        sample_size = np_array.nbytes
        offset = self.size
        self.size += sample_size
        if memview is None:  # dummy run without actually copying data, just for meta data
            return NPSerialized(offset, np_array.shape, np_array.dtype, sample_size)
        buffer = memview[offset:(offset + sample_size)]
        shared_array = np.ndarray(
            np_array.shape, dtype=np_array.dtype, buffer=buffer)
        shared_array[:] = np_array[:]
        return NPSerialized.from_np(offset, shared_array)

    def _add_sample_to_batch(self, memview, sample):
        if isinstance(sample, (tuple, list,)):
            return type(sample)(self._add_sample_to_batch(memview, part) for part in sample)
        if isinstance(sample, np.ndarray):
            return self._add_array_to_batch(memview, sample)
        assert False  # samples are converted to numpy first, it should not take place

    def _to_numpy(self, sample):
        if isinstance(sample, np.ndarray):
            return sample
        if isinstance(sample, (tuple, list,)):
            return type(sample)(self._to_numpy(part) for part in sample)
        if types._is_mxnet_array(sample):
            return sample.asnumpy()
        if types._is_torch_tensor(sample):
            if sample.device.type != 'cpu':
                raise TypeError("GPU tensors are not supported")
            return sample.numpy()
        elif isinstance(sample, tensors.TensorCPU):
            return np.array(sample)
        raise TypeError("Unsupported callback return type. Expected numpy array, pytorch or mxnet cpu tensors, "
                        "or list or tuple of them.")

    def _serialize_meta_data(self, batch):
        """Prepares pickled meta-data on batch content, i.e. info on arrays (or possibly lists or
        tuples of arrays) that batch is comprised of, such as shape, offset of array content
        in underlying buffer etc., but doesn't actually copy any array contents into the buffer.
        Additionally returns required capacity of underlying buffer.
        """
        samples = []
        for idx, sample in batch:
            samples.append((idx, self._add_sample_to_batch(None, sample)))
        serialized = pickle.dumps(samples)
        needed_capacity = self.size + len(serialized)
        self.size = 0
        return serialized, needed_capacity

    def write_batch(self, batch):
        if not batch:
            return
        batch = [(idx, self._to_numpy(sample)) for idx, sample in batch]
        serialized, needed_capacity = self._serialize_meta_data(batch)
        self.meta_data_size = len(serialized)
        if self.mem_batch.capacity < needed_capacity:
            self.mem_batch.resize(max([needed_capacity, 2 * self.mem_batch.capacity]))
        memview = self.mem_batch.shm_chunk.buf
        for _, sample in batch:
            self._add_sample_to_batch(memview, sample)
        # copy meta data at the end of shared memory chunk
        offset = self.size
        buffer = memview[offset:(offset + self.meta_data_size)]
        buffer[:] = serialized
