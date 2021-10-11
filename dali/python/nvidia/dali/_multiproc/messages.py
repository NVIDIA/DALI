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


import struct


class Structure:

    """
    Utility around Python `struct` module that allows to access and modify `_fields` like an ordinary object attributes,
    but also read and write their values from/into the buffer in C struct like format.
    """

    _fields = tuple()

    def __init__(self, *values):
        self.setup_struct()
        self.set_values(*values)

    @classmethod
    def setup_struct(cls):
        if '_struct_desc' not in cls.__dict__:
            cls._struct_desc = "@" + "".join(field_type for _, field_type in cls._fields)
            cls._struct = struct.Struct(cls._struct_desc)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.setup_struct()

    def set_values(self, *values):
        for (field_name, _), value in zip(self._fields, values):
            setattr(self, field_name, value)

    def get_values(self):
        return tuple(getattr(self, field_name) for field_name, _ in self._fields)

    def pack_into(self, buf, offset):
        return self._struct.pack_into(buf, offset, *self.get_values())

    def unpack_from(self, buf, offset):
        values = self._struct.unpack_from(buf, offset)
        self.set_values(*values)
        return self

    def get_size(self):
        return self._struct.size


class ShmMessage(Structure):
    _fields = ("worker_id", "i"), ("shm_chunk_id", "i"), ("shm_capacity", "i"), ("offset", "i"), ("num_bytes", "i")


class BufShmChunkMeta:
    """
    Describes shm chunk passed from the main process to a worker process. Used if `spawn` start method is selected
    and shm chunk handle is passed through a socket to a worker process.
    ----------
    `shm_chunk_id` : Integer identifying shm chunk in communication between the main process and workers.
    `capacity` : Size of the underlying shared memory region.
    """

    def __init__(self, shm_chunk_id, capacity):
        self.shm_chunk_id = shm_chunk_id
        self.capacity = capacity

    @classmethod
    def from_chunk(cls, buf_shm_chunk):
        return cls(buf_shm_chunk.shm_chunk_id, buf_shm_chunk.shm_chunk.capacity)


class WorkerArgs:
    """
    Pack of parameters passed to the worker process on initialization.
    ----------
    `worker_id` : Ordinal of the worker in the workers pool
    `start_method` : Python's multiprocessing start method - `spawn` or `fork`
    `sources_desc` : Dictionary with External Source's SourceDescription instances as values. Keys are ordinals corresponding to
        the order in which callbacks were passed to the pool.
        If `callback_pickler` is not None, actual callback in SourceDescription is replaced with result of its serialization.
    `shm_chunks` : list of either BufShmChunkMeta or BufShmChunk instances (depending on the start method) that
        describes all the shared memory chunks available to the worker (they are identified by ids unique inside the pool).
    `general_task_queue` : Optional[ShmQueue]
        Queue with tasks for sources without dedicated worker, None if all sources have dedicated worker
    `dedicated_task_queue`: Optional[ShmQueue]
        Queue with tasks for sources that are run solely in the given worker.
    `result_queue`: ShmQueue
        Queue to report any task done, no matter if dedicated or general.
    `sock_reader` : Optional[socket]
        Python wrapper around Unix socket used to pass file descriptors identifying shared memory chunk to parent process.
        None if `start_method='fork'`
    `callback_pickler`
        Optional custom pickler that was applied to serialize callbacks in `sources_desc`"""

    def __init__(self, worker_id, start_method, sources_desc, shm_chunks, general_task_queue,
                 dedicated_task_queue, result_queue, sock_reader, callback_pickler):
        self.worker_id = worker_id
        self.start_method = start_method
        self.sources_desc = sources_desc
        self.shm_chunks = shm_chunks
        self.general_task_queue = general_task_queue
        self.dedicated_task_queue = dedicated_task_queue
        self.result_queue = result_queue
        self.sock_reader = sock_reader
        self.callback_pickler = callback_pickler


class BatchArgs:

    @classmethod
    def sample_mode(cls, minibatch_i, samples_args):
        return cls(minibatch_i=minibatch_i, samples_args=samples_args)

    @classmethod
    def batch_mode(cls, batch_args):
        return cls(minibatch_i=0, batch_args=batch_args)

    def __init__(self, minibatch_i, samples_args=None, batch_args=None):
        self.minibatch_i = minibatch_i
        self.samples_args = samples_args
        self.batch_args = batch_args

    def is_sample_mode(self):
        return self.samples_args is not None


class ScheduledTask:
    """Message sent from the pool to a worker to schedule tasks for the worker

    Parameters
    ----------
    `context_i` : int
        Index identifying the callback in the order of parallel callbacks passed to pool.
    `scheduled_i` : int
        Ordinal of the batch that tasks list corresponds to.
    `epoch_start` : int
        The value is increased every time the corresponding context is resetted,
        this way worker can know if the new epoch started, and if it can restart
        iterator that raised StopIteration but is set to cycle=raise.
    `task` : BatchArgs
        Describes the minibatch that should be computed by the worker. If the given source
        is run in batch mode this simply wraps parameters that external source would pass to the
        source in non-parallel mode. In sample mode, it is (part of) the list of nvidia.dali.types.SampleInfo
        produced by the external source.
    """

    def __init__(self, context_i, scheduled_i, epoch_start, task : BatchArgs):
        self.context_i = context_i
        self.scheduled_i = scheduled_i
        self.epoch_start = epoch_start
        self.task = task


class CompletedTask:
    """Message sent from a worker to the pool to notify the pool about completed tasks
    along with meta data needed to fetch and deserialize results stored in the shared memory

    Parameters
    ----------
    `worker_id` : int
        Id of the worker that completed the task.
    `context_i` : int
        Index identifying the callback in the order of parallel callbacks passed to pool.
    `scheduled_i` : int
        Ordinal of the batch that tasks corresponds to.
    `minibatch_i` : int
        Computation of batch might be split into number of minibatches, this is the number
        that identifies which consecutive part of the batch it is.
    `batch_meta` :  nvidia.dali._multiproc.shared_batch.SharedBatchMeta
        Serialized result of the task.
    `exception`
        Exception if the task failed.
    """

    def __init__(
            self, worker_id, context_i, scheduled_i, minibatch_i, batch_meta=None,
            exception=None, traceback_str=None):
        self.worker_id = worker_id
        self.context_i = context_i
        self.scheduled_i = scheduled_i
        self.minibatch_i = minibatch_i
        self.batch_meta = batch_meta
        self.exception = exception
        self.traceback_str = traceback_str

    @classmethod
    def done(cls, worker_id, processed, batch_meta):
        return cls(worker_id, processed.context_i, processed.scheduled_i, processed.minibatch_i,
                   batch_meta=batch_meta)

    @classmethod
    def failed(cls, worker_id, processed):
        return cls(worker_id, processed.context_i, processed.scheduled_i, processed.minibatch_i,
                   exception=processed.exception, traceback_str=processed.traceback_str)

    def is_failed(self):
        return self.exception is not None
