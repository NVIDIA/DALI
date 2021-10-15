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


from typing import Optional
from nvidia.dali.types import SampleInfo
from nvidia.dali._multiproc.struct_message import Structure


class ShmMessage(Structure):
    """
    Type of C-struct like message exchanged via shared memory queue (`ShmQueue`).
    ----------
    `worker_id` : int
        Intger identifying a process that put the message, number from [0, num_workers) range for workers,
        -1 in case of a main process.
    `shm_chunk_id` : int
        Integer identifying shm chunk that contains pickled data to be read by the receiver
    `shm_capacity` : int
        Size of the `shm_chunk_id` chunk, receiver should resize the mapping if the chunk
        was resized by the writer.
    `offset` : int
        Offset in the shm chunk where the serialized message starts
    `num_bytes` : int
        Size in bytes of the serialized message
    """
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
    `source_descs` : Dictionary with External Source's SourceDescription instances as values. Keys are ordinals corresponding to
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
        Optional custom pickler that was applied to serialize callbacks in `source_descs`"""

    def __init__(self, *, worker_id, start_method, source_descs, shm_chunks, general_task_queue,
                 dedicated_task_queue, result_queue, sock_reader, callback_pickler):
        self.worker_id = worker_id
        self.start_method = start_method
        self.source_descs = source_descs
        self.shm_chunks = shm_chunks
        self.general_task_queue = general_task_queue
        self.dedicated_task_queue = dedicated_task_queue
        self.result_queue = result_queue
        self.sock_reader = sock_reader
        self.callback_pickler = callback_pickler


class SampleRange:
    """
    Describes a batch of work in sample mode that consists of SampleInfo instances with consecutive
    indices within the epoch. Used to avoid linear dependency of the task description on the batch size.
    """

    def __init__(self, sample_start, sample_end, iteration, epoch_idx, *, slice_start=0, slice_end=None):
        self.sample_start = sample_start # idx in epoch of first sample in batch
        self.sample_end = sample_end # idx in epoch of one past last sample in batch
        self.iteration = iteration # index of a batch within epoch
        self.epoch_idx = epoch_idx
        # idx of first sample in slice (in a batch not an epoch)
        self.slice_start = slice_start
        if slice_end is None:
            slice_end = sample_end - sample_start
        # idx of one past last sample in slice (in a batch not an epoch)
        self.slice_end = slice_end

    @classmethod
    def slice(cls, sample_range, slice_start, slice_end):
        assert sample_range.slice_start <= slice_start < slice_end <= sample_range.slice_end
        return cls(
            sample_range.sample_start, sample_range.sample_end,
            sample_range.iteration, sample_range.epoch_idx,
            slice_start=slice_start,
            slice_end=slice_end)

    def get_slice(self, slice_start, slice_end):
        return self.slice(self, slice_start, slice_end)

    def __len__(self):
        return self.slice_end - self.slice_start

    def iter_samples(self):
        return (SampleInfo(
            self.sample_start + idx_in_batch,
            idx_in_batch,
            self.iteration,
            self.epoch_idx
            ) for idx_in_batch in range(self.slice_start, self.slice_end))


class TaskArgs:

    @classmethod
    def make_sample(cls, start, end, iteration, epoch_idx):
        sample_range = SampleRange(start, end, iteration, epoch_idx)
        if len(sample_range) <= 0:
            raise RuntimeError("Cannot schedule empty batch")
        return cls(0, sample_range=sample_range)

    @classmethod
    def make_batch(cls, batch_args):
        return cls(0, batch_args=batch_args)

    def __init__(self, minibatch_i, sample_range : Optional[SampleRange]=None, batch_args=None):
        self.minibatch_i = minibatch_i
        self.sample_range = sample_range
        self.batch_args = batch_args
        assert ((self.sample_range is None) != (self.batch_args is None))

    def is_sample_mode(self):
        return self.sample_range is not None


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
    `task` : TaskArgs
        Describes the minibatch that should be computed by the worker. If the given source
        is run in batch mode this simply wraps parameters that external source would pass to the
        source in non-parallel mode. In sample mode, it is (part of) the list of nvidia.dali.types.SampleInfo
        produced by the external source.
    """

    def __init__(self, context_i, scheduled_i, epoch_start, task : TaskArgs):
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
