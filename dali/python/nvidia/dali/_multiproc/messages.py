# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class ShmMessageDesc(Structure):
    """
    Type of C-struct like message exchanged via shared memory queue (`ShmQueue`).
    It describes placement (shared memory chunk, offset etc.) of actual data to be read
    by the receiver of the `ShmMessageDesc` instance.
    ----------
    worker_id : int
        Integer identifying a process that put the message, number from [0, num_workers) range
        for workers or -1 in case of a main process.
    shm_chunk_id : int
        Integer identifying shm chunk that contains pickled data to be read by the receiver
    shm_capacity : unsigned long long int
        Size of the `shm_chunk_id` chunk, receiver should resize the mapping if the chunk
        was resized by the writer.
    offset : unsigned long long int
        Offset in the shm chunk where the serialized message starts
    num_bytes : unsigned long long int
        Size in bytes of the serialized message
    """

    _fields = (
        ("worker_id", "i"),
        ("shm_chunk_id", "i"),
        ("shm_capacity", "Q"),
        ("offset", "Q"),
        ("num_bytes", "Q"),
    )


class WorkerArgs:
    """
    Pack of parameters passed to the worker process on initialization.
    ----------
    worker_id : Ordinal of the worker in the workers pool
    start_method : Python's multiprocessing start method - `spawn` or `fork`
    source_descs : Dictionary with External Source's SourceDescription instances as values.
        Keys are ordinals corresponding to the order in which callbacks were passed to the pool.
        If `callback_pickler` is not None, actual callback in SourceDescription is replaced
        with result of its serialization.
    shm_chunks : list of BufShmChunk instances that describes all the shared memory chunks
        available to the worker (they are identified by ids unique inside the pool).
    general_task_queue : Optional[ShmQueue]
        Queue with tasks for sources without dedicated worker
        or None if all sources have dedicated worker
    dedicated_task_queue : Optional[ShmQueue]
        Queue with tasks for sources that are run solely in the given worker.
        If `dedicated_task_queue` is None, `general_task_queue` must be provided.
    result_queue : ShmQueue
        Queue to report any task done, no matter if dedicated or general.
    setup_socket : Optional[socket]
        Python wrapper around Unix socket used to pass file descriptors identifying
        shared memory chunk to child process. None if `start_method='fork'`
    `callback_pickler`
        Optional custom pickler that was applied to serialize callbacks in `source_descs`"""

    def __init__(
        self,
        *,
        worker_id,
        start_method,
        source_descs,
        shm_chunks,
        general_task_queue,
        dedicated_task_queue,
        result_queue,
        setup_socket,
        callback_pickler,
    ):
        self.worker_id = worker_id
        self.start_method = start_method
        self.source_descs = source_descs
        self.shm_chunks = shm_chunks
        self.general_task_queue = general_task_queue
        self.dedicated_task_queue = dedicated_task_queue
        self.result_queue = result_queue
        self.setup_socket = setup_socket
        self.callback_pickler = callback_pickler


class SampleRange:
    """
    Describes a batch or sub-batch of work in sample mode that consists of SampleInfo
    instances with consecutive indices. It denotes range of samples within given `iteration`
    of given `epoch_idx`, optionally specifying a slice/sub-range of the sample range.
    It does not support spanning over multiple batches. Used to avoid linear dependency of the task
    description size on the batch size.
    """

    def __init__(
        self, sample_start, sample_end, iteration, epoch_idx, *, slice_start=0, slice_end=None
    ):
        self.sample_start = sample_start  # idx in epoch of first sample in batch
        self.sample_end = sample_end  # idx in epoch of one past last sample in batch
        self.iteration = iteration  # index of a batch within epoch
        self.epoch_idx = epoch_idx
        if slice_end is None:
            slice_end = sample_end - sample_start
        assert slice_start >= 0 and slice_start <= sample_end - sample_start
        assert slice_end >= slice_start and slice_end <= sample_end - sample_start
        # idx of first sample in slice (in a batch not an epoch)
        self.slice_start = slice_start
        # idx of one past last sample in slice (in a batch not an epoch)
        self.slice_end = slice_end

    def _get_index(self, idx, bound):
        if idx is None:
            return bound
        if idx < 0:
            return self.slice_end + idx
        return self.slice_start + idx

    def _get_slice(self, range_slice: slice):
        if range_slice.step is not None and range_slice.step != 1:
            raise ValueError("SampleRange only supports slicing with step 1")

        slice_start = self._get_index(range_slice.start, self.slice_start)
        slice_end = self._get_index(range_slice.stop, self.slice_end)
        slice_start = min(slice_start, self.slice_end)
        slice_end = max(min(slice_end, self.slice_end), slice_start)
        return SampleRange(
            self.sample_start,
            self.sample_end,
            self.iteration,
            self.epoch_idx,
            slice_start=slice_start,
            slice_end=slice_end,
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._get_slice(idx)
        if idx < 0:
            idx_in_batch = self.slice_end + idx
        else:
            idx_in_batch = self.slice_start + idx
        if idx_in_batch < self.slice_start or idx_in_batch >= self.slice_end:
            raise IndexError("Index {} out of range for slice of length {}".format(idx, len(self)))
        return SampleInfo(
            self.sample_start + idx_in_batch, idx_in_batch, self.iteration, self.epoch_idx
        )

    def __len__(self):
        return self.slice_end - self.slice_start


class TaskArgs:
    @classmethod
    def make_sample(cls, sample_range):
        if len(sample_range) <= 0:
            raise RuntimeError("Cannot schedule empty batch")
        return cls(0, sample_range=sample_range)

    @classmethod
    def make_batch(cls, batch_args):
        return cls(0, batch_args=batch_args)

    def __init__(self, minibatch_i, sample_range: Optional[SampleRange] = None, batch_args=None):
        self.minibatch_i = minibatch_i
        self.sample_range = sample_range
        self.batch_args = batch_args
        assert (self.sample_range is None) != (self.batch_args is None)

    def is_sample_mode(self):
        return self.sample_range is not None


class ScheduledTask:
    """Message sent from the pool to a worker to schedule tasks for the worker

    Parameters
    ----------
    context_i : int
        Index identifying the callback in the order of parallel callbacks passed to pool.
    scheduled_i : int
        Ordinal of the batch that tasks list corresponds to.
    epoch_start : int
        The value is increased every time the corresponding context is reset,
        this way worker can know if the new epoch started, and if it can restart
        iterator that raised StopIteration but is set to cycle=raise.
    task : TaskArgs
        Describes the minibatch that should be computed by the worker. If the given source
        is run in batch mode this simply wraps parameters that external source would pass to
        the source in non-parallel mode. In sample mode, it is (part of) the list
        of nvidia.dali.types.SampleInfo produced by the external source.
    """

    def __init__(self, context_i, scheduled_i, epoch_start, task: TaskArgs):
        self.context_i = context_i
        self.scheduled_i = scheduled_i
        self.epoch_start = epoch_start
        self.task = task


class CompletedTask:
    """Message sent from a worker to the pool to notify the pool about completed tasks
    along with meta data needed to fetch and deserialize results stored in the shared memory

    Parameters
    ----------
    worker_id : int
        Id of the worker that completed the task.
    context_i : int
        Index identifying the callback in the order of parallel callbacks passed to pool.
    scheduled_i : int
        Ordinal of the batch that tasks corresponds to.
    minibatch_i : int
        Computation of batch might be split into number of minibatches, this is the number
        that identifies which consecutive part of the batch it is.
    batch_meta :  nvidia.dali._multiproc.shared_batch.SharedBatchMeta
        Serialized result of the task.
    `exception`
        Exception if the task failed.
    """

    def __init__(
        self,
        worker_id,
        context_i,
        scheduled_i,
        minibatch_i,
        batch_meta=None,
        exception=None,
        traceback_str=None,
    ):
        self.worker_id = worker_id
        self.context_i = context_i
        self.scheduled_i = scheduled_i
        self.minibatch_i = minibatch_i
        self.batch_meta = batch_meta
        self.exception = exception
        self.traceback_str = traceback_str

    @classmethod
    def done(cls, worker_id, processed, batch_meta):
        return cls(
            worker_id,
            processed.context_i,
            processed.scheduled_i,
            processed.minibatch_i,
            batch_meta=batch_meta,
        )

    @classmethod
    def failed(cls, worker_id, processed):
        return cls(
            worker_id,
            processed.context_i,
            processed.scheduled_i,
            processed.minibatch_i,
            exception=processed.exception,
            traceback_str=processed.traceback_str,
        )

    def is_failed(self):
        return self.exception is not None
