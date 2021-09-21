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


class ShmMeta:

    def __init__(self, shm_chunk_id, capacity):
        self.shm_chunk_id = shm_chunk_id
        self.capacity = capacity

    @classmethod
    def from_shm_buffer(cls, shm_buffer):
        return [cls(shm_chunk_id, shm_buffer.initial_chunk_size) for shm_chunk_id in shm_buffer.chunks_ids]


class ProcessContext:
    """
    Parameters[TODO]
    ----------
    `callbacks` : callable list
        List of callables that worker can call to perform a (part of parallelized) task.
    `prefetch_queue_depths` : list of int
        Number of shared memory chunks that should be allocated per callaback, used in cycle buffer manner
        to pass callback results to parent process.
    `initial_chunk_size` : int
        Initial size of shared memory chunk.
    `task_pipe`: Pipe
        Pipe used to read list of tasks that given callback should be run on to produce (part of a) result batch.
    `res_pipe`: Pipe
        Pipe used to notify the parent process about another batch ready to read in the given memory chunk.
    `sock` : socket
        Python wrapper around Unix socket used to pass file descriptors identifying shared memory chunk to parent process.[TODO]"""

    def __init__(self, worker_id, start_method, sources_desc, contexts, general_task_queue,
                 exclusive_task_queue, result_queue, sock_reader, callback_pickler):
        if start_method != "fork":
            contexts_shm = {
                context_i : ShmMeta.from_shm_buffer(contexts[context_i].shm_buffer)
                for context_i in sources_desc}
        else:
            shm_buffers = [contexts[context_i].shm_buffer for context_i in sources_desc]
            contexts_shm = {
                shm_id : shm_buffer.get_chunk_by_id(shm_id)
                for shm_buffer in shm_buffers
                for shm_id in shm_buffer.chunks_ids}
        self.worker_id = worker_id
        self.start_method = start_method
        self.sources_desc = sources_desc
        self.contexts_shm = contexts_shm
        self.general_task_queue = general_task_queue
        self.exclusive_task_queue = exclusive_task_queue
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
    `dst_chunk_i` : int
        Index of the memory chunk in the circular buffer to store the output in
    `task` : nvidia.dali.types.SampleInfo list [TODO]
        List of task ordered to be computed.
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
    `batch_meta` :  nvidia.dali._multiproc.shared_batch.SharedBatchMeta
        Serialized result of the task.
    `exception`
        Exception if the task failed. [TODO]
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
