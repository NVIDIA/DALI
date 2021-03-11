# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


class ScheduledTasks:
    """Message sent from the pool to a worker to schedule tasks for the worker

    Parameters
    ----------
    `context_i` : int
        Index identifying the callback in the order of parallel callbacks passed to pool.
    `batch_i` : int
        Ordinal of the batch that tasks list corresponds to.
    `dst_chunk_i` : int
        Index of the memory chunk in the circular buffer to store the output in
    `tasks` : nvidia.dali.types.SampleInfo list
        List of task ordered to be computed.
    """

    def __init__(self, context_i, batch_i, dst_chunk_i, tasks):
        self.context_i = context_i
        self.batch_i = batch_i
        self.dst_chunk_i = dst_chunk_i
        self.tasks = tasks


class CompletedTasks:
    """Message sent from a worker to the pool to notify the pool about completed tasks
    along with meta data needed to fetch and deserialize results stored in the shared memory

    Parameters
    ----------
    `worker_id` : int
        Id of the worker that completed the task.
    `context_i` : int
        Index identifying the callback in the order of parallel callbacks passed to pool.
    `batch_i` : int
        Ordinal of the batch that tasks corresponds to.
    `serialized_batch` :  nvidia.dali._multiproc.shared_batch.SharedBatchMeta
        Serialized result of the task.
    `exception`
        Exception if the task failed.
    """

    def __init__(
            self, worker_id, context_i, batch_i, serialized_batch=None, exception=None,
            traceback_str=None):
        self.worker_id = worker_id
        self.context_i = context_i
        self.batch_i = batch_i
        self.serialized_batch = serialized_batch
        self.exception = exception
        self.traceback_str = traceback_str

    @classmethod
    def done(cls, worker_id, processed, serialized_batch):
        return cls(worker_id, processed.context_i, processed.batch_i,
                   serialized_batch=serialized_batch)

    @classmethod
    def failed(cls, worker_id, processed):
        return cls(worker_id, processed.context_i, processed.batch_i, exception=processed.exception,
                   traceback_str=processed.traceback_str)

    def is_failed(self):
        return self.exception is not None
