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
    """Message send from pool to a worker to schedule tasks for the worker

    Parameters
    ----------
    `context_i` : int
        Index identifying the callback in the order of parallel callbacks passed to pool.
    `batch_i` : int
        Ordinal of the batch that tasks list corresponds to.
    `tasks` : nvidia.dali.types.SampleInfo list
        List of task ordered to be computed.
    """

    def __init__(self, context_i, batch_i, tasks):
        self.context_i = context_i
        self.batch_i = batch_i
        self.tasks = tasks


class CompletedTasks:
    """Message send from a worker to the pool to notify the pool about completed tasks
    along with meta data needed to fetch and deserialize results stored in the shared memory

    Parameters
    ----------
    `worker_id` : int
        Id of the worker that completed the task.
    `context_i` : int
        Index identifying the callback in the order of parallel callbacks passed to pool.
    `batch_i` : int
        Ordinal of the batch that tasks corresponds to.
    `batch_serialized` :  nvidia.dali.shared_batch.SharedBatchMeta
        Serialized result of computing the task.
    `exception`
        Exception if the task was failed.
    """

    def __init__(
            self, worker_id, context_i, batch_i, batch_serialized=None, exception=None,
            traceback_str=None):
        self.worker_id = worker_id
        self.context_i = context_i
        self.batch_i = batch_i
        self.batch_serialized = batch_serialized
        self.exception = exception
        self.traceback_str = traceback_str

    @classmethod
    def done(cls, worker_id, processed, batch_serialized):
        return cls(worker_id, processed.context_i, processed.batch_i,
                   batch_serialized=batch_serialized)

    @classmethod
    def failed(cls, worker_id, processed):
        return cls(worker_id, processed.context_i, processed.batch_i, exception=processed.exception,
                   traceback_str=processed.traceback_str)

    def is_failed(self):
        return self.exception is not None
