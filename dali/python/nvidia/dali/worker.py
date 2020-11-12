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

import threading
from multiprocessing import reduction
from nvidia.dali.shared_batch import SharedMemChunk, SharedBatchSerialized, SharedBatchWriter


class ScheduledTasks:
    """Message send from pool to a worker to schedule tasks for the worker"""

    def __init__(self, context_i, batch_i, tasks):
        self.context_i = context_i
        self.batch_i = batch_i
        self.tasks = tasks


class _ProcessedTasks:
    """Internal worker message send to disptacher with completed tasks where it is
serialized and dispatched to the pool"""

    def __init__(self, scheduled, mem_chunk=None, data_batch=None, exception=None):
        self.context_i = scheduled.context_i
        self.batch_i = scheduled.batch_i
        self.mem_chunk = mem_chunk
        self.data_batch = data_batch
        self.exception = exception

    @classmethod
    def done(cls, scheduled, mem_chunk, data_batch):
        return cls(scheduled, mem_chunk, data_batch)

    @classmethod
    def failed(cls, scheduled, exception):
        return cls(scheduled, exception=exception)


class CompletedTasks:
    """Message send from a worker to the pool to notify the pool about completed tasks
along with meta data needed to fetch and deserialize results stored in the shared memory"""

    def __init__(self, worker_id, context_i, batch_i, batch_serialized=None, exception=None):
        self.worker_id = worker_id
        self.context_i = context_i
        self.batch_i = batch_i
        self.batch_serialized = batch_serialized
        self.exception = exception

    @classmethod
    def done(cls, worker_id, processed, batch_serialized):
        return cls(worker_id, processed.context_i, processed.batch_i, batch_serialized=batch_serialized)

    @classmethod
    def failed(cls, worker_id, processed):
        return cls(worker_id, processed.context_i, processed.batch_i, exception=processed.exception)


class SharedBatchesDispatcher:
    """SharedBatchesDispatcher serializes batches, puts them into provided
shared memory chunks and notifies parent process of batch ready to be read from shared memory.
It keeps track of what shared memory chunks have been already sent and if needed, sends
file descriptors of memory chunks that parent process hasn't seen yet.

Parameters
----------
`worker_id` : int
    Id of the worker passed by the parent process. Added to messages sent over the ``res_pipe``
    to simplify bookeeping in parent process.
`sock` : socket
    Python wrapper around Unix socket, capable of sending file descriptors between processes.
`res_pipe`: pipe
    Pipe used to send parent process a notification (along with essential meta data info) about
    ready batch in a given shared memory chunk.
"""
    def __init__(self, worker_id, sock, res_pipe):
        self.worker_id = worker_id
        self.fd_sent = set()
        self.sock = sock
        self.res_pipe = res_pipe

    def send(self, processed_tasks):
        if processed_tasks.exception is not None:  # one of the tasks failed
            completed_tasks = CompletedTasks.failed(self.worker_id, processed_tasks)
            self.res_pipe.send(completed_tasks)
            return
        mem_chunk = processed_tasks.mem_chunk
        writer = SharedBatchWriter(processed_tasks.mem_chunk)
        writer.write_batch(processed_tasks.data_batch)
        batch_serialized = SharedBatchSerialized.from_writer(writer)
        completed_tasks = CompletedTasks.done(self.worker_id, processed_tasks, batch_serialized)
        self.res_pipe.send(completed_tasks)
        # send file descriptor for underlaying shared memory chunk if it hasn't been sent ever before
        mem_chunk_id = batch_serialized.mem_chunk_id
        if mem_chunk_id not in self.fd_sent:
            self.fd_sent.add(mem_chunk_id)
            reduction.sendfds(self.sock, [mem_chunk.shm_chunk.fd])


class CallbackContext:
    """Worker can run multiple Python callbacks, CallbackContext is used to (independently from other callbacks)
manage shared memory used to pass results of the callback calls.
"""

    def __init__(self, callback, mem_chunks):
        self.callback = callback
        self.mem_chunks = mem_chunks
        self.batch_i = 0

    def next_batch_i(self):
        res = self.batch_i
        self.batch_i = (self.batch_i + 1) % len(self.mem_chunks)
        return res

    def next_mem_chunk(self):
        batch_i = self.next_batch_i()
        return self.mem_chunks[batch_i]

    def close(self):
        for chunk in self.mem_chunks:
            chunk.close()


def dispatcher(batch_dispatcher, ready_cv, ready_queue):
    """Receives batches produced in the main thread and dispatches them to the parent process.
It is run in a separate thread because both callback and dispatcher may
wait on IO operations a lot and in that case Python threads provide some performance gain.
"""
    while True:
        with ready_cv:
            while len(ready_queue) == 0:
                ready_cv.wait()
            message = ready_queue.pop(0)
        if message is None:
            break
        batch_dispatcher.send(message)


def receiver(task_pipe, tasks_cv, tasks_queue):
    """Receives list of tasks scheduled to be done by the worker, run in a separate thread
to avoid blocking of the main process when it schedules another batch for the worker and
worker is busy with previously scheduled computations.
"""
    try:
        while True:
            scheduled = task_pipe.recv()
            if scheduled is None:
                break
            with tasks_cv:
                tasks_queue.append(scheduled)
                tasks_cv.notify()
    finally:
        with tasks_cv:
            tasks_queue.insert(0, None)
            tasks_cv.notify()


def worker(worker_id, callbacks, prefetch_queue_depths, initial_chunk_size, task_pipe, res_pipe, sock):
    """Entry point of worker process.

Parameters
----------
`callbacks` : callable list
    List of callables that worker can call to perform a (part of parallelized) task.
`prefetch_queue_depths` : int
    Number of shared memory chunks that should be allocated per callaback, used in cycle buffer manner
    to pass callback results to parent process.
`initial_chunk_size` : int
    Initial size of shared memory chunk.
`task_pipe`: Pipe
    Pipe used to read list of tasks that given callback should be run on to produce (part of a) result batch.
`res_pipe`: Pipe
    Pipe used to notify the parent process about another batch ready to read in the given memory chunk.
`sock` : socket
    Python wrapper around Unix socket used to pass file descriptors identifying shared memory chunk to parent process.
"""
    contexts = None
    ready_cv = threading.Condition()
    tasks_cv = threading.Condition()
    ready_queue, tasks_queue = [], []
    batch_dispatcher = SharedBatchesDispatcher(worker_id, sock, res_pipe)
    # run the thread as a daemon so that even when results queue blocks worker process can exit anyway
    # and can be joined in the parent process
    dispatcher_thread = threading.Thread(target=dispatcher, args=(batch_dispatcher, ready_cv, ready_queue), daemon=True)
    receiver_thread = threading.Thread(target=receiver, args=(task_pipe, tasks_cv, tasks_queue), daemon=True)
    dispatcher_thread.start()
    receiver_thread.start()
    try:
        contexts = [
            CallbackContext(callback, [
                SharedMemChunk("chunk_{}_{}_{}".format(worker_id, i, j), initial_chunk_size)
                for j in range(prefetch_queue_depth)
            ])
            for i, (callback, prefetch_queue_depth) in enumerate(zip(callbacks, prefetch_queue_depths))
        ]
        while True:
            with tasks_cv:
                while len(tasks_queue) == 0:
                    tasks_cv.wait()
                scheduled = tasks_queue.pop(0)
            if scheduled is None:
                break
            context = contexts[scheduled.context_i]
            callback = context.callback
            try:
                data_batch = [(task_id, callback(*task_args)) for (task_id, task_args) in scheduled.tasks]
            except Exception as exception:
                processed = _ProcessedTasks.failed(scheduled, exception)
            else:
                mem_chunk = context.next_mem_chunk()
                processed = _ProcessedTasks.done(scheduled, mem_chunk, data_batch)
            with ready_cv:
                if len(ready_queue) >= prefetch_queue_depths[scheduled.context_i]:
                    raise RuntimeError("Worker queue size exceeded")
                ready_queue.append(processed)
                ready_cv.notify()
    finally:
        with ready_cv:
            ready_queue.insert(0, None)
            ready_cv.notify()
        if contexts is not None:
            for context in contexts:
                context.close()
