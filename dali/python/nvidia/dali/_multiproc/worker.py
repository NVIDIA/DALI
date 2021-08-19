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

import threading
import traceback
import os
import socket
from multiprocessing import reduction
from nvidia.dali._multiproc.shared_batch import SharedMemChunk, write_batch, assert_valid_data_type
from nvidia.dali._multiproc.messages import CompletedTasks


class _ProcessedTasks:
    """Internal worker message send to disptacher with completed tasks where it is
    serialized and dispatched to the pool"""

    def __init__(self, scheduled, mem_chunk=None, data_batch=None, exception=None,
                 traceback_str=None):
        self.context_i = scheduled.context_i
        self.batch_i = scheduled.batch_i
        self.mem_chunk = mem_chunk
        self.data_batch = data_batch
        self.exception = exception
        self.traceback_str = traceback_str

    @classmethod
    def done(cls, scheduled, mem_chunk, data_batch):
        return cls(scheduled, mem_chunk, data_batch)

    @classmethod
    def failed(cls, scheduled, exception, traceback_str=None):
        return cls(scheduled, exception=exception, traceback_str=traceback_str)

    def is_failed(self):
        return self.exception is not None


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
        self.handle_sent = set()
        self.sock = sock
        self.res_pipe = res_pipe
        self.ready_cv = threading.Condition()
        self.ready_queue = []

    def dispatch(self, processed_task: _ProcessedTasks):
        """Pass the processed task (or None to end) to the dispatcher.
        """
        with self.ready_cv:
            if processed_task is None:
                self.ready_queue.insert(0, None)
            else:
                self.ready_queue.append(processed_task)
            self.ready_cv.notify()

    def dispatcher_thread(self):
        """Receives batches produced in the main thread and dispatches them to the parent process.
        It is intended to be run in a separate thread because both callback and dispatcher may
        wait on IO operations a lot and in that case Python threads provide some performance gain.
        """
        try:
            while True:
                message = self._wait_for_processed()
                if message is None:
                    break
                self._send(message)
        finally:
            # In case of error, we don't know when exactly we were interrupted and the main process
            # may be waiting for differant message that we would try to send. Close the communication
            # to indicate an error for main process and allow the exception to propagate
            # in the worker process.
            self._shutdown()

    def _wait_for_processed(self):
        with self.ready_cv:
            while len(self.ready_queue) == 0:
                self.ready_cv.wait()
            message = self.ready_queue.pop(0)
        return message

    def _send(self, processed_tasks: _ProcessedTasks):
        """Send the processed task back to the main process"""
        if processed_tasks.is_failed():  # one of the tasks failed
            completed_tasks = CompletedTasks.failed(self.worker_id, processed_tasks)
            self.res_pipe.send(completed_tasks)
            return
        serialized_batch = write_batch(processed_tasks.mem_chunk, processed_tasks.data_batch)
        completed_tasks = CompletedTasks.done(self.worker_id, processed_tasks, serialized_batch)
        self.res_pipe.send(completed_tasks)
        # send shared memory handle for underlaying shared memory chunk
        # if it hasn't been sent ever before
        mem_chunk_id = serialized_batch.mem_chunk_id
        if mem_chunk_id not in self.handle_sent:
            self.handle_sent.add(mem_chunk_id)
            reduction.send_handle(self.sock, processed_tasks.mem_chunk.shm_chunk.handle, os.getppid())

    def _shutdown(self):
        """Force to close all communication channels (sockets and pipes) to unlock main process
        in case of error, when it may be waiting for messages that we can't deliver or the
        state of protocol is mismatched"""
        self.res_pipe.close()
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

class TaskReceiver:
    def __init__(self, task_pipe):
        self.task_pipe = task_pipe
        self.tasks_cv = threading.Condition()
        self.tasks_queue = []

    def get_task(self):
        with self.tasks_cv:
            while len(self.tasks_queue) == 0:
                self.tasks_cv.wait()
            scheduled = self.tasks_queue.pop(0)
        return scheduled

    def receiver_thread(self):
        """Receives list of tasks scheduled to be done by the worker.
        Intended to be run in a separate thread to avoid blocking of the main process when
        it schedules another batch for the worker and worker is busy with previously scheduled
        computations.
        """
        try:
            while True:
                scheduled = self.task_pipe.recv()
                if scheduled is None:
                    break
                self._insert_task(scheduled)
        finally:
            self._insert_task(None)

    def _insert_task(self, scheduled_task):
        with self.tasks_cv:
            if scheduled_task is None:
                self.tasks_queue.insert(0, None)
            else:
                self.tasks_queue.append(scheduled_task)
            self.tasks_cv.notify()


class CallbackContext:
    """Worker can run multiple Python callbacks, CallbackContext is used to
    (independently from other callbacks) manage shared memory used to pass
    results of the callback calls.
    """

    def __init__(self, callback, mem_chunks):
        self.callback = callback
        self.mem_chunks = mem_chunks

    def close(self):
        for chunk in self.mem_chunks:
            chunk.close()


def worker(worker_id, callbacks, prefetch_queue_depths, initial_chunk_size, task_pipe, res_pipe, sock, callback_pickler):
    """Entry point of worker process.

    Computes the data in the main thread, in separate threads:
    * waits for incoming tasks,
    * serializes results and passes them to the main process.

    Parameters
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
        Python wrapper around Unix socket used to pass file descriptors identifying shared memory chunk to parent process.
    """
    if callback_pickler is not None:
        callbacks = callback_pickler.loads(callbacks)
    contexts = None
    batch_dispatcher = SharedBatchesDispatcher(worker_id, sock, res_pipe)
    task_receiver = TaskReceiver(task_pipe)
    # run the thread as a daemon so that even when results queue blocks, worker process can exit anyway
    # and can be joined in the parent process
    dispatcher_thread = threading.Thread(target=batch_dispatcher.dispatcher_thread, daemon=True)
    receiver_thread = threading.Thread(target=task_receiver.receiver_thread, daemon=True)
    dispatcher_thread.start()
    receiver_thread.start()
    try:
        contexts = [
            CallbackContext(callback, [
                SharedMemChunk("chunk_{}_{}_{}".format(worker_id, callback_idx, prefetch_idx), initial_chunk_size)
                for prefetch_idx in range(prefetch_queue_depth)
            ])
            for callback_idx, (callback, prefetch_queue_depth) in enumerate(zip(callbacks, prefetch_queue_depths))
        ]
        while True:
            scheduled = task_receiver.get_task()
            if scheduled is None:
                break
            context = contexts[scheduled.context_i]
            callback = context.callback
            try:
                data_batch = [(task_id, callback(*task_args))
                              for (task_id, task_args) in scheduled.tasks]
                for i, sample in data_batch:
                    assert_valid_data_type(sample)
            except Exception as exception:
                tb_str = traceback.format_exc()
                processed = _ProcessedTasks.failed(scheduled, exception, tb_str)
            else:
                processed = _ProcessedTasks.done(scheduled, context.mem_chunks[scheduled.dst_chunk_i], data_batch)
            batch_dispatcher.dispatch(processed)
    finally:
        batch_dispatcher.dispatch(None)
        if contexts is not None:
            for context in contexts:
                context.close()
