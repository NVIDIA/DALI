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
from multiprocessing import reduction, Pipe, connection
from nvidia.dali._utils.external_source_impl import SourceKind, _is_generator_function
from nvidia.dali._multiproc.shared_batch import write_batch, assert_valid_data_type
from nvidia.dali._multiproc.messages import CompletedTasks, ProcessContext
from nvidia.dali._multiproc import shared_mem


class _ProcessedTasks:
    """Internal worker message send to disptacher with completed tasks where it is
    serialized and dispatched to the pool"""

    def __init__(self, scheduled, shm_chunk=None, data_batch=None, exception=None,
                 traceback_str=None):
        self.context_i = scheduled.context_i
        self.scheduled_i = scheduled.scheduled_i
        self.minibatch_i = scheduled.task.minibatch_i
        self.shm_chunk_id = scheduled.shm_chunk_id
        self.shm_capacity = scheduled.shm_capacity
        self.shm_chunk = shm_chunk
        self.data_batch = data_batch
        self.exception = exception
        self.traceback_str = traceback_str

    @classmethod
    def done(cls, scheduled, shm_chunk, data_batch):
        return cls(scheduled, shm_chunk, data_batch)

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

    def __init__(self, worker_id, res_pipe):
        self.worker_id = worker_id
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
        if processed_tasks.shm_capacity != processed_tasks.shm_chunk.capacity:
            processed_tasks.shm_chunk.resize(processed_tasks.shm_capacity, trunc=False)
        batch_meta = write_batch(processed_tasks.shm_chunk, processed_tasks.data_batch)
        completed_tasks = CompletedTasks.done(self.worker_id, processed_tasks, batch_meta)
        self.res_pipe.send(completed_tasks)

    def _shutdown(self):
        """Force to close all communication channels (sockets and pipes) to unlock main process
        in case of error, when it may be waiting for messages that we can't deliver or the
        state of protocol is mismatched"""
        self.res_pipe.close()

class ReceiverWorker:

    def __init__(self, tasks_queue, tasks_cv, exclusive_tasks_pipe):
        self.tasks_queue = tasks_queue
        self.tasks_cv = tasks_cv
        self.exclusive_tasks_pipe = exclusive_tasks_pipe
        self.thread = threading.Thread(target=self.receiver_loop, daemon=True)
        self.thread.start()

    def wait_for_task(self):
        return self.exclusive_tasks_pipe.recv()

    def receiver_loop(self):
        try:
            while True:
                scheduled = self.wait_for_task()
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


class MixedReceiverWorker(ReceiverWorker):

    def __init__(self, tasks_queue, tasks_cv, exclusive_tasks_pipe, general_tasks_pipe, general_tasks_lock, is_idle_pipe):
        self.general_tasks_pipe = general_tasks_pipe
        self.general_tasks_lock = general_tasks_lock
        self.is_idle_pipe = is_idle_pipe
        self.is_idle = True
        super().__init__(tasks_queue, tasks_cv, exclusive_tasks_pipe)

    def wait_for_task(self):
        while True:
            pipes = [self.exclusive_tasks_pipe, self.is_idle_pipe]
            if self.is_idle:
                pipes.append(self.general_tasks_pipe)
            ready = connection.wait(pipes)
            if self.is_idle_pipe in ready:
                self.is_idle_pipe.recv()
                self.is_idle = True
                continue
            if self.exclusive_tasks_pipe in ready:
                self.is_idle = False
                return self.exclusive_tasks_pipe.recv()
            if self.general_tasks_pipe in ready:
                self.is_idle = False
                with self.general_tasks_lock:
                    return self.general_tasks_pipe.recv()


class TaskReceiver:

    def __init__(self, exclusive_tasks_pipe):
        assert exclusive_tasks_pipe is not None, "exclusive tasks queue must be provided"
        self.exclusive_tasks_pipe = exclusive_tasks_pipe
        self.tasks_cv = threading.Condition()
        self.tasks_queue = []
        self.receiver_worker = self.start_receiver_worker()

    def start_receiver_worker(self):
        return ReceiverWorker(self.tasks_queue, self.tasks_cv, self.exclusive_tasks_pipe)

    def get_task(self):
        with self.tasks_cv:
            while len(self.tasks_queue) == 0:
                self.tasks_cv.wait()
            scheduled = self.tasks_queue.pop(0)
        return scheduled


class MixedTaskReceiver(TaskReceiver):

    def __init__(self, exclusive_tasks_pipe, general_tasks_pipe, general_tasks_lock):
        self.general_tasks_pipe = general_tasks_pipe
        self.general_tasks_lock = general_tasks_lock
        self.is_idle_pipe, self.notify_idle_pipe = Pipe(duplex=False)
        super().__init__(exclusive_tasks_pipe)

    def start_receiver_worker(self):
        return MixedReceiverWorker(
            self.tasks_queue, self.tasks_cv, self.exclusive_tasks_pipe,
            self.general_tasks_pipe, self.general_tasks_lock, self.is_idle_pipe)

    def get_task(self):
        self.notify_idle_pipe.send(None)
        return super().get_task()


class SimpleQueueTaskReceiver:

    def __init__(self, task_pipe, task_lock):
        self.task_pipe = task_pipe
        self.task_lock = task_lock

    def get_task(self):
        try:
            with self.task_lock:
                return self.task_pipe.recv()
        except EOFError:
            return None


def receive_shm(worker_id, contexts_shm_meta, sock_reader, result_pipe):
    shm_chunks = {}
    contexts_range = max(contexts_shm_meta.keys()) + 1
    for context_i in range(contexts_range):
        if context_i not in contexts_shm_meta:
            continue
        for shm_meta in contexts_shm_meta[context_i]:
            handle, shm_chunk = -1, None
            try:
                handle = reduction.recv_handle(sock_reader)
                # TODO(windows): We're pretending here that the handle is not a fd, which in fact
                # it is. On windows the call below probably needs to be adjusted.
                assert os.fstat(handle).st_size >= shm_meta.capacity
                shm_chunk = shared_mem.SharedMem.open(handle, shm_meta.capacity)
                shm_chunks[shm_meta.shm_chunk_id] = shm_chunk
            except:
                if shm_chunk is not None:
                    shm_chunk.close()
                # close handle manually if shm_chunk creation failed, otherwise shm_chunk
                # is responsible for doing so
                elif handle >= 0:
                    os.close(handle)
                raise
    result_pipe.send(worker_id)
    sock_reader.shutdown(socket.SHUT_RDWR)
    sock_reader.close()
    return shm_chunks


class IterableSource:

    def __init__(self, source_desc):
        self.source_desc = source_desc
        self.epoch_start = 0
        self.raised_stop_iter = False
        self.iter = IterableSource.get_iter(source_desc)

    def __call__(self, scheduled):
        epoch_start = scheduled.epoch_start
        if self.raised_stop_iter:
            # if iterator is not resetable after failure or epoch was not restarted
            if epoch_start <= self.epoch_start or self.source_desc.cycle != "raise":
                raise StopIteration
            else:
                self.iter = IterableSource.get_iter(self.source_desc)
                self.raised_stop_iter = False
                self.epoch_start = epoch_start
        try:
            return next(self.iter)
        except StopIteration:
            if self.source_desc.cycle != "quiet":
                self.raised_stop_iter = True
                raise
            self.iter = IterableSource.get_iter(self.source_desc)
            return next(self.iter)

    @staticmethod
    def get_iter(source_desc):
        source = source_desc.source
        if _is_generator_function(source):
            source = source()
        return iter(source)


class CallableSource:

    def __init__(self, source_desc):
        self.callback = source_desc.source

    def __call__(self, scheduled):
        task = scheduled.task
        if task.is_sample_mode():
            data_batch = [self.callback(*task_args) for task_args in task.samples_args]
        else:
            data_batch = self.callback(*task.batch_args)
        return data_batch


def get_source_from_desc(sources_desc):
    if sources_desc.kind == SourceKind.CALLABLE:
        return CallableSource(sources_desc)
    elif sources_desc.kind in (SourceKind.ITERABLE, SourceKind.GENERATOR_FUNC):
        return IterableSource(sources_desc)
    raise RuntimeError("Unsupported source type")


def init_callbacks(process_context):
    sources_desc = process_context.sources_desc
    if process_context.callback_pickler is not None:
        for source_desc in sources_desc.values():
            source_desc.source = process_context.callback_pickler.loads(source_desc.source)
    callbacks = {
        context_i : get_source_from_desc(source_desc)
        for context_i, source_desc in sources_desc.items()}
    return callbacks


def init_task_receiver(process_context):
    if process_context.exclusive_task_pipe is None:
        return SimpleQueueTaskReceiver(process_context.general_task_pipe, process_context.general_task_lock)
    if process_context.general_task_pipe is None:
        return TaskReceiver(process_context.exclusive_task_pipe)
    else:
        return MixedTaskReceiver(
            process_context.exclusive_task_pipe, process_context.general_task_pipe,
            process_context.general_task_lock)


def worker(process_context : ProcessContext):
    """Entry point of worker process.

    Computes the data in the main thread, in separate threads:
    * waits for incoming tasks,
    * serializes results and passes them to the main process.

    [TODO]
    """
    callbacks = init_callbacks(process_context)
    shm_chunks = receive_shm(
        process_context.worker_id, process_context.contexts_shm_meta,
        process_context.sock_reader, process_context.result_pipe)
    batch_dispatcher = SharedBatchesDispatcher(process_context.worker_id, process_context.result_pipe)
    # run the thread as a daemon so that even when results queue blocks, worker process can exit anyway
    # and can be joined in the parent process
    dispatcher_thread = threading.Thread(target=batch_dispatcher.dispatcher_thread, daemon=True)
    dispatcher_thread.start()
    task_receiver = init_task_receiver(process_context)
    try:
        while True:
            scheduled = task_receiver.get_task()
            if scheduled is None:
                break
            try:
                callback = callbacks[scheduled.context_i]
                data_batch = callback(scheduled)
                for sample in data_batch:
                    assert_valid_data_type(sample)
            except Exception as exception:
                tb_str = traceback.format_exc()
                processed = _ProcessedTasks.failed(scheduled, exception, tb_str)
            else:
                processed = _ProcessedTasks.done(scheduled, shm_chunks[scheduled.shm_chunk_id], data_batch)
            batch_dispatcher.dispatch(processed)
    finally:
        batch_dispatcher.dispatch(None)
        if shm_chunks is not None:
            for shm_chunk in shm_chunks.values():
                shm_chunk.close()
