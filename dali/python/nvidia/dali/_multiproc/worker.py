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

from typing import List
import threading
import traceback
import os
import socket
from collections import deque
from multiprocessing import reduction
from nvidia.dali._utils.external_source_impl import SourceKind, _is_generator_function
from nvidia.dali._multiproc.shared_batch import SharedBatchWriter, SharedBatchMeta, assert_valid_data_type, \
    read_shm_message, write_shm_message
from nvidia.dali._multiproc.messages import CompletedTask, ProcessContext, ShmMessage
from nvidia.dali._multiproc import shared_mem
from nvidia.dali._multiproc.shared_queue import Dispatcher, DispatcherWorker


class _ProcessedTask:
    """Internal worker message send to disptacher with completed tasks where it is
    serialized and dispatched to the pool"""

    def __init__(self, scheduled, shm_chunk, shm_chunk_id, data_batch=None, exception=None,
                 traceback_str=None):
        self.context_i = scheduled.context_i
        self.scheduled_i = scheduled.scheduled_i
        self.minibatch_i = scheduled.task.minibatch_i
        self.shm_chunk_id = shm_chunk_id
        self.shm_chunk = shm_chunk
        self.data_batch = data_batch
        self.exception = exception
        self.traceback_str = traceback_str

    @classmethod
    def done(cls, scheduled, shm_chunk, shm_chunk_id, data_batch):
        return cls(scheduled, shm_chunk, shm_chunk_id, data_batch)

    @classmethod
    def failed(cls, scheduled, shm_chunk, shm_chunk_id, exception, traceback_str=None):
        return cls(scheduled, shm_chunk, shm_chunk_id, exception=exception, traceback_str=traceback_str)

    def is_failed(self):
        return self.exception is not None


class SharedBatchDispatcherWorker(DispatcherWorker):
    """SharedBatchesDispatcher serializes batches, puts them into provided
    shared memory chunks along with completed task description and puts information
    about ready chunks into the `queue`"""

    def __init__(self, pending_cv, pending, queue, worker_id, recv_queues):
        super().__init__(pending_cv, pending, queue)
        self.worker_id = worker_id
        self.recv_queues = recv_queues

    def serialize_failed_task(self, processed_task):
        shm_chunk = processed_task.shm_chunk
        shm_chunk_id = processed_task.shm_chunk_id
        completed_task = CompletedTask.failed(self.worker_id, processed_task)
        return write_shm_message(
            self.worker_id, shm_chunk, shm_chunk_id, completed_task, 0, resize=True)

    def serialize_done_task(self, processed_task):
        shm_chunk = processed_task.shm_chunk
        shm_chunk_id = processed_task.shm_chunk_id
        sbw = SharedBatchWriter(shm_chunk, processed_task.data_batch)
        batch_meta = SharedBatchMeta.from_writer(sbw)
        completed_task = CompletedTask.done(self.worker_id, processed_task, batch_meta)
        return write_shm_message(
            self.worker_id, shm_chunk, shm_chunk_id, completed_task, sbw.total_size, resize=True)

    def serialize_msgs(self, processed_tasks: List[_ProcessedTask]):
        shm_msgs = []
        for processed_task in processed_tasks:
            if processed_task.is_failed():  # one of the tasks failed
                shm_msgs.append(self.serialize_failed_task(processed_task))
            else:
                shm_msgs.append(self.serialize_done_task(processed_task))
        return shm_msgs

    def close(self):
        # dispatcher thread is finishing, so close receiving queues to prevent
        # main thread from blocking on waiting for incoming tasks
        for queue in self.recv_queues:
            queue.close()


class SimpleQueueTaskReceiver:
    """
    Simple wrapper around shm queue, pops first element from the queue
    and returns
    """

    def __init__(self, queue):
        self.queue = queue

    def get_task(self):
        recv = self.queue.get()
        if recv is None:
            return
        [task] = recv
        return task

    def get_recv_queues(self):
        return [self.queue]

    def close(self):
        self.queue.close()


class MixedTaskReceiver:
    """
    Mixes eager and idle workers each reading from different queue and putting tasks into common tasks_queue.
    Eager worker reads whenever any data is available and moves them into the common queue,
    whereas idle worker serves as a fallback that aims to read a single item only if eager worker has no tasks
    available and main thread waits for tasks.
    """

    class MixedReceiverState:

        def __init__(self):
            self.lock = threading.Lock()
            self.tasks_cv = threading.Condition(lock=self.lock)
            self.idle_cv = threading.Condition(lock=self.lock)
            self.is_idle = False
            self.is_interrupted = False
            self.tasks_queue = deque()

        def insert_task(self, recv):
            with self.tasks_cv:
                if recv is None:
                    self.tasks_queue.appendleft(recv)
                else:
                    self.tasks_queue.extend(recv)
                self.tasks_cv.notify()

    def __init__(self, recv_queues):
        self.recv_queues = recv_queues
        self.state = self.MixedReceiverState()
        self.threads = []

    def start_thread(self, excl_worker, general_worker):
        for worker in (excl_worker, general_worker):
            thread = threading.Thread(target=worker.receiver_loop, daemon=True)
            thread.start()
            self.threads.append(thread)

    def get_recv_queues(self):
        return self.recv_queues

    def close(self):
        if self.threads:
            with self.state.lock:
                self.state.is_interrupted = True
                self.state.idle_cv.notify()
            for queue in self.recv_queues:
                queue.close()
            for thread in self.threads:
                thread.join()
            self.threads.clear()

    def get_task(self):
        with self.state.tasks_cv:
            waited = False
            while len(self.state.tasks_queue) == 0:
                # there's only one consumer of tasks_queue, so no stealing of tasks between waits can happen
                if not waited:
                    waited = True
                    self.state.is_idle = True
                    self.state.idle_cv.notify()
                self.state.tasks_cv.wait()
            self.state.is_idle = False
            task = self.state.tasks_queue.popleft()
        return task


class EagerReceiverWorker:

    def __init__(self, recv_queue, receiver_state : MixedTaskReceiver.MixedReceiverState):
        self.recv_queue = recv_queue
        self.receiver_state = receiver_state

    def receiver_loop(self):
        try:
            while True:
                recv = self.recv_queue.get(num_samples=None)
                if recv is None:
                    break
                self.receiver_state.insert_task(recv)
        finally:
            self.receiver_state.insert_task(None)


class IdleReceiverWorker:

    def __init__(self, recv_queue, receiver_state : MixedTaskReceiver.MixedReceiverState):
        self.recv_queue = recv_queue
        self.receiver_state = receiver_state

    def _is_idle_state(self):
        return self.receiver_state.is_idle and len(self.receiver_state.tasks_queue) == 0

    def _recheck_should_take(self):
        with self.receiver_state.idle_cv:
            return not self.receiver_state.is_interrupted and self._is_idle_state()

    def receiver_loop(self):
        try:
            while True:
                with self.receiver_state.idle_cv:
                    while not self.receiver_state.is_interrupted and not self._is_idle_state():
                        self.receiver_state.idle_cv.wait()
                    if self.receiver_state.is_interrupted:
                        break
                # Worker has no dedicated work to do (is idle), so take one task from general queue.
                # If general queue is empty, the call will block and then recheck the condition
                recv = self.recv_queue.get(get_if_waited=self._recheck_should_take)
                if recv is None:
                    break
                if len(recv):  # if _recheck_should_take returned False, recv is an empty list
                    self.receiver_state.insert_task(recv)
        finally:
            self.receiver_state.insert_task(None)


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


def init_callbacks(sources_desc, callback_pickler):
    sources_desc = sources_desc
    if callback_pickler is not None:
        for source_desc in sources_desc.values():
            source_desc.source = callback_pickler.loads(source_desc.source)
    callbacks = {
        context_i : get_source_from_desc(source_desc)
        for context_i, source_desc in sources_desc.items()}
    return callbacks


def recv_shm(socket, capacity):
    handle, shm_chunk = -1, None
    try:
        handle = reduction.recv_handle(socket)
        assert os.fstat(handle).st_size >= capacity
        return shared_mem.SharedMem.open(handle, capacity)
    except:
        if shm_chunk is not None:
            shm_chunk.close()
        # close handle manually if shm_chunk creation failed, otherwise shm_chunk
        # is responsible for doing so
        elif handle >= 0:
            os.close(handle)
        raise


def init_queue(sock_reader, queue):
    shm = recv_shm(sock_reader, queue.shm_min_capacity)
    queue.set_shm(shm)
    return queue


def init_task_receiver(general_task_queue, exclusive_task_queue, sock_reader):
    assert general_task_queue is not None or exclusive_task_queue is not None
    if exclusive_task_queue is None or general_task_queue is None:
        return SimpleQueueTaskReceiver(general_task_queue or exclusive_task_queue)
    receiver = MixedTaskReceiver([general_task_queue, exclusive_task_queue])
    excl_worker = EagerReceiverWorker(exclusive_task_queue, receiver.state)
    general_worker = IdleReceiverWorker(general_task_queue, receiver.state)
    receiver.start_thread(excl_worker, general_worker)
    return receiver


def init_shm(contexts_shm, sock_reader):
    shm_chunks = {}
    contexts_range = max(contexts_shm.keys()) + 1
    for context_i in range(contexts_range):
        if context_i not in contexts_shm:
            continue
        for shm_meta in contexts_shm[context_i]:
            shm_chunks[shm_meta.shm_chunk_id] = recv_shm(sock_reader, shm_meta.capacity)
    return shm_chunks


def init_dispatcher(worker_id, results_queue, recv_queues):
    dispatcher = Dispatcher(results_queue)
    worker = SharedBatchDispatcherWorker(dispatcher.pending_cv, dispatcher.pending, results_queue, worker_id, recv_queues)
    dispatcher.start_thread(worker)
    return dispatcher


def sync_init(worker_id, result_queue, sock_reader):
    # let main process know shared resources setup is done
    result_queue.put([ShmMessage(worker_id, 0, 0, 0, 0)])
    sock_reader.shutdown(socket.SHUT_RDWR)
    sock_reader.close()


def worker(process_context : ProcessContext):
    """Entry point of worker process.

    Computes the data in the main thread, in separate threads:
    * waits for incoming tasks,
    * serializes results and passes them to the main process.

    [TODO]
    """
    callbacks = init_callbacks(process_context.sources_desc, process_context.callback_pickler)
    if process_context.start_method == "fork":
        shm_chunks = process_context.contexts_shm
    else:
        init_queue(process_context.sock_reader, process_context.result_queue)
        if process_context.general_task_queue is not None:
            init_queue(process_context.sock_reader, process_context.general_task_queue)
        if process_context.exclusive_task_queue is not None:
            init_queue(process_context.sock_reader, process_context.exclusive_task_queue)
        shm_chunks = init_shm(process_context.contexts_shm, process_context.sock_reader)
        sync_init(process_context.worker_id, process_context.result_queue, process_context.sock_reader)
    task_receiver, batch_dispatcher = None, None
    try:
        task_receiver = init_task_receiver(
            process_context.general_task_queue, process_context.exclusive_task_queue,
            process_context.sock_reader)
        batch_dispatcher = init_dispatcher(process_context.worker_id, process_context.result_queue, task_receiver.get_recv_queues())
        while True:
            scheduled_meta = task_receiver.get_task()
            if scheduled_meta is None:
                break
            shm_chunk_id = scheduled_meta.shm_chunk_id
            shm_chunk = shm_chunks[shm_chunk_id]
            scheduled = read_shm_message(shm_chunk, scheduled_meta)
            try:
                callback = callbacks[scheduled.context_i]
                data_batch = callback(scheduled)
                for sample in data_batch:
                    assert_valid_data_type(sample)
            except Exception as exception:
                tb_str = traceback.format_exc()
                processed = _ProcessedTask.failed(scheduled, shm_chunk, shm_chunk_id, exception, tb_str)
            else:
                processed = _ProcessedTask.done(scheduled, shm_chunk, shm_chunk_id, data_batch)
            batch_dispatcher.append(processed)
    finally:
        if batch_dispatcher is not None:
            batch_dispatcher.close()
        if task_receiver is not None:
            task_receiver.close()
        if shm_chunks is not None:
            for shm_chunk in shm_chunks.values():
                shm_chunk.close()
