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
from nvidia.dali._multiproc.shared_batch import SharedBatchWriter, SharedBatchMeta, BufShmChunk, \
    assert_valid_data_type, read_shm_message, write_shm_message
from nvidia.dali._multiproc.messages import CompletedTask, WorkerArgs, ShmMessage, ScheduledTask
from nvidia.dali._multiproc import shared_mem
from nvidia.dali._multiproc.shared_queue import Dispatcher, DispatcherWorker, ShmQueue


class _ProcessedTask:
    """Internal worker message send to disptacher with completed tasks where it is
    serialized and dispatched to the pool"""

    def __init__(self, scheduled, shm_chunk, data_batch=None, exception=None,
                 traceback_str=None):
        self.context_i = scheduled.context_i
        self.scheduled_i = scheduled.scheduled_i
        self.minibatch_i = scheduled.task.minibatch_i
        self.shm_chunk = shm_chunk
        self.data_batch = data_batch
        self.exception = exception
        self.traceback_str = traceback_str

    @classmethod
    def done(cls, scheduled, shm_chunk, data_batch):
        return cls(scheduled, shm_chunk, data_batch)

    @classmethod
    def failed(cls, scheduled, shm_chunk, exception, traceback_str=None):
        return cls(scheduled, shm_chunk, exception=exception, traceback_str=traceback_str)

    def is_failed(self):
        return self.exception is not None


class SharedBatchDispatcherWorker(DispatcherWorker):
    """SharedBatchesDispatcher serializes batches, puts them into provided
    shared memory chunks along with completed task description and puts information
    about ready chunks into the `queue`"""

    def __init__(self, worker_id, recv_queues, pending_cv, pending, queue):
        super().__init__(pending_cv, pending, queue)
        self.worker_id = worker_id
        self.recv_queues = recv_queues

    def serialize_failed_task(self, processed_task : _ProcessedTask):
        shm_chunk = processed_task.shm_chunk
        completed_task = CompletedTask.failed(self.worker_id, processed_task)
        return write_shm_message(
            self.worker_id, shm_chunk, completed_task, 0, resize=True)

    def serialize_done_task(self, processed_task : _ProcessedTask):
        shm_chunk = processed_task.shm_chunk
        sbw = SharedBatchWriter(shm_chunk.shm_chunk, processed_task.data_batch)
        batch_meta = SharedBatchMeta.from_writer(sbw)
        completed_task = CompletedTask.done(self.worker_id, processed_task, batch_meta)
        return write_shm_message(
            self.worker_id, shm_chunk, completed_task, sbw.total_size, resize=True)

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

    def start_threads(self, workers):
        for worker in workers:
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

    def wait(self):
        with self.receiver_state.idle_cv:
            while not self.receiver_state.is_interrupted and not self._is_idle_state():
                self.receiver_state.idle_cv.wait()
            return self.receiver_state.is_interrupted

    def receiver_loop(self):
        try:
            while True:
                if self.wait():
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
    """Wraps iterator/generator passed to External Source to enforce ES `cycle` policy specified by the user.
    It is a counterpart of _CycleIter/_CycleGenIter wrappers from non parallel mode.
    However due to prefetching in parallel mode `cycle`=raise will raise StopIteration in consecutive calls
    until the new epoch starts (i.e. which happens with pipline.reset call)"""

    def __init__(self, source_desc):
        self.source_desc = source_desc
        self.reset(epoch_start=0)

    def __call__(self, scheduled : ScheduledTask):
        epoch_start = scheduled.epoch_start
        if self.raised_stop_iter:
            # if iterator is not resetable after failure or epoch was not restarted
            if epoch_start <= self.epoch_start or self.source_desc.cycle != "raise":
                raise StopIteration
            else:
                self.reset(epoch_start)
        try:
            return self.get_next()
        except StopIteration:
            if self.source_desc.cycle != "quiet":
                raise
            # in quiet mode immediately reset the source and return first iteration
            self.reset(epoch_start)
            return self.get_next()

    def reset(self, epoch_start):
        self.iter = IterableSource.get_iter(self.source_desc)
        self.raised_stop_iter = False
        self.epoch_start = epoch_start

    def get_next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.raised_stop_iter = True
            raise

    @staticmethod
    def get_iter(source_desc):
        source = source_desc.source
        if _is_generator_function(source):
            source = source()
        return iter(source)


class CallableSource:

    def __init__(self, source_desc):
        self.callback = source_desc.source

    def __call__(self, scheduled : ScheduledTask):
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
    if callback_pickler is not None:
        for source_desc in sources_desc.values():
            source_desc.source = callback_pickler.loads(source_desc.source)
    callbacks = {
        context_i : get_source_from_desc(source_desc)
        for context_i, source_desc in sources_desc.items()}
    return callbacks


def recv_shm(sock_reader, capacity):
    handle, shm_chunk = -1, None
    try:
        handle = reduction.recv_handle(sock_reader)
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


def init_queue(sock_reader, queue : ShmQueue):
    shm = recv_shm(sock_reader, queue.shm_capacity)
    queue.set_shm(shm)
    shm.seal()
    return queue


def init_task_receiver(general_task_queue, dedicated_task_queue):
    assert general_task_queue is not None or dedicated_task_queue is not None
    if dedicated_task_queue is None or general_task_queue is None:
        return SimpleQueueTaskReceiver(general_task_queue or dedicated_task_queue)
    receiver = MixedTaskReceiver([general_task_queue, dedicated_task_queue])
    excl_worker = EagerReceiverWorker(dedicated_task_queue, receiver.state)
    general_worker = IdleReceiverWorker(general_task_queue, receiver.state)
    receiver.start_threads([excl_worker, general_worker])
    return receiver


def init_shm(shm_chunks_meta, sock_reader):
    shm_chunks = {}
    for shm_chunk_meta in shm_chunks_meta:
        shm_chunk_id = shm_chunk_meta.shm_chunk_id
        shm = recv_shm(sock_reader, shm_chunk_meta.capacity)
        shm_chunks[shm_chunk_id] = BufShmChunk(shm_chunk_id, shm)
    return shm_chunks


def init_dispatcher(worker_id, results_queue, recv_queues):
    dispatcher = Dispatcher(results_queue)
    worker = SharedBatchDispatcherWorker(
        worker_id, recv_queues, dispatcher.pending_cv, dispatcher.pending, results_queue)
    dispatcher.start_thread(worker)
    return dispatcher


def sync_init(worker_id, result_queue, sock_reader):
    # let main process know shared resources setup is done
    result_queue.put([ShmMessage(worker_id, 0, 0, 0, 0)])
    if sock_reader is not None:
        sock_reader.shutdown(socket.SHUT_RDWR)
        sock_reader.close()


def worker(worker_args : WorkerArgs):
    """Entry point of worker process.

    Computes the data in the main thread, in separate threads:
    * waits for incoming tasks,
    * serializes results and passes them to the main process.
    """
    callbacks = init_callbacks(worker_args.sources_desc, worker_args.callback_pickler)
    if worker_args.start_method == "fork":
        shm_chunks = {shm.shm_chunk_id : shm for shm in worker_args.shm_chunks}
    else:
        init_queue(worker_args.sock_reader, worker_args.result_queue)
        if worker_args.general_task_queue is not None:
            init_queue(worker_args.sock_reader, worker_args.general_task_queue)
        if worker_args.dedicated_task_queue is not None:
            init_queue(worker_args.sock_reader, worker_args.dedicated_task_queue)
        shm_chunks = init_shm(worker_args.shm_chunks, worker_args.sock_reader)
    sync_init(worker_args.worker_id, worker_args.result_queue, worker_args.sock_reader)
    task_receiver, batch_dispatcher = None, None
    try:
        task_receiver = init_task_receiver(
            worker_args.general_task_queue, worker_args.dedicated_task_queue)
        batch_dispatcher = init_dispatcher(
            worker_args.worker_id, worker_args.result_queue, task_receiver.get_recv_queues())
        while True:
            scheduled_meta = task_receiver.get_task()
            if scheduled_meta is None:
                break
            shm_chunk = shm_chunks[scheduled_meta.shm_chunk_id]
            scheduled = read_shm_message(shm_chunk.shm_chunk, scheduled_meta)
            try:
                callback = callbacks[scheduled.context_i]
                data_batch = callback(scheduled)
                for sample in data_batch:
                    assert_valid_data_type(sample)
            except Exception as exception:
                tb_str = traceback.format_exc()
                processed = _ProcessedTask.failed(scheduled, shm_chunk, exception, tb_str)
            else:
                processed = _ProcessedTask.done(scheduled, shm_chunk, data_batch)
            batch_dispatcher.append(processed)
    finally:
        if batch_dispatcher is not None:
            batch_dispatcher.close()
        if task_receiver is not None:
            task_receiver.close()
