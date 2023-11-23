# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Optional, Tuple
import threading
import traceback
import socket
from collections import deque
from multiprocessing import reduction
from nvidia.dali._utils.external_source_impl import SourceKind, _is_generator_function
from nvidia.dali._multiproc.shared_batch import (
    SharedBatchWriter,
    SharedBatchMeta,
    BufShmChunk,
    assert_valid_data_type,
    read_shm_message,
    write_shm_message,
)
from nvidia.dali._multiproc.messages import CompletedTask, WorkerArgs, ShmMessageDesc, ScheduledTask
from nvidia.dali._multiproc.shared_queue import Dispatcher


class _WorkerProcessingResult:
    """Internal worker message containing computed minibatch or error message sent from the main
    thread to the dispatcher thread. The dispatcher thread serializes the batch or the error and
    forwards the result as `CompletedTask` to the main process"""

    def __init__(self, scheduled, shm_chunk, data_batch=None, exception=None, traceback_str=None):
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


class SharedBatchDispatcher(Dispatcher):
    """SharedBatchesDispatcher serializes batches, puts them into provided
    shared memory chunks along with completed task description and puts information
    about ready chunks into the `queue`. It processes tasks in a separate thread to
    overlap serialization of minibatches with next minibatches computation in case of
    a callback waiting on IO extensively and to avoid multiple worker processes
    waiting on inter-process ShmQueue access"""

    def __init__(self, worker_id, result_queue, recv_queues):
        # close receiving queues if writing results fails to unblock
        # the main thread that may be waiting on new tasks to process
        def on_thread_exit():
            for queue in recv_queues:
                queue.close()

        super().__init__(result_queue, on_thread_exit)
        self.worker_id = worker_id

    def _serialize_failed_task(self, processed_task: _WorkerProcessingResult):
        """
        Puts CompletedTask instance (that describes an error encountered when producing batch)
        in the provided shared memory chunk (`processed_task.shm_chunk`).
        Returns `ShmMessageDesc` instance, that describes shared memory chunk and placement
         offset=0, size) of the serialized CompletedTask instance in the chunk.
        """
        shm_chunk = processed_task.shm_chunk
        completed_task = CompletedTask.failed(self.worker_id, processed_task)
        return write_shm_message(self.worker_id, shm_chunk, completed_task, 0, resize=True)

    def _serialize_done_task(self, processed_task: _WorkerProcessingResult):
        """
        Puts produced batch in the provided shared memory chunk (`processed_task.shm_chunk`).
        Layout of the data in the chunk:
        [1. samples from the batch | 2. batch meta-data | 3. completed task].
        1. Binary encoded samples from the batch (underlying data of numpy arrays),
           aimed to be used as initialization buffers for arrays with no additional copy
           or deserialization.
        2. Pickled list of meta-data of each sample, such as the sample's binary data offset in
           the chunk, a shape and a type of the array.
        3. Pickled CompletedTask instance (that contains offset and size of the serialized list
           from the second point).
        Returns `ShmMessageDesc` instance, that describes shared memory chunk and placement
        (offset, size) of the serialized CompletedTask instance in the chunk.
        """
        shm_chunk = processed_task.shm_chunk
        sbw = SharedBatchWriter(shm_chunk, processed_task.data_batch)
        batch_meta = SharedBatchMeta.from_writer(sbw)
        completed_task = CompletedTask.done(self.worker_id, processed_task, batch_meta)
        return write_shm_message(
            self.worker_id, shm_chunk, completed_task, sbw.total_size, resize=True
        )

    def serialize_msgs(self, processed_tasks: List[_WorkerProcessingResult]):
        shm_msgs = []
        for processed_task in processed_tasks:
            if processed_task.is_failed():  # one of the tasks failed
                shm_msgs.append(self._serialize_failed_task(processed_task))
            else:
                shm_msgs.append(self._serialize_done_task(processed_task))
        return shm_msgs


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
    Mixes eager and idle worker threads each taking tasks from a different inter-process queue and
    putting the tasks into a single (worker's internal) `task_queue`. Eager worker thread takes
    tasks from the dedicated queue, i.e. tasks that can be processed only by the given worker
    process. Idle worker thread takes tasks from the general queue, i.e. tasks that can be
    processed by any worker process from the pool.
    Eager worker reads tasks whenever any is available and moves them into the worker's internal
    queue, whereas idle worker serves as a fallback that aims to read a single item only if
    the internal queue is empty and the main thread does not process any task (is idle).
    """

    class EagerReceiverWorker:
        """
        Worker thread waiting for any tasks available in the inter-process queue
        `dedicated_task_queue`. If anything is available, it takes all the items
        and puts them into worker's internal task queue.
        """

        def __init__(self, receiver_state, dedicated_task_queue):
            self.receiver_state = receiver_state
            self.dedicated_task_queue = dedicated_task_queue
            self.thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.thread.start()

        def _receiver_loop(self):
            try:
                while True:
                    recv = self.dedicated_task_queue.get(num_samples=None)
                    if recv is None:
                        break
                    self.receiver_state.insert_task(recv)
            finally:
                self.receiver_state.insert_task(None)

        def close(self):
            self.dedicated_task_queue.close()
            self.thread.join()

    class IdleReceiverWorker:
        """
        Worker thread that, when notified, takes a single task from the inter-process queue and
        puts it into worker's internal task queue. It aims to take the task only if the main thread
        reports it has no tasks to process - it rechecks that condition if it had to wait on empty
        inter-process queue.
        """

        def __init__(self, receiver_state, general_task_queue):
            self.receiver_state = receiver_state
            self.general_task_queue = general_task_queue
            self.thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.thread.start()

        def _receiver_loop(self):
            try:
                while True:
                    if not self.receiver_state.wait_for_idle():
                        break
                    # Worker has no dedicated work to do (is idle), so take one task from
                    # general queue.
                    # If general queue is empty, the call will block and then
                    # recheck the condition
                    recv_pred = self.receiver_state.is_idle_and_uninterrupted
                    recv = self.general_task_queue.get(predicate=recv_pred)
                    if recv is None:
                        break
                    # if `is_idle_and_uninterrupted` returned False, recv is an empty list
                    if len(recv):
                        self.receiver_state.insert_task(recv)
            finally:
                self.receiver_state.insert_task(None)

        def close(self):
            self.receiver_state.interrupt_idle()
            self.general_task_queue.close()
            self.thread.join()

    class MixedReceiverState:
        def __init__(self):
            self.lock = threading.Lock()
            self.tasks_cv = threading.Condition(lock=self.lock)
            self.idle_cv = threading.Condition(lock=self.lock)
            self.is_idle = False
            self.is_interrupted = False
            self.task_queue = deque()

        def _is_idle_state(self):
            return self.is_idle and len(self.task_queue) == 0

        def is_idle_and_uninterrupted(self):
            with self.lock:
                return not self.is_interrupted and self._is_idle_state()

        def wait_for_idle(self):
            with self.lock:
                while not self.is_interrupted and not self._is_idle_state():
                    self.idle_cv.wait()
                return not self.is_interrupted

        def interrupt_idle(self):
            with self.lock:
                self.is_interrupted = True
                self.idle_cv.notify()

        def insert_task(self, recv):
            with self.lock:
                if recv is None:
                    self.task_queue.appendleft(recv)
                else:
                    self.task_queue.extend(recv)
                self.tasks_cv.notify()

        def get_task(self):
            with self.lock:
                waited = False
                while len(self.task_queue) == 0:
                    # there's only one consumer of task_queue,
                    # so no stealing of tasks between waits can happen
                    if not waited:
                        waited = True
                        self.is_idle = True
                        self.idle_cv.notify()
                    self.tasks_cv.wait()
                self.is_idle = False
                task = self.task_queue.popleft()
            return task

    def __init__(self, dedicated_task_queue, general_task_queue):
        self.dedicated_task_queue = dedicated_task_queue
        self.general_task_queue = general_task_queue
        self.state = self.MixedReceiverState()
        self.receivers = []
        try:
            self.receivers.append(self.EagerReceiverWorker(self.state, self.dedicated_task_queue))
            self.receivers.append(self.IdleReceiverWorker(self.state, self.general_task_queue))
        except:  # noqa E722
            self.close()
            raise

    def get_recv_queues(self):
        return [self.general_task_queue, self.dedicated_task_queue]

    def get_task(self):
        return self.state.get_task()

    def close(self):
        for receiver in self.receivers:
            receiver.close()
        self.receivers.clear()


class IterableSource:
    """Wraps iterator/generator passed to External Source to enforce
    ES `cycle` policy specified by the user.
    It is a counterpart of _CycleIter/_CycleGenIter wrappers from non parallel mode.
    However due to prefetching in parallel mode `cycle`=raise
    will raise StopIteration in consecutive calls until the new epoch starts
    (i.e. which happens with pipeline.reset call)"""

    def __init__(self, source_desc):
        self.source_desc = source_desc
        self._reset_iter(0)

    def __call__(self, scheduled: ScheduledTask):
        if self.raised_stop_iter:
            # if iterator runs in "raise" mode and a new epoch started
            # (i.e. source context was reset)
            if self.source_desc.cycle == "raise" and self.epoch_start < scheduled.epoch_start:
                self._reset_iter(scheduled.epoch_start)
            else:
                raise StopIteration
        return self._get_next()

    def _reset_iter(self, epoch_start):
        self.iter = IterableSource.get_iter(self.source_desc)
        self.raised_stop_iter = False
        self.epoch_start = epoch_start

    def _get_next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.raised_stop_iter = True
            if self.source_desc.cycle != "quiet" and self.source_desc.cycle is not True:
                raise
            # in quiet mode immediately reset the source and return the first iteration
            self.iter = IterableSource.get_iter(self.source_desc)
            next_iter = next(self.iter)
            # Set the `raised_stop_iter` flag to False after the __next__ call, so that,
            # if it raises StopIteration immediately after the reset, the wrapper can consistently
            # raise StopIteration from then on.
            # The `epoch_start` is not updated - keeping track of it is not necessary
            # in the quiet mode
            self.raised_stop_iter = False
            return next_iter

    @staticmethod
    def get_iter(source_desc):
        source = source_desc.source
        if _is_generator_function(source):
            source = source()
        return iter(source)


class CallableSource:
    def __init__(self, source_desc):
        self.callback = source_desc.source

    def __call__(self, scheduled: ScheduledTask):
        task = scheduled.task
        if task.is_sample_mode():
            data_batch = [self.callback(sample_info) for sample_info in task.sample_range]
        else:
            data_batch = self.callback(*task.batch_args)
        return data_batch


def get_source_from_desc(source_descs):
    if source_descs.kind == SourceKind.CALLABLE:
        return CallableSource(source_descs)
    elif source_descs.kind in (SourceKind.ITERABLE, SourceKind.GENERATOR_FUNC):
        return IterableSource(source_descs)
    raise RuntimeError("Unsupported source type")


class WorkerContext:
    """Initializes structures necessary for a worker process to receive,
    compute and send back tasks."""

    def __init__(self, worker_args: WorkerArgs):
        self.worker_id = worker_args.worker_id
        self.callbacks = self._init_callbacks(
            worker_args.source_descs, worker_args.callback_pickler
        )
        self.result_queue = worker_args.result_queue
        self.general_task_queue = worker_args.general_task_queue
        self.dedicated_task_queue = worker_args.dedicated_task_queue
        shm_chunks = worker_args.shm_chunks
        if worker_args.start_method != "fork":
            setup_socket = worker_args.setup_socket
            # NOTE when making any changes here, make sure to reflect them in the main process,
            # so that it sends handles to objects in the same order they are set to objects here
            self._recv_queue_handles(setup_socket)
            for shm_chunk in shm_chunks:
                shm_chunk.open_shm(reduction.recv_handle(setup_socket))
            setup_socket.shutdown(socket.SHUT_RDWR)
            setup_socket.close()
        self.shm_chunks = {shm_chunk.shm_chunk_id: shm_chunk for shm_chunk in shm_chunks}
        self.task_receiver = None
        self.batch_dispatcher = None
        try:
            self.task_receiver = self._init_task_receiver()
            self.batch_dispatcher = SharedBatchDispatcher(
                worker_args.worker_id,
                worker_args.result_queue,
                self.task_receiver.get_recv_queues(),
            )
        except:  # noqa E722
            self.close()
            raise
        # let the main process know that the worker started and shared resources setup is done
        worker_args.result_queue.put([ShmMessageDesc(self.worker_id, 0, 0, 0, 0)])

    def _init_callbacks(self, source_descs, callback_pickler):
        if callback_pickler is not None:
            for source_desc in source_descs.values():
                source_desc.source = callback_pickler.loads(source_desc.source)
        return {
            context_i: get_source_from_desc(source_desc)
            for context_i, source_desc in source_descs.items()
        }

    def _recv_queue_handles(self, setup_socket):
        self.result_queue.open_shm(reduction.recv_handle(setup_socket))
        if self.general_task_queue is not None:
            self.general_task_queue.open_shm(reduction.recv_handle(setup_socket))
        if self.dedicated_task_queue is not None:
            self.dedicated_task_queue.open_shm(reduction.recv_handle(setup_socket))

    def _init_task_receiver(self):
        assert self.general_task_queue is not None or self.dedicated_task_queue is not None
        if self.dedicated_task_queue is None or self.general_task_queue is None:
            return SimpleQueueTaskReceiver(self.general_task_queue or self.dedicated_task_queue)
        return MixedTaskReceiver(self.dedicated_task_queue, self.general_task_queue)

    def get_task(self) -> Tuple[Optional[ScheduledTask], Optional[BufShmChunk]]:
        """
        Returns scheduled task and shm_chunk where results should be placed
        """
        scheduled_meta = self.task_receiver.get_task()
        if scheduled_meta is None:
            return None, None
        shm_chunk = self.shm_chunks[scheduled_meta.shm_chunk_id]
        scheduled = read_shm_message(shm_chunk, scheduled_meta)
        return scheduled, shm_chunk

    def get_callback(self, scheduled):
        return self.callbacks[scheduled.context_i]

    def dispatch(self, processed: _WorkerProcessingResult):
        return self.batch_dispatcher.append(processed)

    def close(self):
        if self.batch_dispatcher is not None:
            self.task_receiver.close()
        if self.task_receiver is not None:
            self.batch_dispatcher.close()


def worker(worker_args: WorkerArgs):
    """Entry point of a worker process.

    Computes minibatches in the main thread.
    """
    worker_context = WorkerContext(worker_args)
    try:
        while True:
            scheduled, shm_chunk = worker_context.get_task()
            if scheduled is None:
                break
            callback = worker_context.get_callback(scheduled)
            try:
                data_batch = callback(scheduled)
                for sample in data_batch:
                    assert_valid_data_type(sample)
            except Exception as exception:
                tb_str = traceback.format_exc()
                processed = _WorkerProcessingResult.failed(scheduled, shm_chunk, exception, tb_str)
            else:
                processed = _WorkerProcessingResult.done(scheduled, shm_chunk, data_batch)
            worker_context.dispatch(processed)
    finally:
        worker_context.close()
