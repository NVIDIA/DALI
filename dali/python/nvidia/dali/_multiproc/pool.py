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

from typing import List, Tuple, Any, Optional
import os
import socket
import threading
import warnings
import multiprocessing
import copy
from collections import deque
from nvidia.dali import backend as _b
from nvidia.dali import pickling
from nvidia.dali._multiproc import shared_mem
from nvidia.dali._utils.external_source_impl import SourceDescription, SourceKind
from nvidia.dali._multiproc.worker import worker
from nvidia.dali._multiproc.messages import BufShmChunkMeta, ScheduledTask, BatchArgs, WorkerArgs
from nvidia.dali._multiproc.shared_batch import deserialize_batch, import_numpy, read_shm_message, \
    BufShmChunk, SharedBatchWriter, write_shm_message, _align_up as align_up
from nvidia.dali._multiproc.shared_queue import ShmQueue, Dispatcher, DispatcherWorker


class SharedMemBuffer:

    def __init__(self, get_next_shm_id, queue_depth, initial_chunk_size, num_minibatches):
        if queue_depth < 1:
            raise RuntimeError("Prefetch queue must have at least one element")
        if initial_chunk_size <= 0:
            raise RuntimeError("Buffer chunk capacity must be a positive integer")
        self.queue_depth = queue_depth
        self.initial_chunk_size = align_up(initial_chunk_size, SharedBatchWriter.BUFFER_ALIGNMENT)
        self.num_minibatches = num_minibatches
        self.chunks_ids_by_pos = [
            [get_next_shm_id() for _ in range(self.num_minibatches)]
            for _ in range(self.queue_depth)]
        self.chunks_ids = [chunk_id for dest_buf in self.chunks_ids_by_pos for chunk_id in dest_buf]
        self._chunks_by_id = {
            chunk_id: BufShmChunk.allocate(chunk_id, self.initial_chunk_size)
            for chunk_id in self.chunks_ids}

    def seal(self):
        for chunk in self._chunks_by_id.values():
            chunk.shm_chunk.seal()

    def get_chunk_by_id(self, shm_chunk_id):
        return self._chunks_by_id[shm_chunk_id]

    def get_chunk_by_dest(self, dest_i, minibatch_i):
        chunk_id = self.chunks_ids_by_pos[dest_i][minibatch_i]
        return self._chunks_by_id[chunk_id]

    def get_chunks(self):
        return [self._chunks_by_id[chunk_id] for chunk_id in self.chunks_ids]

    @property
    def num_chunks(self):
        return len(self.chunks_ids)


class CallbackSetup:

    def __init__(self, source_desc : SourceDescription, shm_buffer : SharedMemBuffer, dedicated_worker : Optional[int]):
        self.source_desc = source_desc
        self.shm_buffer = shm_buffer
        self.dedicated_worker = dedicated_worker


class CallbackContext:
    """Counterpart of worker.py:CallbackContext, for every callback that can be run
in the workers stores shared memory chunks that are used to passed callback results, keeps track
of what tasks has been scheduled to run in the workers and what partial results has been
received so far.
"""

    def __init__(self, setup : CallbackSetup):
        self.setup = setup
        self.scheduled_i = 0  # internal source of scheduled ids for tasks
        self.epoch_start = self.scheduled_i
        self.partially_received = {}
        self.scheduled = {}
        self.iter_failed = {}
        self.tasks_queue = deque()
        self.epoch_synced = True

    def reset(self):
        """Invalidates all batches pending in `tasks_queue`, marks the need to sync the epoch
        before chunks occupied by `tasks_queue` batches can be be reused for prefetching in the next epoch"""
        self.scheduled_i += 1
        self.epoch_start = self.scheduled_i
        self.epoch_synced = False

    def push_scheduled(self, work_batch):
        self.scheduled_i += 1
        self.partially_received[self.scheduled_i] = {}
        self.scheduled[self.scheduled_i] = work_batch
        self.tasks_queue.append(self.scheduled_i)
        return self.scheduled_i

    def clear_scheduled(self, scheduled_i):
        del self.partially_received[scheduled_i]
        del self.scheduled[scheduled_i]
        if scheduled_i in self.iter_failed:
            del self.iter_failed[scheduled_i]

    def pop_scheduled(self):
        return self.tasks_queue.popleft()

    def handle_error(self, batch_i):
        """Check if given batch reported an error and raise it"""
        exception = None
        try:
            if batch_i in self.iter_failed:
                exception, traceback_str = self.iter_failed[batch_i]
                self.clear_scheduled(batch_i)
                if isinstance(exception, StopIteration):
                    raise exception
                else:
                    # Raise new exception propagating the traceback from worker thread as error
                    # message, originating from original exception
                    raise Exception(
                        "\n\nException traceback received from worker thread:\n\n" + traceback_str) from exception
        finally:
            # Fix circular reference problem on StopIteration - the exception contains reference to the
            # traceback that refers a frame that contains local variables and among them the exception.
            # This traceback is then chained into exceptions reraised along the way
            # (eventually at the pipeline level) which in effect introduces a reference to the pipline
            # that would be only removed after garbage collection round, delaying finalization of the pool
            del exception

    def is_error(self, scheduled_i):
        return scheduled_i in self.iter_failed

    def process_task(self, shm_chunk : shared_mem.SharedMem, completed_task):
        scheduled_i = completed_task.scheduled_i
        if completed_task.is_failed():
            if not self.is_error(scheduled_i):
                self.set_error(completed_task)
            self.mark_received(completed_task)
        else:
            if self.is_stale(scheduled_i) or self.is_error(scheduled_i):
                self.mark_received(completed_task)
            else:
                self.receive_chunk(shm_chunk, completed_task)

    def mark_received(self, completed_task):
        scheduled_i = completed_task.scheduled_i
        minibatch_i = completed_task.minibatch_i
        self.partially_received[scheduled_i][minibatch_i] = None

    def set_error(self, completed_task):
        scheduled_i = completed_task.scheduled_i
        self.iter_failed[scheduled_i] = (completed_task.exception, completed_task.traceback_str)

    def is_stale(self, scheduled_i):
        return scheduled_i < self.epoch_start

    def is_not_received(self, scheduled_i):
        """Check if we didn't receive all results for given tasks and scheduled_i"""
        return len(self.partially_received[scheduled_i]) < self.scheduled[scheduled_i].num_minibatches

    def coalesce_received(self, scheduled_i):
        num_minibatches = self.scheduled[scheduled_i].num_minibatches
        minibatches = self.partially_received[scheduled_i]
        if num_minibatches == 1:
            return minibatches[0]
        return [sample for minibatch_i in range(num_minibatches) for sample in minibatches[minibatch_i]]

    def take_processed(self, scheduled_i):
        """Return the full batch, mark it as cleared and consumed"""
        batch = self.coalesce_received(scheduled_i)
        self.clear_scheduled(scheduled_i)
        return batch

    def receive_chunk(self, shm_chunk, completed_task):
        """Obtain the chunk and decode it, add to partially gathered result"""
        scheduled_i = completed_task.scheduled_i
        minibatch_i = completed_task.minibatch_i
        worker_batch = deserialize_batch(shm_chunk, completed_task.batch_meta)
        self.partially_received[scheduled_i][minibatch_i] = worker_batch

    @property
    def queue_depth(self):
        return self.setup.shm_buffer.queue_depth


class WorkBatch:

    @classmethod
    def sample_mode(cls, samples_args):
        if not samples_args:
            raise RuntimeError("Cannot schedule empty batch")
        return cls(samples_args=samples_args)

    @classmethod
    def batch_mode(cls, batch_arg):
        return cls(batch_arg=batch_arg)

    def __init__(self, samples_args=None, batch_arg=None):
        self.samples_args = samples_args
        self.batch_arg = batch_arg
        self.num_minibatches = None
        assert ((self.samples_args is None) != (self.batch_arg is None))

    def is_sample_mode(self):
        return self.samples_args is not None

    def split_work(self, num_samples_split=None):
        if self.is_sample_mode():
            return self._split_samples(num_samples_split)
        self.num_minibatches = 1
        return [BatchArgs.batch_mode(self.batch_arg)]

    def _split_samples(self, num_minibatches):
        tasks = self.samples_args
        tasks_num = len(tasks)
        chunk_size = tasks_num // num_minibatches
        remainder = tasks_num % num_minibatches
        queued_no = 0
        minibatches = []
        for minibatch_i in range(num_minibatches):
            worker_chunk = chunk_size + (minibatch_i < remainder)
            if worker_chunk == 0:
                break
            samples = BatchArgs.sample_mode(minibatch_i, tasks[queued_no: queued_no + worker_chunk])
            minibatches.append(samples)
            queued_no += worker_chunk
        self.num_minibatches = num_minibatches
        return minibatches


class ProcPool:
    """Runs pool of worker processes, stores pipes and sockets used to communicate with the workers,
starts thread keeping track of running processes and initializes communication.
"""
    class WorkerContext:

        def __init__(self, sources_desc : SourceDescription, dedicated_task_queue : Optional[ShmQueue], shm_chunks : List[BufShmChunk], callback_pickler):
            self.sources_desc = sources_desc
            self.dedicated_task_queue = dedicated_task_queue
            self.shm_chunks = shm_chunks
            self.callback_pickler = callback_pickler

    def __init__(self, mp, workers_contexts : List[WorkerContext], res_queue : ShmQueue, general_task_queue : Optional[ShmQueue]):
        start_method = mp.get_start_method()
        if not workers_contexts:
            raise RuntimeError("Cannot start a pool with no workers")
        if start_method == 'fork' and _b.HasCudaContext():
            raise RuntimeError(
                "Error when starting Python worker threads for DALI parallel External Source. "
                "Cannot fork a process when there is a CUDA context already bound to the process. "
                "CUDA context is acquired during ``Pipeline.build()``, or can be acquired by another "
                "library that interacts with CUDA, for example a DL framework creating CUDA tensors."
                "If you are trying to build multiple pipelines that use Python workers, you will need to "
                "call ``start_py_workers`` method on all of them before calling ``build`` method of any pipeline "
                "to start Python workers before CUDA context is acquired by ``build`` or other CUDA operation."
                "Alternatively you can change Python workers starting method from ``fork`` to ``spawn`` "
                "(see DALI Pipeline's ``py_start_method`` option for details). ")
        self._workers_contexts = workers_contexts
        self._res_queue = res_queue
        self._general_task_dispatcher = None
        self._tracker = None
        self._processes = []
        write_socks = []
        try:
            for worker_i, worker_context in enumerate(workers_contexts):
                if start_method == "fork":
                    sock_reader = None
                    worker_shm_chunks = worker_context.shm_chunks
                else:
                    sock_reader, sock_writer = socket.socketpair()
                    write_socks.append(sock_writer)
                    worker_shm_chunks = [BufShmChunkMeta.from_chunk(chunk) for chunk in worker_context.shm_chunks]
                process_context = WorkerArgs(
                    worker_i, start_method, worker_context.sources_desc, worker_shm_chunks, general_task_queue,
                    worker_context.dedicated_task_queue, res_queue, sock_reader, worker_context.callback_pickler)
                process = mp.Process(target=worker, args=(process_context,))
                self._processes.append(process)
            self._start_processes(mp, start_method, write_socks, general_task_queue)
        finally:
            for sock in write_socks:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()

    @classmethod
    def from_callback_setups(cls, setups : List[CallbackSetup], num_workers, start_method="fork", py_callback_pickler=None):
        mp = multiprocessing.get_context(start_method)
        general_sources_buffs = [setup.shm_buffer for setup in setups if setup.dedicated_worker is None]
        if not general_sources_buffs:
            general_task_queue = None
        else:
            general_task_queue = ShmQueue(mp, capacity=sum(
                shm_buffer.num_chunks for shm_buffer in general_sources_buffs))
        res_queue = ShmQueue(mp, capacity=sum(
            setup.shm_buffer.num_chunks for setup in setups))
        worker_contexts = cls.create_worker_contexts(mp, setups, num_workers, start_method, py_callback_pickler)
        # close underlying file descriptors that are not needed anymore once
        # passed to the workers processes
        instance = None
        try:
            instance = cls(mp, worker_contexts, res_queue, general_task_queue)
            for setup in setups:
                setup.shm_buffer.seal()
            if general_task_queue is not None:
                general_task_queue.seal()
            res_queue.seal()
            for worker_context in worker_contexts:
                if worker_context.dedicated_task_queue is not None:
                    worker_context.dedicated_task_queue.seal()
            return instance
        except:
            if instance is not None:
                instance.close()
            raise

    @classmethod
    def create_worker_contexts(cls, mp, setups : List[CallbackSetup], num_workers, start_method, py_callback_pickler):
        if start_method == "fork":
            callback_pickler = None
            sources_desc = [setup.source_desc for setup in setups]
        else:
            callback_pickler = pickling._CustomPickler.create(py_callback_pickler)
            sources_desc = [copy.copy(setup.source_desc) for setup in setups]
            for source_desc in sources_desc:
                source_desc.source = callback_pickler.dumps(source_desc.source)
        dedicated_workers = [setup.dedicated_worker for setup in setups if setup.dedicated_worker is not None]
        worker_contexts = []
        for worker_id in range(num_workers):
            worker_sources = {
                source_i : source_desc
                for source_i, (source_desc, setup) in enumerate(zip(sources_desc, setups))
                if setup.dedicated_worker is None or setup.dedicated_worker == worker_id}
            worker_shm_chunks = [
                shm_chunk for source_i, setup
                in enumerate(setups) if source_i in worker_sources
                for shm_chunk in setup.shm_buffer.get_chunks()]
            if worker_id not in dedicated_workers:
                dedicated_task_queue = None
            else:
                dedicated_task_queue = ShmQueue(mp, capacity=sum(
                    setup.shm_buffer.num_chunks for setup in setups
                    if setup.dedicated_worker == worker_id))
            worker_context = cls.WorkerContext(
                worker_sources, dedicated_task_queue, worker_shm_chunks, callback_pickler)
            worker_contexts.append(worker_context)
        return worker_contexts

    @property
    def num_workers(self):
        return len(self._workers_contexts)

    def pids(self):
        """Get pids of the processes started by this pool.
        """
        return [proc.pid for proc in self._processes]

    def close(self):
        if self._tracker is None:
            return
        self._tracker.close()

    def wait_for_res(self):
        return self._res_queue.get(None)

    def send(self, tasks : List[Tuple[BufShmChunk, Any]], dedicated_worker):
        if dedicated_worker is None:
            return self._general_task_dispatcher.extend(tasks)
        msgs = TaskDispatcherWorker.write_to_shm(tasks)
        if self._workers_contexts[dedicated_worker].dedicated_task_queue.put(msgs) is None:
            raise RuntimeError("Sending task for worker {} failed".format(dedicated_worker))

    def _sync_initialized_workers(self):
        workers_received = []
        while len(workers_received) < self.num_workers:
            shm_msgs = self.wait_for_res()
            if shm_msgs is None:
                raise RuntimeError("Workers initialization failed")
            synced_ids = [shm_msg.worker_id for shm_msg in shm_msgs]
            assert (0 <= worker_id <= self.num_workers and worker_id in workers_received for worker_id in synced_ids)
            workers_received.extend(synced_ids)

    def _send_queues(self, socks, general_task_queue):
        pid = os.getppid()
        all_worker_queues = [self._res_queue]
        if general_task_queue is not None:
            all_worker_queues.append(general_task_queue)
        for queue in all_worker_queues:
            for sock in socks:
                multiprocessing.reduction.send_handle(sock, queue.shm.handle, pid)
        for sock, worker_context in zip(socks, self._workers_contexts):
            if worker_context.dedicated_task_queue is not None:
                multiprocessing.reduction.send_handle(
                    sock, worker_context.dedicated_task_queue.shm.handle, pid)

    def _send_shm(self, socks):
        pid = os.getppid()
        for sock, worker_context in zip(socks, self._workers_contexts):
            for shm_chunk in worker_context.shm_chunks:
                multiprocessing.reduction.send_handle(sock, shm_chunk.shm_chunk.handle, pid)

    def _init_dispatcher(self, queue):
        dispatcher = Dispatcher(queue)
        worker = TaskDispatcherWorker(dispatcher.pending_cv, dispatcher.pending, queue, self._tracker)
        dispatcher.start_thread(worker)
        return dispatcher

    def _start_processes(self, mp, start_method, write_socks, general_task_queue):
        try:
            for process in self._processes:
                process.start()
            self._tracker = Tracker(mp)
            if general_task_queue is not None:
                self._general_task_dispatcher = self._init_dispatcher(general_task_queue)
            dedicated_task_queues = [
                worker_context.dedicated_task_queue
                for worker_context in self._workers_contexts
                if worker_context.dedicated_task_queue is not None]
            self._tracker.start_thread(self._processes, self._general_task_dispatcher, dedicated_task_queues, self._res_queue)
            if start_method != "fork":
                self._send_queues(write_socks, general_task_queue)
                self._send_shm(write_socks)
                self._sync_initialized_workers()
        except:
            if self._tracker is None or not self._tracker.close():
                if self._general_task_dispatcher is not None:
                    # dispatcher starts separate thread so it needs to be closed
                    # res_queue and dedicated queues can be simply garbage collected
                    # as the workers are forcefully terminated and main thread is here, so
                    # it is not blocked on receiving
                    self._general_task_dispatcher.stop_thread()
                for proc in self._processes:
                    if proc.is_alive():
                        proc.terminate()
                for proc in self._processes:
                    if proc.pid is not None:
                        proc.join()
            raise


def tracker_thread(processes, tracker_interruption_pipe, general_task_dispatcher, dedicated_task_queues, res_queue):
    """Observer thread for ProcPool used for joining processes and distributing
    stop signal (`None` message).

    Parameters
    ----------
    `processes` : List of multiprocessing.Process
        Worker processes.
    `tracker_pipe` : Pipe
        Read pipe for communicating stop to tracker_thread from main process.
    `main_thread_pipe` : Pipe
        Pipe where stop will be sent for main thread.
   [TODO]
    """
    exit_gently = True
    try:
        ps = {p.sentinel: p for p in processes}
        listen_for = list(ps.keys()) + [tracker_interruption_pipe]
        # Once one process exits stop the whole group (gracefully if possible)
        while True:
            sentinels = multiprocessing.connection.wait(listen_for)
            proc_sentinels = [s for s in sentinels if s != tracker_interruption_pipe]
            if tracker_interruption_pipe in sentinels:
                break
            if any(ps[sentinel].exitcode is not None for sentinel in proc_sentinels):
                exit_gently = False
                break
    except:
        exit_gently = False
        raise
    finally:
        res_queue.close()  # main thread may be blocked on reading the queue
        if exit_gently:
            # try to close task queues and notify waiting processes, so that they can
            # cleanup and exit. Unfortunately if all workers exited abraptly when waiting,
            # an attempt to notify workers with multiprocessing.Condition might lead to deadlock on
            # underlying semaphore. For this reason it is done only if none of the workers reported
            # to have exited.
            for queue in dedicated_task_queues:
                queue.close()
            if general_task_dispatcher is not None:
                general_task_dispatcher.close()
            for proc in processes:
                proc.join(1)
        if general_task_dispatcher is not None:
            general_task_dispatcher.stop_thread()
        for proc in processes:
            if proc.exitcode is None:
                proc.terminate()
                proc.join()


class Tracker:

    def __init__(self, mp):
        self.lock = threading.Lock()
        self.is_interrupted = False
        self.interruption_pipe, self.interrupt_pipe = mp.Pipe(duplex=False)
        self.thread = None

    def interrupt(self):
        # Can be called from dispatcher workers or main thread
        with self.lock:
            if self.is_interrupted:
                return
            self.interrupt_pipe.send(None)
            self.is_interrupted = True

    def close(self):
        # called from main thread only
        if self.thread is not None:
            self.interrupt()
            self.thread.join()
            self.thread = None
            return True

    def start_thread(self, processes, general_task_dispatcher, dedicated_task_queues, res_queue):
        thread = threading.Thread(
            target=tracker_thread,
            args=(processes, self.interruption_pipe, general_task_dispatcher, dedicated_task_queues, res_queue),
            daemon=True)
        thread.start()
        self.thread = thread


class TaskDispatcherWorker(DispatcherWorker):

    def __init__(self, pending_cv, pending, queue, tracker):
        super().__init__(pending_cv, pending, queue)
        self.tracker = tracker

    @staticmethod
    def write_to_shm(msgs : List[Tuple[BufShmChunk, Any]]):
        return [
            write_shm_message(-1, shm_chunk, msg, 0, resize=False)
            for shm_chunk, msg in msgs]

    def close(self):
        self.tracker.interrupt()

    def serialize_msgs(self, msgs):
        return self.write_to_shm(msgs)


class WorkerPool:
    """"Combines worker processes pool with callback contexts, can be used to schedule batches
    to be run on the workers and to receive resulting batches from the workers."""

    def __init__(self, contexts : List[CallbackContext], pool : ProcPool):
        """
        Parameters
        ----------
        `contexts` : [TODO]
            Number of callabacks that can be run in the workers, each callback will have separate
            context with dedicated shared memory pool.
        `pool` : ProcPool
            ProcPool instance enabling basic communication with worker processes.
        """
        self.contexts = contexts
        self.pool = pool
        # shm chunks ids are unique across the pool, so we can identify the callback context
        # by the id of shm chunk with the data
        self.shm_chunks_contexts = {
             chunk_id: context
             for context in self.contexts
             for chunk_id in context.setup.shm_buffer.chunks_ids}

    @classmethod
    def from_groups(
            cls, groups, keep_alive_queue_size, start_method="fork", num_workers=1,
            min_shm_chunk_size=1024 * 1024, py_callback_pickler=None):
        """Creates new WorkerPool instance for given list of ExternalSource groups.

        Parameters
        ----------
        `groups` : _ExternalSourceGroup list
            List of external source groups.
        `keep_alive_queue_size` : int
            Number of the most recently produced batches whose underlying shared memory should
            remain untouched (because they might still be referenced further in the pipeline).
            Note that the actual number of simultaneously kept batches will be greater by the length
            of parallel external source prefetching queue which is at least one.
        `start_method` : str
            Method of starting worker processes, either fork or spawn.
        `num_workers` : int
            Number of workers to be created in ProcPool.
        `min_shm_chunk_size` : int
            Minimal initial size of each shared memory chunk, NOTE it must be enough to accommodate serialized `ScheduledTask` instance.
        """
        import_numpy()
        if len(groups) == 0:
            raise RuntimeError("Cannot create Python workers pool because there are no callbacks provided")
        if num_workers < 1:
            raise RuntimeError("Number of Python workers for parallel ExternalSource must be positive")
        if any(group.source_desc.kind == SourceKind.CALLABLE and not group.source_desc.has_inputs for group in groups):
            raise RuntimeError("Callable `source` of parallel external source must accept argument")
        if any(group.source_desc.kind != SourceKind.CALLABLE and not group.batch for group in groups):
            raise RuntimeError("Parallel external source with iterator or generator must run in batch mode")
        # iterators and generators are stateful and run always in the same dedicated worker
        num_cbs_dedicated = sum(cls.needs_dedicated_worker(group) for group in groups)
        num_cbs_general = len(groups) - num_cbs_dedicated
        if num_cbs_general == 0:
            if num_workers > num_cbs_dedicated:
                warnings.warn(
                "There will be run only {} python worker{}, even though {} were specified to run. "
                "This may happen when all your ExternalSource callbacks are stateful (for instance they are iterators) and "
                "there is less of them than ```py_num_workers```".format(
                    num_cbs_dedicated, "s" if num_cbs_dedicated > 1 else "", num_workers), Warning)
                num_workers = num_cbs_dedicated
        sources_desc = [group.source_desc for group in groups]
        dedicated_workers = cls.assign_excl_workers(groups, num_workers)
        shm_buffers = cls.allocate_buffers(groups, keep_alive_queue_size, num_workers, min_shm_chunk_size)
        setups = [
            CallbackSetup(source_desc, shm_buffer, dedicated_worker)
            for source_desc, shm_buffer, dedicated_worker
            in zip(sources_desc, shm_buffers, dedicated_workers)]
        contexts = [CallbackContext(setup) for setup in setups]
        pool = None
        try:
            pool = ProcPool.from_callback_setups(setups, num_workers, start_method, py_callback_pickler)
            return cls(contexts, pool)
        except:
            if pool is not None:
                pool.close()
            raise

    @classmethod
    def assign_excl_workers(cls, groups, num_workers):
        next_excl_worker = num_workers - 1
        def get_next_excl_worker():
            nonlocal next_excl_worker
            next_excl_worker = (next_excl_worker + 1) % num_workers
            return next_excl_worker
        return [get_next_excl_worker() if cls.needs_dedicated_worker(group) else None for group in groups]

    @classmethod
    def allocate_buffers(cls, groups, keep_alive_queue_size, num_workers, min_shm_chunk_size):
        shm_count = 0
        def get_next_shm_id():
            nonlocal shm_count
            shm_count += 1
            return shm_count
        return [
            SharedMemBuffer(
                get_next_shm_id,
                keep_alive_queue_size + group.prefetch_queue_depth,
                cls.get_initial_shm_chunk(group.bytes_per_minibatch_hint, min_shm_chunk_size),
                1 if group.batch else num_workers
            ) for group in groups]

    @classmethod
    def needs_dedicated_worker(cls, group):
        return group.source_desc.kind != SourceKind.CALLABLE

    @classmethod
    def get_initial_shm_chunk(cls, bytes_per_minibatch_hint, min_shm_chunk_size):
        if bytes_per_minibatch_hint is None or bytes_per_minibatch_hint < min_shm_chunk_size:
            return min_shm_chunk_size
        return bytes_per_minibatch_hint

    def schedule_batch(self, context_i, dst_chunk_i, work_batch : WorkBatch):
        """Distribute `tasks` among workers to run them by calling `context_i`th callaback

        Parameters
        ----------
        `context_i` : int
            Specifies which callback will be used to run the task, it must be the index corresponding
            to the order of callbacks passed when constructing WorkerPool.
        `dst_chunk_i` : int
            Index of the memory chunk in the circular buffer to store the output in
        `work_batch` : list of (nvidia.dali.types.SampleInfo,) [TODO]
            You can think of resulting batch as [callback(*task) for task in tasks] with the exception that
            callbacks will be run in parallel.
        """
        context = self.contexts[context_i]
        if not context.epoch_synced:
            self.sync_and_discard(context_i)
            context.epoch_synced = True
        if context.iter_failed:
            # there is no point in scheduling anything for the context that has reached the end of data
            # or failed with error, once user receives batch that raised exception they should reset
            # the context before scheduling new tasks
            return
        scheduled_i = context.push_scheduled(work_batch)
        self._distribute(context_i, scheduled_i, dst_chunk_i, work_batch)

    def _distribute(self, context_i, scheduled_i, dst_chunk_i, work_batch : WorkBatch):
        context = self.contexts[context_i]
        minibatches = work_batch.split_work(self.pool.num_workers)
        scheduled_tasks = [(
            context.setup.shm_buffer.get_chunk_by_dest(dst_chunk_i, minibatch_i),
            ScheduledTask(context_i, scheduled_i, context.epoch_start, task))
            for minibatch_i, task in enumerate(minibatches)
        ]
        dedicated_worker = context.setup.dedicated_worker
        self.pool.send(scheduled_tasks, dedicated_worker)

    def sync_and_discard(self, context_i):
        context = self.contexts[context_i]
        while context.tasks_queue:
            try:
                self.receive_batch(context_i)
            except StopIteration:
                pass

    def receive_batch(self, context_i):
        """Returns the next produced batch (in the order of schedule_batch calls) for the
        ``context_i``th callaback.

        Parameters
        ----------
        `context_i` : int
            Specifies which callback you want the results from, ordering corresponds to the order of
            callbacks passed when constructing the pool.
        """
        context = self.contexts[context_i]
        assert len(context.scheduled) > 0, "No task has been scheduled"
        scheduled_i = context.pop_scheduled()
        while context.is_not_received(scheduled_i):
            self._receive_chunk()
        context.handle_error(scheduled_i)
        if context.is_stale(scheduled_i):
            context.clear_scheduled(scheduled_i)
            return None
        res = context.take_processed(scheduled_i)
        return res

    def _receive_chunk(self):
        completed_tasks_meta = self.pool.wait_for_res()
        if completed_tasks_meta is None:
            raise RuntimeError("Worker data receiving interrupted")
        for completed_task_meta in completed_tasks_meta:
            context = self.shm_chunks_contexts[completed_task_meta.shm_chunk_id]
            shm = context.setup.shm_buffer.get_chunk_by_id(completed_task_meta.shm_chunk_id)
            shm_chunk = shm.shm_chunk
            completed_task = read_shm_message(shm_chunk, completed_task_meta)
            context.process_task(shm_chunk, completed_task)

    def pids(self):
        """Get pids of the processes started by this pool.
        """
        return self.pool.pids()

    def reset(self):
        for context in self.contexts:
            context.reset()

    def reset_context(self, context_i):
        self.contexts[context_i].reset()

    def close(self):
        self.pool.close()
