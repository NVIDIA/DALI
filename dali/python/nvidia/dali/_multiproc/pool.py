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

from typing import Tuple, List, Optional
import os
import select
import socket
import threading
import warnings
import multiprocessing
from collections import deque
from nvidia.dali import backend as _b
from nvidia.dali import pickling
from nvidia.dali._utils.external_source_impl import SourceKind
from nvidia.dali._multiproc.worker import worker
from nvidia.dali._multiproc.messages import ScheduledTask, BatchArgs, ProcessContext
from nvidia.dali._multiproc.shared_batch import deserialize_batch, import_numpy
from nvidia.dali._multiproc import shared_mem


class SharedMemBuffer:
    class SharedMemChunk:
        def __init__(self, shm_chunk_id, shm_chunk):
            self.shm_chunk_id = shm_chunk_id
            self.shm_chunk = shm_chunk

        @classmethod
        def allocate(cls, shm_chunk_id, initial_chunk_size):
            return cls(shm_chunk_id, shared_mem.SharedMem.allocate(initial_chunk_size))

    def __init__(self, context_i, queue_depth, initial_chunk_size, num_minibatches):
        if queue_depth < 1:
            raise RuntimeError("Prefetch queue must have at least one element")
        self.context_i = context_i
        self.queue_depth = queue_depth
        self.initial_chunk_size = initial_chunk_size # TODO differentiate initial_chunk for a whole batch vs minibatch?
        self.num_minibatches = num_minibatches
        self.chunks_ids = [
            self.get_chunk_id(queue_i, minibatch_i)
            for queue_i in range(self.queue_depth)
            for minibatch_i in range(self.num_minibatches)]
        self._chunks_by_id = None

    def allocate(self):
        if self._chunks_by_id is not None:
            raise RuntimeError("Shared memory buffer has already been initialized")
        self._chunks_by_id = {
            chunk_id: self.SharedMemChunk.allocate(chunk_id, self.initial_chunk_size)
            for chunk_id in self.chunks_ids}

    def seal(self):
        for chunk in self._chunks_by_id.values():
            chunk.shm_chunk.seal()

    def get_shm_chunk(self, completed_task):
        shm_chunk = self._chunks_by_id[completed_task.shm_chunk_id].shm_chunk
        if shm_chunk.capacity != completed_task.shm_capacity:
            shm_chunk.resize(completed_task.shm_capacity, trunc=False)
        return shm_chunk

    def get_chunk_by_dest(self, dest_i, minibatch_i):
        return self._chunks_by_id[self.get_chunk_id(dest_i, minibatch_i)]

    def get_chunk_id(self, dest_i, minibatch_i):
        return 'shm_{}_{}_{}'.format(self.context_i, dest_i, minibatch_i)

    def get_chunks(self):
        return [self._chunks_by_id[chunk_id] for chunk_id in self.chunks_ids]


class CallbackContext:
    """Counterpart of worker.py:CallbackContext, for every callback that can be run
in the workers stores shared memory chunks that are used to passed callback results, keeps track
of what tasks has been scheduled to run in the workers and what partial results has been
received so far.
"""

    def __init__(self, source_desc, exclusive_worker, shm_buffer):
        self.source_desc = source_desc
        self.exclusive_worker = exclusive_worker
        self.shm_buffer = shm_buffer
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

    def process_task(self, completed_task):
        scheduled_i = completed_task.scheduled_i
        if completed_task.is_failed():
            if not self.is_error(scheduled_i):
                self.set_error(completed_task)
            self.mark_received(completed_task)
        else:
            self.adjust_shm(completed_task)
            if self.is_stale(scheduled_i) or self.is_error(scheduled_i):
                self.mark_received(completed_task)
            else:
                self.receive_chunk(completed_task)

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

    def receive_chunk(self, completed_task):
        """Obtain the chunk and decode it, add to partially gathered result"""
        scheduled_i = completed_task.scheduled_i
        minibatch_i = completed_task.minibatch_i
        shm_chunk = self.shm_buffer.get_shm_chunk(completed_task)
        worker_batch = deserialize_batch(shm_chunk, completed_task.batch_meta)
        self.partially_received[scheduled_i][minibatch_i] = worker_batch

    def adjust_shm(self, completed_task):
        # get_shm_chunk remaps shm if its capacity changed
        self.shm_buffer.get_shm_chunk(completed_task)

    @property
    def queue_depth(self):
        return self.shm_buffer.queue_depth


class WorkBatch:

    @classmethod
    def sample_mode(cls, samples_args):
        samples_args = list(samples_args)
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

    def __init__(
            self, contexts, num_workers, start_method="fork", py_callback_pickler=None):
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
        mp = multiprocessing.get_context(start_method)
        if start_method != "spawn":
            callback_pickler = None
        else:
            callback_pickler = pickling._CustomPickler.create(py_callback_pickler)
        if num_workers < 1:
            raise RuntimeError("Cannot start a pool with no workers")
        self._num_workers = num_workers
        self._processes = []
        self._res_pipes = []
        self._workers_available_contexts = []
        serializable_sources_desc = [context.source_desc for context in contexts]
        if callback_pickler is not None:
            for source_desc in serializable_sources_desc:
                source_desc.source = callback_pickler.dumps(source_desc.source)
        if any(context.exclusive_worker is None for context in contexts):
            general_tasks_pipe_r, general_tasks_pipe_w = mp.Pipe(duplex=False)
            general_tasks_read_lock = mp.Lock()
        else:
            general_tasks_pipe_r, general_tasks_pipe_w, general_tasks_read_lock = None, None, None
        exclusive_tasks_pipes = {}
        write_socks = []
        for worker_id in range(num_workers):
            res_r, res_w = mp.Pipe(duplex=False)
            sock_reader, sock_writer = socket.socketpair()
            sources_desc = {
                context_i : source_desc
                for context_i, (context, source_desc) in enumerate(zip(contexts, serializable_sources_desc))
                if context.exclusive_worker is None or context.exclusive_worker == worker_id}
            if any(context.exclusive_worker == worker_id for context in contexts):
                exclusive_task_r, exclusive_tasks_pipes[worker_id] = mp.Pipe(duplex=False)
            else:
                exclusive_task_r = None
            process_context = ProcessContext(
                worker_id, sources_desc, contexts, general_tasks_pipe_r, general_tasks_read_lock,
                exclusive_task_r, res_w, sock_reader, callback_pickler)
            process = mp.Process(target=worker, args=(process_context,))
            write_socks.append(sock_writer)
            self._res_pipes.append(res_r)
            self._processes.append(process)
            self._workers_available_contexts.append(process_context.sources_desc.keys())
        self._tracker_thread = None
        self._is_receiving_closed = False
        self.task_dispatcher = None
        self._init_interruption_pipes(mp)
        self._start_processes(general_tasks_pipe_w, exclusive_tasks_pipes)
        self._initialize_shm(contexts, write_socks)

    @property
    def num_workers(self):
        return self._num_workers

    def pids(self):
        """Get pids of the processes started by this pool.
        """
        return [proc.pid for proc in self._processes]

    def close(self):
        if self._tracker_thread is None:
            return
        self._close_receiving_pipes()
        try:
            self.interrupt_tracker_pipe.send(None)
        except BrokenPipeError:
            # workers already exited, tracker_thread finished its task and exited and closed the pipe
            pass
        finally:
            self._tracker_thread.join()  # TODO it must join dispatcher thread
            self._tracker_thread = None
            self.task_dispatcher._dispatcher_thread.join()

    def wait_for_res(self):
        if self._is_receiving_closed:
            return None
        ready_workers = multiprocessing.connection.wait(self._get_recv_pipes())
        if self.receiving_interruption_pipe in ready_workers:
            self._close_receiving_pipes()
            return None
        return [worker_pipe.recv() for worker_pipe in ready_workers]

    def _close_receiving_pipes(self):
        self._is_receiving_closed = True
        for pipe in self._res_pipes:
            pipe.close()

    def _get_recv_pipes(self):
        """Return all pipes with incoming communication.

        Note: One pipe is from tracking process that may send `None` in case of shutdown.
        """
        return self._res_pipes + [self.receiving_interruption_pipe]

    def _sync_initialized_workers(self, workers_ids):
        workers_received = []
        while len(workers_received) < self.num_workers:
            workers_ids = self.wait_for_res()
            if workers_ids is None or any(worker_id not in workers_ids or worker_id in workers_received for worker_id in workers_ids):
                self.close()
                raise RuntimeError("Initialization of workers failed")
            workers_received.extend(workers_ids)

    def _initialize_shm(self, contexts, socks):
        pid = os.getppid()
        for context in contexts:
            context.shm_buffer.allocate()
        sync_workers = []
        for worker_id, (worker_available_contexts, sock) in enumerate(zip(self._workers_available_contexts, socks)):
            for context_i, context in enumerate(contexts):
                if context_i not in worker_available_contexts:
                    continue
                for chunk in context.shm_buffer.get_chunks():
                    multiprocessing.reduction.send_handle(sock, chunk.shm_chunk.handle, pid)
            sync_workers.append(worker_id)
        self._sync_initialized_workers(sync_workers)
        for context in contexts:
            context.shm_buffer.seal()
        for sock in socks:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()

    def _init_interruption_pipes(self, mp):
        # from tracker thread to dispatcher thread (when tracker cleans up)
        self.dispatcher_interruption_pipe, self.interrupt_dispatching_pipe = mp.Pipe(duplex=False)
        # from main thread to tracker thread (to start closing the pool)
        self.tracker_interruption_pipe, self.interrupt_tracker_pipe = mp.Pipe(duplex=False)
        # from dispatcher thread to main thread to unblock main thread if it waits for data from workers
        self.receiving_interruption_pipe, self.interrupt_receiving_pipe = mp.Pipe(duplex=False)

    def _start_processes(self, general_tasks_pipe_w, exclusive_tasks_pipes):
        try:
            for process in self._processes:
                process.start()
            self.task_dispatcher = TaskDispatcher()
            def interrupt_dispatcher():
                self.interrupt_dispatching_pipe.send(None)
                self.task_dispatcher.push_tasks([None])
            self.task_dispatcher.start_thread(
                general_tasks_pipe_w, exclusive_tasks_pipes, self.dispatcher_interruption_pipe,
                self.interrupt_receiving_pipe)
            # threads are properly joined when closing is initiated but are run as daemons to prevent Python
            # from trying to join them automatically too early on cleanup
            self._tracker_thread = threading.Thread(
                target=tracker_thread,
                args=(self._processes, self.tracker_interruption_pipe, interrupt_dispatcher),
                daemon=True)
            self._tracker_thread.start()
        except:
            if self._tracker_thread is not None and self._tracker_thread.is_alive():
                self.interrupt_tracker_pipe.send(None)
            elif self.task_dispatcher is not None and self.task_dispatcher.is_alive():
                interrupt_dispatcher()
            else:
                for proc in self._processes:
                    if proc.is_alive():
                        proc.terminate()
                for proc in self._processes:
                    if proc.pid is not None:
                        proc.join()
            raise


class TaskDispatcherWorker:

    def __init__(self, pending_cv, pending, general_tasks_pipe, exclusive_tasks_pipes, dispatcher_interruption_pipe, interrupt_receiving_pipe):
        self.pending_cv = pending_cv
        self.pending = pending
        self.poll = select.poll()
        self.interruption_handle = dispatcher_interruption_pipe.fileno()
        self.poll.register(self.interruption_handle, select.POLLIN)
        self.exclusive_tasks_pipes = exclusive_tasks_pipes
        self.exclusive_tasks_handles = {worker_id: pipe.fileno() for worker_id, pipe in exclusive_tasks_pipes.items()}
        self.pending_writes = {pipe.fileno(): deque() for pipe in exclusive_tasks_pipes.values()}
        self.pipes = {pipe.fileno(): pipe for pipe in exclusive_tasks_pipes.values()}
        self.general_tasks_pipe = general_tasks_pipe
        if general_tasks_pipe is not None:
            self.general_tasks_handle = general_tasks_pipe.fileno()
            self.pending_writes[self.general_tasks_handle] = deque()
            self.pipes[self.general_tasks_handle] = self.general_tasks_pipe
        self.interrupt_receiving_pipe = interrupt_receiving_pipe
        self.is_interrupted = False

    def dispatch_loop(self):
        try:
            while not self.is_interrupted:
                with self.pending_cv:
                    while len(self.pending) == 0:
                        self.pending_cv.wait()
                    to_send = list(self.pending)
                    self.pending.clear()
                if any(task is None for task in to_send):
                    self.is_interrupted = True
                    break
                self.send(to_send)
        finally:
            if self.general_tasks_pipe is not None:
                self.general_tasks_pipe.close()
            for pipe in self.exclusive_tasks_pipes.values():
                pipe.close()
            self.interrupt_receiving_pipe.send(None)

    def send(self, to_send):
        """TODO describe blocking and how it is possibly avoided"""
        to_send_count = len(to_send)
        for exclusive_worker_id, task in to_send:
            if exclusive_worker_id is None:
                task_handle = self.general_tasks_handle
            else:
                task_handle = self.exclusive_tasks_handles[exclusive_worker_id]
            queue = self.pending_writes[task_handle]
            if len(queue) == 0:
                self.poll.register(task_handle, select.POLLOUT)
            self.pending_writes[task_handle].append(task)
        while to_send_count > 0:
            ready = self.poll.poll()
            if any(handle == self.interruption_handle for handle, _ in ready):
                self.is_interrupted = True
                break
            for handle, _ in ready:
                queue = self.pending_writes[handle]
                if len(queue) == 0:
                    self.poll.unregister(handle)
                    continue
                task = queue.popleft()
                self.pipes[handle].send(task)
                to_send_count -= 1


class TaskDispatcher:

    def __init__(self):
        self._pending = []
        self._pending_cv = threading.Condition()
        self._dispatcher_thread = None

    def start_thread(self, general_tasks_pipe, exclusive_tasks_pipes, dispatcher_interruption, interrupt_receiving):
        worker = TaskDispatcherWorker(self._pending_cv, self._pending, general_tasks_pipe, exclusive_tasks_pipes,
            dispatcher_interruption, interrupt_receiving)
        self._dispatcher_thread = threading.Thread(target=worker.dispatch_loop, daemon=True)
        self._dispatcher_thread.start()

    def push_tasks(self, scheduled_tasks : List[Optional[Tuple[int, ScheduledTask]]]):
        with self._pending_cv:
            self._pending.extend(scheduled_tasks)
            self._pending_cv.notify()

    def is_alive(self):
        return self._dispatcher_thread is not None and self._dispatcher_thread.is_alive()


def tracker_thread(processes, tracker_interruption_pipe, interrupt_dispatcher):
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
    try:
        ps = {p.sentinel: p for p in processes}
        listen_for = list(ps.keys()) + [tracker_interruption_pipe]
        # Once one process exits stop the whole group (gracefully if possible)
        while True:
            sentinels = multiprocessing.connection.wait(listen_for)
            proc_sentinels = [s for s in sentinels if s != tracker_interruption_pipe]
            if tracker_interruption_pipe in sentinels or any(
                    ps[sentinel].exitcode is not None for sentinel in proc_sentinels):
                break
    except:
        for proc in processes:
            if proc.exitcode is None:
                proc.terminate()
        raise
    finally:
        interrupt_dispatcher()
        for proc in processes:
            proc.join(1)
        for proc in processes:
            if proc.exitcode is None:
                proc.terminate()
                proc.join()


class WorkerPool:
    """"Combines worker processes pool with callback contexts, can be used to schedule batches
    to be run on the workers and to receive resulting batches from the workers."""

    def __init__(self, contexts, pool):
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

    @classmethod
    def from_groups(
            cls, groups, keep_alive_queue_size, start_method="fork", num_workers=1,
            initial_chunk_size=1024 * 1024, py_callback_pickler=None):
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
        `initial_chunk_size` : int
            Initial size of each shared memory chunk.
        """
        import_numpy()
        if initial_chunk_size <= 0:
            raise RuntimeError("Chunk capacity must be a positive integer")
        if len(groups) == 0:
            raise RuntimeError("Cannot create Python workers pool because there are no callbacks provided")
        if num_workers < 1:
            raise RuntimeError("Number of Python workers for parallel ExternalSource must be positive")
        if any(group.source_desc.kind == SourceKind.CALLABLE and not group.source_desc.has_inputs for group in groups):
            raise RuntimeError("Callable `source` of parallel external source must accept argument")
        if any(group.source_desc.kind != SourceKind.CALLABLE and not group.batch for group in groups):
            raise RuntimeError("Parallel external source with iterator or generator must run in batch mode")
        # iterators and generators are stateful and run always in the same dedicated worker
        is_exclusive_cbs = [WorkerPool.needs_exclusive_worker(group) for group in groups]
        num_exclusive_cbs = sum(is_exclusive_cbs)
        num_general_cbs = len(groups) - num_exclusive_cbs
        if num_general_cbs == 0:
            if num_workers > num_exclusive_cbs:
                warnings.warn(
                "There will be run only {} python worker{}, even though {} were specified to run. "
                "This may happen when all your ExternalSource callbacks are stateful (for instance they are iterators) and "
                "there is less of them than ```py_num_workers```".format(
                    num_exclusive_cbs, "s" if num_exclusive_cbs > 1 else "", num_workers), Warning)
                num_workers = num_exclusive_cbs
        next_excl_worker = num_workers - 1
        def get_next_excl_worker():
            nonlocal next_excl_worker
            next_excl_worker = (next_excl_worker + 1) % num_workers
            return next_excl_worker
        sources_desc = [group.source_desc for group in groups]
        exclusive_workers = [get_next_excl_worker() if is_exclusive_cb else None for is_exclusive_cb in is_exclusive_cbs]
        shm_buffers = [
            SharedMemBuffer(
                callback_i,
                keep_alive_queue_size + group.prefetch_queue_depth,
                initial_chunk_size,
                1 if group.batch else num_workers
            ) for callback_i, group in enumerate(groups)]
        contexts = [
            CallbackContext(source_desc, exclusive_worker, shm_buffer) for source_desc, exclusive_worker, shm_buffer
            in zip(sources_desc, exclusive_workers, shm_buffers)]
        pool = ProcPool(contexts, num_workers, start_method, py_callback_pickler)
        return cls(contexts, pool)

    @staticmethod
    def needs_exclusive_worker(group):
        return group.source_desc.kind != SourceKind.CALLABLE

    def schedule_batch(self, context_i, dst_chunk_i, work_batch):
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

    def _distribute(self, context_i, scheduled_i, dst_chunk_i, work_batch):
        minibatches = work_batch.split_work(self.pool.num_workers)
        context = self.contexts[context_i]
        scheduled_tasks = [
            (self.contexts[context_i].exclusive_worker,
            ScheduledTask(context_i, scheduled_i, context.epoch_start,
                          context.shm_buffer.get_chunk_by_dest(dst_chunk_i, minibatch_i), task))
            for minibatch_i, task in enumerate(minibatches)
        ]
        self.pool.task_dispatcher.push_tasks(scheduled_tasks)

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
        completed_tasks = self.pool.wait_for_res()
        if completed_tasks is None:
            raise RuntimeError("Worker data receiving interrupted")
        for completed_task in completed_tasks:
            context = self.contexts[completed_task.context_i]
            context.process_task(completed_task)

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
