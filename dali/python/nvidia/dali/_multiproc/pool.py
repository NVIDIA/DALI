# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali._utils.external_source_impl import SourceDescription, SourceKind
from nvidia.dali._multiproc.worker import worker
from nvidia.dali._multiproc.messages import ScheduledTask, TaskArgs, WorkerArgs
from nvidia.dali._multiproc.shared_batch import (
    deserialize_batch,
    import_numpy,
    read_shm_message,
    BufShmChunk,
    SharedBatchWriter,
    write_shm_message,
    _align_up as align_up,
)
from nvidia.dali._multiproc.shared_queue import ShmQueue


"""
A pipeline with parallel external sources creates `WorkerPool` to parallelize sources computation.
Each external source in the pipeline has its own `ShmChunkManager` with a view on shm chunks
dedicated to data computed by the given source. Those chunks are also used to pass minibatch (task)
description to the workers. All the chunks for all the external sources in the pipeline
are created and stored in the common list shared by `ShmChunkManager` instances (index in the list
makes a unique id for a chunk used in communication between workers and the pool), but a single
chunk is always allocated and used by a single `ShmChunkManager` instance.
`CallbackContext` combines the source (callback, iterator or generator function)
with its `ShmChunkManager` instance and contains optional dedicated_worker_id if the source is
stateful and needs to be run in a single dedicated worker. Context keeps track of scheduled tasks
and partial results received from the workers for the given source.
`Pool` manages actual worker processes and communication between them and the main process,
it is responsible for starting the workers and additional setup steps (such as passing shm chunks
through sockets if `spawn` start method is used).
`Pool` instance uses `ShmQueue` to communicate and synchronize with workers: actual tasks and data
are serialized and put in shm chunks from ShmChunkManagers, whereas messages in ShmQueue contain
only simple fixed-size meta data such as the id of shm chunk from the buffer to be read,
current capacity of the shm chunk (which may increase if worker couldn't fit the data)
and offset of the data.
If the pipeline contains any stateless source, that can be run in parallel, all the workers will
share a common ShmQueue instance with tasks related to any such source.
If some source gets a dedicated worker assigned, the worker will receive a dedicated queue not
shared with other workers and will receive all the dedicated tasks there.
Thus, a worker can have up to two queues with tasks.
Additionally there is a single result queue shared by all the workers, that is used to notify main
process about completed tasks being ready for consumption by the main process.
"""


class ShmChunkManager:
    """Two dimensional buffer of shared memory chunks (queue_depth X num_minibatches),
    chunks can be accessed either by providing two coordinates or via shm chunk's unique id.
    Each ExternalSource callback gets its own buffer, first dimension is cycled
    over when scheduling and receiving consecutive batches, second dimension is
    used to separate minibatches."""

    def __init__(
        self, shm_pool: List[BufShmChunk], queue_depth, initial_chunk_capacity, num_minibatches
    ):
        if queue_depth < 1:
            raise RuntimeError("Prefetch queue must have at least one element")
        if initial_chunk_capacity < 1:
            raise RuntimeError("Buffer chunk capacity must be a positive integer")
        self.shm_pool = shm_pool
        self.queue_depth = queue_depth
        self.initial_chunk_capacity = align_up(
            initial_chunk_capacity, SharedBatchWriter.BUFFER_ALIGNMENT
        )
        self.num_minibatches = num_minibatches
        self.chunks_ids_by_pos = []
        for _ in range(self.queue_depth):
            self.chunks_ids_by_pos.append(
                [
                    self.allocate_chunk(self.initial_chunk_capacity)
                    for _ in range(self.num_minibatches)
                ]
            )
        self.chunks_ids = [chunk_id for dest_buf in self.chunks_ids_by_pos for chunk_id in dest_buf]

    def allocate_chunk(self, capacity):
        chunk_id = len(self.shm_pool)
        chunk = BufShmChunk.allocate(chunk_id, capacity)
        self.shm_pool.append(chunk)
        return chunk_id

    def close_handles(self):
        for shm_chunk_id in self.chunks_ids:
            self.shm_pool[shm_chunk_id].close_handle()

    def get_chunk_by_id(self, shm_chunk_id):
        return self.shm_pool[shm_chunk_id]

    def get_chunk_by_dest(self, dest_i, minibatch_i):
        chunk_id = self.chunks_ids_by_pos[dest_i][minibatch_i]
        return self.get_chunk_by_id(chunk_id)

    def get_chunks(self):
        return [self.get_chunk_by_id(chunk_id) for chunk_id in self.chunks_ids]

    @property
    def num_chunks(self):
        return len(self.chunks_ids)


class CallbackContext:
    """Keeps track of tasks and partially received results for a given source.
    Contains source description, dedicated ShmChunkManager instance and
    information about dedicated worker id if applicable."""

    def __init__(
        self,
        source_desc: SourceDescription,
        shm_manager: ShmChunkManager,
        dedicated_worker_id: Optional[int],
    ):
        self.source_desc = source_desc
        self.shm_manager = shm_manager
        self.dedicated_worker_id = dedicated_worker_id
        # counts all the batches ever scheduled for the given source, serves as
        # id for the next scheduled tasks
        self.scheduled_i = 0
        # counts all the batches ever returned from the context, used to calculate position
        # in the ShmChunkManager circular buffer for a next batch
        self.produced_i = 0
        self.epoch_start = self.scheduled_i
        self.partially_received = {}
        self.scheduled_minibatches = {}
        self.iter_failed = {}
        self.task_queue = deque()
        self.epoch_synced = True

    def reset(self):
        """Invalidates all batches pending in `task_queue`, marks the need to sync the epoch
        before `ShmChunkManager` chunks occupied by `task_queue` batches can be be reused for
        prefetching in the next epoch"""
        self.epoch_start = self.scheduled_i
        self.epoch_synced = False

    def push_scheduled(self, num_minibatches):
        scheduled_i = self.scheduled_i
        self.scheduled_i += 1
        self.partially_received[scheduled_i] = {}
        self.scheduled_minibatches[scheduled_i] = num_minibatches
        dest_chunk_i = (self.produced_i + self.scheduled_ahead) % self.queue_depth
        self.task_queue.append(scheduled_i)
        return scheduled_i, dest_chunk_i

    def clear_scheduled(self, scheduled_i):
        del self.partially_received[scheduled_i]
        del self.scheduled_minibatches[scheduled_i]
        if scheduled_i in self.iter_failed:
            del self.iter_failed[scheduled_i]

    def pop_scheduled(self):
        return self.task_queue.popleft()

    def handle_error(self, batch_i):
        """Check if given batch reported an error and raise it"""
        if batch_i in self.iter_failed:
            exception, traceback_str = self.iter_failed[batch_i]
            try:
                self.clear_scheduled(batch_i)
                if isinstance(exception, StopIteration):
                    raise exception
                else:
                    # Raise new exception propagating the traceback from worker thread as error
                    # message, originating from original exception
                    raise Exception(
                        "\n\nException traceback received from worker thread:\n\n" + traceback_str
                    ) from exception
            finally:
                # Fix circular reference problem on StopIteration - the exception contains
                # reference to the traceback that refers a frame that contains local variables
                # and among them the exception.
                # This traceback is then chained into exceptions reraised along the way
                # (eventually at the pipeline level) which in effect introduces a reference to
                # the pipeline that would be only removed after garbage collection round,
                # delaying finalization of the pool
                del exception

    def is_error(self, scheduled_i):
        return scheduled_i in self.iter_failed

    def process_task(self, shm_chunk: BufShmChunk, completed_task):
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
        return len(self.partially_received[scheduled_i]) < self.scheduled_minibatches[scheduled_i]

    def coalesce_received(self, scheduled_i):
        num_minibatches = self.scheduled_minibatches[scheduled_i]
        minibatches = self.partially_received[scheduled_i]
        if num_minibatches == 1:
            return minibatches[0]
        return [
            sample for minibatch_i in range(num_minibatches) for sample in minibatches[minibatch_i]
        ]

    def take_processed(self, scheduled_i):
        """Return the full batch, mark it as cleared and consumed"""
        batch = self.coalesce_received(scheduled_i)
        self.clear_scheduled(scheduled_i)
        self.produced_i += 1
        return batch

    def receive_chunk(self, shm_chunk, completed_task):
        """Obtain the chunk and decode it, add to partially gathered result"""
        scheduled_i = completed_task.scheduled_i
        minibatch_i = completed_task.minibatch_i
        worker_batch = deserialize_batch(shm_chunk, completed_task.batch_meta)
        self.partially_received[scheduled_i][minibatch_i] = worker_batch

    @property
    def queue_depth(self):
        return self.shm_manager.queue_depth

    @property
    def scheduled_ahead(self):
        # at the beginning of a new epoch all previously scheduled tasks are discarded
        if not self.epoch_synced:
            return 0
        return len(self.task_queue)


class WorkerContext:
    def __init__(
        self,
        source_descs: SourceDescription,
        dedicated_task_queue: Optional[ShmQueue],
        shm_chunks: List[BufShmChunk],
    ):
        self.source_descs = source_descs
        self.dedicated_task_queue = dedicated_task_queue
        self.shm_chunks = shm_chunks


def create_worker_contexts(
    mp, callback_contexts: List[CallbackContext], num_workers, callback_pickler
) -> List[WorkerContext]:
    """
    Prepares list of `WorkerContext` instances.
    Each instance describes parameters specific to a given worker process (as opposed to
    parameters common for all processes in the pool).
    WorkerContext contains sources that the worker will receive and shared memory chunks
    corresponding to the sources. It also contains dedicated `ShmQueue` instance if any of
    the sources was assigned a dedicated worker.
    """
    if callback_pickler is None:
        source_descs = [cb_context.source_desc for cb_context in callback_contexts]
    else:
        source_descs = [copy.copy(cb_context.source_desc) for cb_context in callback_contexts]
        for source_desc in source_descs:
            source_desc.source = callback_pickler.dumps(source_desc.source)
    general_cb_contexts = [
        i
        for i, cb_context in enumerate(callback_contexts)
        if cb_context.dedicated_worker_id is None
    ]
    worker_contexts = []
    for worker_id in range(num_workers):
        dedicated_cb_contexts = [
            i
            for i, cb_context in enumerate(callback_contexts)
            if cb_context.dedicated_worker_id == worker_id
        ]
        worker_cb_contexts = general_cb_contexts + dedicated_cb_contexts
        worker_sources = {i: source_descs[i] for i in worker_cb_contexts}
        worker_shm_chunks = [
            shm_chunk
            for i in worker_cb_contexts
            for shm_chunk in callback_contexts[i].shm_manager.get_chunks()
        ]
        if not dedicated_cb_contexts:
            dedicated_task_queue = None
        else:
            # Each scheduled task has a shm chunk assigned for results, the number of
            # scheduled tasks won't exceed the number of chunks available for results
            dedicated_task_queue = ShmQueue(
                mp,
                capacity=sum(
                    callback_contexts[i].shm_manager.num_chunks for i in dedicated_cb_contexts
                ),
            )
        worker_context = WorkerContext(worker_sources, dedicated_task_queue, worker_shm_chunks)
        worker_contexts.append(worker_context)
    return worker_contexts


class ProcPool:
    """Runs pool of worker processes, stores pipes and sockets used to communicate with
    the workers, starts thread keeping track of running processes and initializes communication.
    """

    def __init__(
        self,
        mp,
        workers_contexts: List[WorkerContext],
        result_queue: ShmQueue,
        general_task_queue: Optional[ShmQueue],
        callback_pickler,
    ):
        start_method = mp.get_start_method()
        if not workers_contexts:
            raise RuntimeError("Cannot start a pool with no workers")
        if start_method == "fork" and _b.IsDriverInitialized():
            raise RuntimeError(
                "Error when starting Python worker threads for DALI parallel External Source. "
                "Cannot fork a process when the CUDA has been initialized in the process. "
                "CUDA is initialized during ``Pipeline.build()``, or can be initialized by another"
                " library that interacts with CUDA, for example a DL framework creating "
                "CUDA tensors. If you are trying to build multiple pipelines that use Python "
                "workers, you will need to call ``start_py_workers`` method on all of them before "
                "calling ``build`` method of any pipeline to start Python workers before CUDA is "
                "initialized by ``build`` or other CUDA operation. Alternatively you can change "
                "Python workers starting method from ``fork`` to ``spawn`` "
                "(see DALI Pipeline's ``py_start_method`` option for details). "
            )
        self._workers_contexts = workers_contexts
        self._result_queue = result_queue
        self._general_task_queue = general_task_queue
        self._observer = None
        self._processes = []
        write_sockets = []
        try:
            for worker_i, worker_context in enumerate(workers_contexts):
                if start_method == "fork":
                    read_socket = None
                else:
                    read_socket, write_socket = socket.socketpair()
                    write_sockets.append(write_socket)
                process_context = WorkerArgs(
                    worker_id=worker_i,
                    start_method=start_method,
                    source_descs=worker_context.source_descs,
                    shm_chunks=worker_context.shm_chunks,
                    general_task_queue=general_task_queue,
                    dedicated_task_queue=worker_context.dedicated_task_queue,
                    result_queue=result_queue,
                    setup_socket=read_socket,
                    callback_pickler=callback_pickler,
                )
                process = mp.Process(target=worker, args=(process_context,))
                self._processes.append(process)
            self._start_processes(mp, start_method, write_sockets)
        finally:
            for sock in write_sockets:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()

    @classmethod
    def from_contexts(
        cls,
        contexts: List[CallbackContext],
        num_workers,
        start_method="fork",
        py_callback_pickler=None,
    ):
        mp = multiprocessing.get_context(start_method)
        # checks if there are any sources without dedicated worker id, if so,
        # the `general_task_queue` instance is needed to distribute tasks among all the workers
        general_sources_buffs = [
            context.shm_manager for context in contexts if context.dedicated_worker_id is None
        ]
        if not general_sources_buffs:
            general_task_queue = None
        else:
            # Each scheduled task has a shm chunk assigned for results, the number of
            # scheduled tasks won't exceed the number of chunks available for results
            general_task_queue = ShmQueue(
                mp, capacity=sum(shm_manager.num_chunks for shm_manager in general_sources_buffs)
            )
        # Each computed minibatch makes for one message in the results queue, the number of
        # messages won't exceed the number of shm chunks available to store the minibatches
        # in all the `ShmChunkManager` instances.
        scheduled_tasks_upper_bound = sum(context.shm_manager.num_chunks for context in contexts)
        # assure enough space for messages sent to confirm initialization of the workers
        result_queue_capacity = max(scheduled_tasks_upper_bound, num_workers)
        result_queue = ShmQueue(mp, capacity=result_queue_capacity)
        callback_pickler = (
            None if start_method == "fork" else pickling._CustomPickler.create(py_callback_pickler)
        )
        worker_contexts = create_worker_contexts(mp, contexts, num_workers, callback_pickler)
        instance = None
        try:
            instance = cls(mp, worker_contexts, result_queue, general_task_queue, callback_pickler)
            if general_task_queue is not None:
                general_task_queue.close_handle()
            result_queue.close_handle()
            for worker_context in worker_contexts:
                if worker_context.dedicated_task_queue is not None:
                    worker_context.dedicated_task_queue.close_handle()
            return instance
        except:  # noqa: E722
            if instance is not None:
                instance.close()
            raise

    @property
    def num_workers(self):
        return len(self._workers_contexts)

    def pids(self):
        """Get pids of the processes started by this pool."""
        return [proc.pid for proc in self._processes]

    def close(self):
        if self._observer is None:
            return
        self._observer.close()
        self._observer = None

    def wait_for_res(self):
        if self._observer is None:
            raise RuntimeError("Cannot receive data from the pool that has been closed")
        return self._result_queue.get(None)

    def send(self, tasks: List[Tuple[BufShmChunk, Any]], dedicated_worker_id):
        if self._observer is None:
            raise RuntimeError("Cannot send tasks to the pool that has been closed")
        shm_msg_descs = [
            write_shm_message(-1, shm_chunk, msg, 0, resize=False) for shm_chunk, msg in tasks
        ]
        if dedicated_worker_id is None:
            if self._general_task_queue.put(shm_msg_descs) is None:
                raise RuntimeError("Sending tasks to workers failed")
        else:
            worker_ctx = self._workers_contexts[dedicated_worker_id]
            if worker_ctx.dedicated_task_queue.put(shm_msg_descs) is None:
                raise RuntimeError("Sending tasks to worker {} failed".format(dedicated_worker_id))

    def _sync_initialized_workers(self):
        workers_received = []
        while len(workers_received) < self.num_workers:
            shm_msgs = self.wait_for_res()
            if shm_msgs is None:
                raise RuntimeError("Workers initialization failed")
            synced_ids = [shm_msg.worker_id for shm_msg in shm_msgs]
            assert all(
                0 <= worker_id < self.num_workers and worker_id not in workers_received
                for worker_id in synced_ids
            )
            workers_received.extend(synced_ids)

    def _send_queue_handles(self, write_sockets):
        pid = os.getppid()
        all_worker_queues = [self._result_queue]
        if self._general_task_queue is not None:
            all_worker_queues.append(self._general_task_queue)
        for queue in all_worker_queues:
            for sock in write_sockets:
                multiprocessing.reduction.send_handle(sock, queue.shm.handle, pid)
        for sock, worker_context in zip(write_sockets, self._workers_contexts):
            if worker_context.dedicated_task_queue is not None:
                multiprocessing.reduction.send_handle(
                    sock, worker_context.dedicated_task_queue.shm.handle, pid
                )

    def _send_shm_handles(self, socks):
        pid = os.getppid()
        for sock, worker_context in zip(socks, self._workers_contexts):
            for shm_chunk in worker_context.shm_chunks:
                multiprocessing.reduction.send_handle(sock, shm_chunk.handle, pid)

    def _start_processes(self, mp, start_method, write_sockets):
        try:
            for process in self._processes:
                process.start()
            task_queues = [
                worker_context.dedicated_task_queue
                for worker_context in self._workers_contexts
                if worker_context.dedicated_task_queue is not None
            ]
            if self._general_task_queue is not None:
                task_queues.append(self._general_task_queue)
            self._observer = Observer(mp, self._processes, task_queues, self._result_queue)
            if start_method != "fork":
                # NOTE when making any changes here, make sure to reflect them in the worker
                # process, so that it sets received handles to objects in the same order
                self._send_queue_handles(write_sockets)
                self._send_shm_handles(write_sockets)
            self._sync_initialized_workers()
        except:  # noqa: E722
            if self._observer is not None:
                self._observer.close()
                self._observer = None
            else:
                for proc in self._processes:
                    if proc.is_alive():
                        proc.terminate()
                for proc in self._processes:
                    if proc.pid is not None:
                        proc.join()
            raise


class Observer:
    """
    Closes the whole pool of worker processes if any of the processes exits. The processes can also
    be closed from the main process by calling observer `close` method.
    ----------
    mp : Python's multiprocessing context (depending on start method used: `spawn` or `fork`)
    processes : List of multiprocessing Process instances
    task_queues : List[ShmQueue]
        Queues that worker processes take tasks from. If `close` method is called and none of
        the processes exited abruptly so far, the queues will be used to notify the workers about
        closing to let the workers gracefully exit.
    result_queue : ShmQueue
        Queue where worker processes report completed tasks. It gets closed along with the worker
        processes, to prevent the main process blocking on waiting for results from the workers.
    """

    def __init__(self, mp, processes, task_queues, result_queue):
        self._interruption_pipe, self.interrupt_pipe = mp.Pipe(duplex=False)
        self._processes = processes
        self._task_queues = task_queues
        self._result_queue = result_queue
        self.thread = threading.Thread(target=self._observer_thread, daemon=True)
        self.thread.start()

    def _observer_thread(self):
        """Observer thread for ProcPool used for stopping and joining processes."""
        exit_gently = True
        try:
            ps = {p.sentinel: p for p in self._processes}
            listen_for = list(ps.keys()) + [self._interruption_pipe]
            # Once one process exits stop the whole group (gracefully if possible)
            while True:
                sentinels = multiprocessing.connection.wait(listen_for)
                proc_sentinels = [s for s in sentinels if s != self._interruption_pipe]
                if self._interruption_pipe in sentinels:
                    break
                if any(ps[sentinel].exitcode is not None for sentinel in proc_sentinels):
                    exit_gently = False
                    break
        except:  # noqa: E722
            exit_gently = False
            raise
        finally:
            self._result_queue.close()  # main thread may be blocked on reading the queue
            if exit_gently:
                # try to close task queues and notify waiting processes, so that they can
                # cleanup and exit. Unfortunately if all workers exited abruptly when waiting,
                # an attempt to notify workers with multiprocessing.Condition might lead to
                # deadlock on underlying semaphore. For this reason it is done only if none of
                # the workers reported to have exited.
                for queue in self._task_queues:
                    queue.close()
                for proc in self._processes:
                    proc.join(1)
            for proc in self._processes:
                if proc.exitcode is None:
                    proc.terminate()
                    proc.join()

    def close(self):
        if self.thread is not None:
            # send anything via interruption_pipe to notify observer thread about closing
            self.interrupt_pipe.send(None)
            self.thread.join()
            self.thread = None


def create_shm_chunk_manager_for_group(
    group, shm_pool, keep_alive_queue_size, min_initial_chunk_size, num_workers, batch_size=None
):
    num_mini_batches = 1 if group.batch else num_workers
    if group.bytes_per_sample_hint is None or batch_size is None:
        initial_chunk_size = min_initial_chunk_size
    else:
        num_samples_per_mini_batch = (batch_size + num_mini_batches - 1) // num_mini_batches
        initial_chunk_size = num_samples_per_mini_batch * group.bytes_per_sample_hint
        initial_chunk_size = max(min_initial_chunk_size, initial_chunk_size)
    return ShmChunkManager(
        shm_pool,
        keep_alive_queue_size + group.prefetch_queue_depth,
        initial_chunk_size,
        num_mini_batches,
    )


class WorkerPool:
    """ "Combines worker processes pool with callback contexts, can be used to schedule batches
    to be run on the workers and to receive resulting batches from the workers."""

    def __init__(self, contexts: List[CallbackContext], pool: ProcPool):
        """
        Parameters
        ----------
        contexts : List[CallbackContext]
            List of callbacks' contexts to be handled by the Worker.
        pool : ProcPool
            ProcPool instance enabling basic communication with worker processes, it should be
            initialized with `contexts`.
        """
        self.contexts = contexts
        self.pool = pool
        # shm chunks ids must be unique across the pool and each chunk must belong to
        # exactly one context.
        # Thanks to that callback context can be identified by the id of shm chunk.
        self.shm_chunks_contexts = {
            chunk_id: context
            for context in self.contexts
            for chunk_id in context.shm_manager.chunks_ids
        }

    @classmethod
    def from_groups(
        cls,
        groups,
        keep_alive_queue_size,
        batch_size=None,
        start_method="fork",
        num_workers=1,
        min_initial_chunk_size=1024 * 1024,
        py_callback_pickler=None,
    ):
        """Creates new WorkerPool instance for given list of ExternalSource groups.

        Parameters
        ----------
        groups : _ExternalSourceGroup list
            List of external source groups.
        keep_alive_queue_size : int
            Number of the most recently produced batches whose underlying shared memory should
            remain untouched (because they might still be referenced further in the pipeline).
            Note that the actual number of simultaneously kept batches will be greater by the length
            of parallel external source prefetching queue which is at least one.
        batch_size : int, optional
            Maximal batch size. For now, used only to estimate initial capacity of virtual
            memory slots.
        start_method : str
            Method of starting worker processes, either fork or spawn.
        num_workers : int
            Number of workers to be created in ProcPool.
        min_initial_chunk_size : int
            Minimal initial size of each shared memory chunk.
            NOTE it must be enough to accommodate serialized `ScheduledTask` instance.
        """
        import_numpy()
        if len(groups) == 0:
            raise RuntimeError(
                "Cannot create Python workers pool because" " there are no callbacks provided"
            )
        if num_workers < 1:
            raise RuntimeError(
                "Number of Python workers for parallel" " ExternalSource must be positive"
            )
        if any(
            group.source_desc.kind != SourceKind.CALLABLE and not group.batch for group in groups
        ):
            raise RuntimeError(
                "Parallel external source with iterator" " or generator must run in batch mode"
            )
        # iterators and generators are stateful and run always in the same dedicated worker
        num_cbs_dedicated = sum(cls.is_iterable_group(group) for group in groups)
        num_cbs_general = len(groups) - num_cbs_dedicated
        if num_cbs_general == 0:
            if num_workers > num_cbs_dedicated:
                warn_args = (num_cbs_dedicated, "s" if num_cbs_dedicated > 1 else "", num_workers)
                warnings.warn(
                    "There will be run only {} python worker{}, even though {} were"
                    " specified to run. This may happen when all your ExternalSource"
                    " callbacks are stateful (for instance they are iterators) and there"
                    " is less of them than ```py_num_workers```".format(*warn_args),
                    Warning,
                )
                num_workers = num_cbs_dedicated
        source_descs = [group.source_desc for group in groups]
        dedicated_workers = cls.assign_dedicated_workers(groups, num_workers)
        # common list for all the chunks allocated by ShmChunkManagers
        # of all sources in the pipeline
        shm_pool = []
        shm_managers = [
            create_shm_chunk_manager_for_group(
                group,
                shm_pool,
                keep_alive_queue_size,
                min_initial_chunk_size,
                num_workers,
                batch_size,
            )
            for group in groups
        ]
        contexts = [
            CallbackContext(source_desc, shm_manager, dedicated_worker_id)
            for source_desc, shm_manager, dedicated_worker_id in zip(
                source_descs, shm_managers, dedicated_workers
            )
        ]
        pool = None
        try:
            pool = ProcPool.from_contexts(contexts, num_workers, start_method, py_callback_pickler)
            # close underlying file descriptors that are not needed anymore once
            # passed to the workers processes
            for context in contexts:
                context.shm_manager.close_handles()
            return cls(contexts, pool)
        except:  # noqa: E722
            if pool is not None:
                pool.close()
            raise

    @classmethod
    def assign_dedicated_workers(cls, groups, num_workers):
        def get_next_dedicated_worker():
            next_dedicated_worker = num_workers - 1
            while True:
                next_dedicated_worker = (next_dedicated_worker + 1) % num_workers
                yield next_dedicated_worker

        next_dedicated_worker = get_next_dedicated_worker()
        return [
            next(next_dedicated_worker) if cls.is_iterable_group(group) else None
            for group in groups
        ]

    @classmethod
    def is_iterable_group(cls, group):
        return group.source_desc.kind != SourceKind.CALLABLE

    def schedule_batch(self, context_i, work_batch: TaskArgs):
        """Distribute `work_batch` among workers.

        Parameters
        ----------
        context_i : int
            Specifies which callback will be used to run the task, it must be the index
            corresponding to the order of callbacks passed when constructing WorkerPool.
        work_batch : TaskArgs
            Wrapper around parameters produced by the ExternalSource describing the next batch.
        """
        context = self.contexts[context_i]
        if not context.epoch_synced:
            self._sync_and_discard(context_i)
            context.epoch_synced = True
        if context.iter_failed:
            # there is no point in scheduling anything for the context that has reached the end of
            # data or failed with an error, once user receives batch that raised the exception they
            # should reset the context before scheduling new tasks
            return False
        minibatches = self._split_work(work_batch)
        num_minibatches = len(minibatches)
        assert num_minibatches <= context.shm_manager.num_minibatches
        scheduled_i, dst_chunk_i = context.push_scheduled(num_minibatches)
        self._distribute(context_i, scheduled_i, dst_chunk_i, minibatches)
        return True

    def _split_work(self, work_batch: TaskArgs):
        if not work_batch.is_sample_mode():
            return [work_batch]
        num_minibatches = self.pool.num_workers
        sample_range = work_batch.sample_range
        samples_num = len(sample_range)
        chunk_size = samples_num // num_minibatches
        remainder = samples_num % num_minibatches
        queued_no = 0
        minibatches = []
        for minibatch_i in range(num_minibatches):
            worker_chunk = chunk_size + (minibatch_i < remainder)
            if worker_chunk == 0:
                break
            sample_slice = sample_range[queued_no : queued_no + worker_chunk]
            minibatch = TaskArgs(minibatch_i, sample_range=sample_slice)
            minibatches.append(minibatch)
            queued_no += worker_chunk
        return minibatches

    def _distribute(self, context_i, scheduled_i, dst_chunk_i, minibatches):
        context = self.contexts[context_i]
        scheduled_tasks = [
            (
                context.shm_manager.get_chunk_by_dest(dst_chunk_i, minibatch_i),
                ScheduledTask(context_i, scheduled_i, context.epoch_start, task),
            )
            for minibatch_i, task in enumerate(minibatches)
        ]
        dedicated_worker_id = context.dedicated_worker_id
        self.pool.send(scheduled_tasks, dedicated_worker_id)

    def _sync_and_discard(self, context_i):
        context = self.contexts[context_i]
        assert not context.epoch_synced
        while context.task_queue:
            try:
                batch = self.receive_batch(context_i)
                assert batch is None
            except StopIteration:
                pass

    def receive_batch(self, context_i):
        """Returns the next produced batch (in the order of schedule_batch calls) for the
        ``context_i``th callback.

        Parameters
        ----------
        context_i : int
            Specifies which callback you want the results from, ordering corresponds to the order of
            callbacks passed when constructing the pool.
        """
        context = self.contexts[context_i]
        assert len(context.task_queue) > 0, "No task has been scheduled"
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
            shm_chunk = context.shm_manager.get_chunk_by_id(completed_task_meta.shm_chunk_id)
            completed_task = read_shm_message(shm_chunk, completed_task_meta)
            context.process_task(shm_chunk, completed_task)

    def pids(self):
        """Get pids of the processes started by this pool."""
        return self.pool.pids()

    def reset(self):
        for context in self.contexts:
            context.reset()

    def reset_context(self, context_i):
        self.contexts[context_i].reset()

    def close(self):
        self.pool.close()
