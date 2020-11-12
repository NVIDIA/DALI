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

import os
import socket
import pickle
import threading
import multiprocessing
from collections import OrderedDict
from nvidia.dali import backend as _b
from nvidia.dali.worker import worker, ScheduledTasks
from nvidia.dali.shared_batch import deserialize_batch, import_numpy
from nvidia.dali import shared_mem


class MemChunk:

    def __init__(self, shm_chunk, capacity):
        self.shm_chunk = shm_chunk
        self.capacity = capacity


class SharedBatchesConsumer:
    """
Counterpart of worker.py:SharedBatchesDispatcher. Can receive and deserialize batch from the worker,
keeps track of already received memory chunks and opens new chunks or resizes exiting ones if necessary.
"""

    def __init__(self):
        import_numpy()
        self.batch_pool = {}

    def get_mem_chunk(self, sock, batch):
        chunk = self.batch_pool.get(batch.mem_chunk_id)
        if chunk is not None:
            if chunk.capacity != batch.capacity:
                chunk.shm_chunk.resize(batch.capacity, trunc=False)
                chunk.capacity = batch.capacity
            return chunk
        fd, shm_chunk = -1, None
        try:
            [fd] = multiprocessing.reduction.recvfds(sock, 1)
            assert os.fstat(fd).st_size >= batch.capacity
            shm_chunk = shared_mem.SharedMem.open(fd, batch.capacity)
        except:
            if shm_chunk is not None:
                shm_chunk.close()
            # close fd manually if shm_chunk creation failed, otherwise shm_chunk
            # is responsible for doing so
            elif fd >= 0:
                os.close(fd)
            raise
        chunk = MemChunk(shm_chunk, batch.capacity)
        self.batch_pool[batch.mem_chunk_id] = chunk
        return chunk

    def load_batch(self, sock, batch):
        chunk = self.get_mem_chunk(sock, batch)
        meta_data = chunk.shm_chunk.buf[batch.meta_offset:batch.meta_offset + batch.meta_size]
        samples = pickle.loads(meta_data)
        return deserialize_batch(chunk.shm_chunk.buf, samples)


class CallbackContext:
    """Counterpart of worker.py:CallbackContext, for every callback that can be run
in the workers stores shared memory chunks that are used to passed callback results, keeps track
of what tasks has been scheduled to run in the workers and what partial results has been
received so far.
"""

    def __init__(self):
        self.batch_consumer = SharedBatchesConsumer()
        self.reset()

    def reset(self):
        self.partially_received = {}
        self.scheduled = OrderedDict()
        self.iter_failed = {}


class ProcPool:
    """Runs pool of worker processes, stores pipes and sockets used to communicate with the workers,
starts thread keeping track of running processes and initializes communication.
"""

    def __init__(self, callbacks, prefetch_queue_depths, workers_num=None, init_method="fork", initial_chunk_size=1024 * 1024):
        if len(callbacks) != len(prefetch_queue_depths):
            raise RuntimeError("Number of prefetch queues must match number of callbacks")
        if any(prefetch_queue_depth <= 0 for prefetch_queue_depth in prefetch_queue_depths):
            raise RuntimeError("Prefetch queue must have at least one element")
        if initial_chunk_size <= 0:
            raise RuntimeError("Chunk capacity must be positive integer")
        if init_method == 'fork' and _b.HasCudaContext():
            raise RuntimeError(
                "Cannot fork process when there is CUDA context bound to it. "
                "Make sure you build pipeline before CUDA context is acquired or change Python workers "
                "starting method from ``fork`` to ``spawn`` (see Piepline's ``py_workers_init`` option). "
                "If you are trying to build multiple pipelines that use Python workers, you will need to "
                "call ``start_py_workers`` method on all of them before calling build method of any pipeline.")
        mp = multiprocessing.get_context(init_method)
        workers_num = workers_num or mp.cpu_count()
        if workers_num < 1:
            raise RuntimeError("workers_num must be a positive integer")
        self.workers_num = workers_num
        self.processes = []
        self.task_pipes = []
        self.res_pipes = []
        self.socks = []
        self.task_pipes_lock = threading.Lock()
        self.from_tracker = None
        self.to_tracker = None
        self.tracker_thread = None
        for i in range(self.workers_num):
            task_r, task_w = mp.Pipe(duplex=False)
            res_r, res_w = mp.Pipe(duplex=False)
            sock_reader, sock_writer = socket.socketpair(family=socket.AF_UNIX)
            process = mp.Process(
                target=worker,
                args=(i, callbacks, prefetch_queue_depths, initial_chunk_size,
                      task_r, res_w, sock_writer),
            )
            self.task_pipes.append(task_w)
            self.res_pipes.append(res_r)
            self.processes.append(process)
            self.socks.append(sock_reader)
        self._start_processes()

    def close(self):
        if self.tracker_thread is None:
            return
        try:
            self.to_tracker.send(None)
        except BrokenPipeError:
            """workers already exitied, tracker_thread finished its task and exitied closing the pipe"""
        self.tracker_thread.join()
        self.tracker_thread = None

    def _start_processes(self):
        try:
            for process in self.processes:
                process.start()
            from_tracker_r, from_tracker_w = multiprocessing.Pipe(duplex=False)
            to_tracker_r, to_tracker_w = multiprocessing.Pipe(duplex=False)
            self.from_tracker = from_tracker_r
            self.to_tracker = to_tracker_w
            # thread is properly joined in .close method, but it is run as daemon to prevent Python
            # from trying to join it automatically too early on cleanup
            self.tracker_thread = threading.Thread(
                target=join_processes, args=(
                    self.processes, to_tracker_r, from_tracker_w, self.task_pipes, self.task_pipes_lock
                ), daemon=True)
            self.tracker_thread.start()
        except:
            for proc in self.processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in self.processes:
                if proc.pid is not None:
                    proc.join()
            raise


class WorkersPool:
    """"Combines worker processes pool with callback contexts, can be used to schedule batches to be run on
the workers and to receive resulting batches from the workers.
"""

    def __init__(self, callbacks_num, pool):
        """
        Parameters
        ----------
        `callbacks_num` : int
            Number of callabacks that can be run in the workers, each callback will have separate
            context with dedicated shared memory pool.
        `pool` : ProcPool
            ProcPool instance enabling basic communication with worker processes.
        """
        self.contexts = [CallbackContext() for _ in range(callbacks_num)]
        self.pool = pool
        self.rec_pipes = self.pool.res_pipes + [self.pool.from_tracker]

    @classmethod
    def from_groups(cls, groups, keep_alive_queue_size, init_method="fork", workers_num=None, initial_chunk_size=1024 * 1024):
        """Creates new WorkerPool instance for given list of ExternalSource groups.

        Parameters
        ----------
        `groups` : _ExternalSourceGroup list
            List of external source groups.
        `workers_num` : int
            Number of workers to be created in ProcPool.
        `init_method` : str
            Method of starting worker processes, either fork or spawn.
        `keep_alive_queue_size` : int
            Number of the most recently produced batches whose underlying shared memory should
            remain untouched (because they might still be referenced further in the pipeline).
            Note that actual number of simultaneously kept batches will probably be greater
            because of additional prefetching.
        `initial_chunk_size` : int
            Initial size of each shared memory chunk.
        """
        callbacks = [group.callback for group in groups]
        queue_depths = [keep_alive_queue_size + group.prefetch_queue_depth for group in groups]
        pool = ProcPool(callbacks, queue_depths, workers_num, init_method, initial_chunk_size)
        return cls(len(callbacks), pool)

    def schedule_batch(self, context_i, batch_i, tasks):
        """Distribute `tasks` among workers to run them by calling `context_i`th callaback

        Parameters
        ----------
        `context_i` : int
            Specifies which callback will be used to run the task, it must be the index corresponding
            to the order of callbacks passed when constructing WorkerPool.
        `batch_i` : int
            Ordinal of the batch that tasks list corresponds to.
        `tasks` : nvidia.dali.types.SampleInfo list
            You can think of resulting batch as [callback(task) for task in tasks] with the exception that
            callbacks will be run in parallel.
        """
        tasks = list(enumerate(tasks))
        if not tasks:
            raise RuntimeError("Cannot schedule empty list of tasks")
        context = self.contexts[context_i]
        if batch_i in context.scheduled:
            raise RuntimeError("Given batch has already been scheduled")
        if context.iter_failed:
             # there is no point in scheduling anything for the context that has reached the end of data
             # or failed with error, once user receives batch that raised exception they should reset
             # the context before scheduling new tasks
            return
        self._distribute(context_i, batch_i, tasks)
        context.partially_received[batch_i] = {}
        context.scheduled[batch_i] = tasks

    def _distribute(self, context_i, batch_i, tasks):
        workers_num = self.pool.workers_num
        tasks_no = len(tasks)
        chunk_size = tasks_no // workers_num
        remainder = tasks_no % workers_num
        queued_no = 0
        with self.pool.task_pipes_lock:
            task_pipes = self.pool.task_pipes
            for worker_id in range(workers_num):
                worker_chunk = chunk_size + (worker_id < remainder)
                if worker_chunk == 0:
                    break
                scheduled_tasks = ScheduledTasks(context_i, batch_i, tasks[queued_no:queued_no + worker_chunk])
                queued_no += worker_chunk
                task_pipes[worker_id].send(scheduled_tasks)

    def receive_batch(self, context_i):
        """Returns the next produced batch (in the order of schedule_batch calls) for the ``context_i``th callaback.

        Parameters
        ----------
        `context_i` : int
            Specifies which callback you want the results from, ordering corresponds to the order of
            callbacks passed when constructing the pool.
        """
        context = self.contexts[context_i]
        assert len(context.scheduled) > 0, "No task has been scheduled"
        batch_i, tasks = context.scheduled.popitem(last=False)
        awaiting_batch = context.partially_received[batch_i]
        while len(awaiting_batch) < len(tasks) and batch_i not in context.iter_failed:
            self._receive_chunk()
        if batch_i in context.iter_failed:
            exception = context.iter_failed[batch_i]
            del context.iter_failed[batch_i]
            del context.partially_received[batch_i]
            raise exception
        res = [awaiting_batch[i] for i, _ in tasks]
        del context.partially_received[batch_i]
        return res

    def _receive_chunk(self):
        ready_workers = multiprocessing.connection.wait(self.rec_pipes)
        for worker_pipe in ready_workers:
            completed_tasks = worker_pipe.recv()
            if completed_tasks is None:
                raise RuntimeError("Worker exited unexpectedly")
            worker_id = completed_tasks.worker_id
            batch_i = completed_tasks.batch_i
            context = self.contexts[completed_tasks.context_i]
            if batch_i not in context.partially_received or batch_i in context.iter_failed:  # batch has been discarded
                continue
            # iteration failed with exception
            if completed_tasks.exception is not None:
                context.iter_failed[batch_i] = completed_tasks.exception
            else:
                worker_batch = context.batch_consumer.load_batch(
                    self.pool.socks[worker_id], completed_tasks.batch_serialized)
                context.partially_received[batch_i].update(worker_batch)

    def reset(self):
        for context in self.contexts:
            context.reset()

    def reset_context(self, context_i):
        self.contexts[context_i].reset()

    def close(self):
        self.pool.close()


def join_processes(processes, tracker_pipe, main_thread_pipe, task_pipes, task_pipes_lock):
    ps = {p.sentinel: p for p in processes}
    listen_for = list(ps.keys()) + [tracker_pipe]
    try:
        # Once one process exits stop the whole group (gracefully if possible)
        while True:
            sentinels = multiprocessing.connection.wait(listen_for)
            proc_sentinels = [s for s in sentinels if s != tracker_pipe]
            if tracker_pipe in sentinels or any(ps[sentinel].exitcode is not None for sentinel in proc_sentinels):
                break
        with task_pipes_lock:
            for proc, task_pipe in zip(processes, task_pipes):
                if proc.exitcode is None:
                    task_pipe.send(None)
    except:
        for proc in processes:
            if proc.exitcode is not None:
                proc.terminate()
        raise
    finally:
        for proc in processes:
            proc.join(1)
        for proc in processes:
            if proc.exitcode is None:
                proc.terminate()
                proc.join()
        # if workers exited unexpectedly main thread might be blocked waiting for the data;
        # let it know it is over
        main_thread_pipe.send(None)
