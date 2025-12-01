# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List, Optional
import os
import threading
from nvidia.dali._multiproc import shared_mem
from nvidia.dali._multiproc.messages import Structure, ShmMessageDesc
from nvidia.dali._multiproc.shared_batch import _align_up as align_up


class QueueMeta(Structure):
    _fields = ("capacity", "i"), ("size", "i"), ("begining", "i"), ("is_closed", "i")


class ShmQueue:
    """
    Simple fixed capacity shared memory queue of fixed size messages.
    Writing to a full queue fails, attempt to get from an empty queue blocks until data is
    available or the queue is closed.
    """

    MSG_CLASS = ShmMessageDesc
    ALIGN_UP_MSG = 4
    ALIGN_UP_BUFFER = 4096

    def __init__(self, mp, capacity):
        self.lock = mp.Lock()
        self.cv_not_empty = mp.Condition(self.lock)
        self.capacity = capacity
        self.meta = QueueMeta(capacity, 0, 0, 0)
        self.meta_size = align_up(self.meta.get_size(), self.ALIGN_UP_MSG)
        dummy_msg = self.MSG_CLASS()
        self.msg_size = align_up(dummy_msg.get_size(), self.ALIGN_UP_MSG)
        self.shm_capacity = align_up(
            self.meta_size + capacity * self.msg_size, self.ALIGN_UP_BUFFER
        )
        self.shm = shared_mem.SharedMem.allocate(self.shm_capacity)
        self.is_closed = False
        self._init_offsets()
        self._write_meta()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["msgs_offsets"] = None
        state["shm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_offsets()

    def _init_offsets(self):
        self.msgs_offsets = [i * self.msg_size + self.meta_size for i in range(self.capacity)]

    def _read_meta(self):
        self.meta.unpack_from(self.shm.buf, 0)

    def _write_meta(self):
        self.meta.pack_into(self.shm.buf, 0)

    def _read_msg(self, i):
        offset = self.msgs_offsets[i]
        msg = self.MSG_CLASS()
        msg.unpack_from(self.shm.buf, offset)
        return msg

    def _write_msg(self, i, msg):
        offset = self.msgs_offsets[i]
        msg.pack_into(self.shm.buf, offset)

    def _recv_samples(self, num_samples):
        num_take = self.meta.size
        if num_samples is not None and num_samples < num_take:
            num_take = num_samples
        recv = [
            self._read_msg((self.meta.begining + i) % self.meta.capacity) for i in range(num_take)
        ]
        self.meta.size -= num_take
        self.meta.begining = (self.meta.begining + num_take) % self.meta.capacity
        self._write_meta()
        return recv

    def _wait_for_samples(self):
        waited = False
        self._read_meta()
        while not self.meta.size > 0 and not self.meta.is_closed:
            self.cv_not_empty.wait()
            waited = True
            self._read_meta()
        return waited

    def open_shm(self, handle, close_handle=True):
        try:
            shm = shared_mem.SharedMem.open(handle, self.shm_capacity)
            self.shm = shm
            if close_handle:
                shm.close_handle()
        except:  # noqa: E722
            if close_handle:
                os.close(handle)
            raise

    def close_handle(self):
        self.shm.close_handle()

    def close(self):
        if self.is_closed:
            return
        with self.lock:
            self._read_meta()
            self.is_closed = True
            if not self.meta.is_closed:
                self.meta.is_closed = 1
                self._write_meta()
                # Notify only one waiting worker about closing the queue, the woken up worker
                # will notify the next one. Avoid notify_all at this point, due to possible
                # deadlock if one of the notified workers exited abruptly when waiting
                # on cv_not_empty without proper releasing of the underlying semaphore.
                self.cv_not_empty.notify()

    def put(self, msgs: List[MSG_CLASS]) -> Optional[int]:
        assert len(msgs), "Cannot write an empty list of messages"
        if self.is_closed:
            return
        with self.lock:
            self._read_meta()
            if self.meta.size + len(msgs) > self.meta.capacity:
                raise RuntimeError("The queue is full")
            if self.meta.is_closed:
                self.is_closed = True
                return
            msgs_len = len(msgs)
            next_slot = (self.meta.begining + self.meta.size) % self.meta.capacity
            for msg in msgs:
                self._write_msg(next_slot, msg)
                next_slot = (next_slot + 1) % self.meta.capacity
            self.meta.size += msgs_len
            self._write_meta()
            self.cv_not_empty.notify()
        return msgs_len

    def get(self, num_samples=1, predicate=None) -> Optional[List[MSG_CLASS]]:
        """
        Args:
        ----------
        num_samples : optional positive integer
            Maximal number of messages to take from the queue, if set to None all available messages
            will be taken. The call blocks until there are any messages available.
            It may return less than `num_samples`, but an empty list is returned only if `predicate`
            was specified and it evaluated to False after waiting on empty queue.
            The call returns None iff the queue was closed.
        predicate : a parameterless callable
            Used for double-checking if the item should really be taken after waiting on empty
            queue.
        """
        if self.is_closed:
            return
        with self.cv_not_empty:  # equivalent to `with self.lock`
            waited = self._wait_for_samples()
            if self.meta.is_closed:
                self.is_closed = True
                self.cv_not_empty.notify()
                return
            if waited and predicate is not None and not predicate():
                recv = []
            else:
                recv = self._recv_samples(num_samples)
            if self.meta.size > 0:
                self.cv_not_empty.notify()
        return recv


class Dispatcher:
    """Wrapper around the queue that enables writing to the queue in a separate thread, just in
    case a writing process would have to wait too long for a lock on the queue when multiple
    readers pop the items one by one."""

    def __init__(self, target_queue, on_thread_exit=None):
        self.pending_cv = threading.Condition()
        self.pending = []
        self.target_queue = target_queue
        self.on_thread_exit = on_thread_exit
        self.thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.thread.start()

    def _dispatch_loop(self):
        try:
            while True:
                with self.pending_cv:
                    while len(self.pending) == 0:
                        self.pending_cv.wait()
                    msgs = list(self.pending)
                    self.pending.clear()
                if any(msg is None for msg in msgs):
                    break
                msgs = self.serialize_msgs(msgs)
                if self.target_queue.put(msgs) is None:
                    break
        finally:
            if self.on_thread_exit is not None:
                self.on_thread_exit()

    def close(self):
        self.target_queue.close()
        self.stop_thread()

    def stop_thread(self):
        if self.thread is not None:
            self.append(None)
            self.thread.join()
            self.thread = None

    def extend(self, msgs):
        with self.pending_cv:
            self.pending.extend(msgs)
            self.pending_cv.notify()

    def append(self, msg):
        with self.pending_cv:
            self.pending.append(msg)
            self.pending_cv.notify()

    def serialize_msgs(self, msgs):
        raise NotImplementedError
