# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali._multiproc import shared_mem
from nvidia.dali._multiproc.messages import Structure, ShmMessage
from nvidia.dali._multiproc.shared_batch import _align_up as align_up


class QueueMeta(Structure):
    _fields = ("capacity", "i"), ("size", "i"), ("begining", "i"), ("is_closed", "i")


class ShmQueue:

    MSG_CLASS = ShmMessage
    ALIGN_UP_MSG = 4
    ALIGN_UP_BUFFER = 4096

    def __init__(self, mp, capacity):
        self.lock = mp.Lock()
        self.cv_empty = mp.Condition(self.lock)
        self.capacity = capacity
        self.meta = QueueMeta(capacity, 0, 0, 0)
        self.meta_capacity = align_up(self.meta.get_size(), self.ALIGN_UP_MSG)
        dummy_msg = self.MSG_CLASS()
        self.msg_capacity = align_up(dummy_msg.get_size(), self.ALIGN_UP_MSG)
        self.shm_min_capacity = align_up(self.meta_capacity + capacity * self.msg_capacity, self.ALIGN_UP_BUFFER)
        self.shm = shared_mem.SharedMem.allocate(self.shm_min_capacity)
        self.is_closed = False
        self.init_offsets()
        self.write_meta()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['msgs_offsets'] = None
        state['shm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.init_offsets()

    def init_offsets(self):
        self.msgs_offsets = [i * self.msg_capacity + self.meta_capacity for i in range(self.capacity)]

    def set_shm(self, shm):
        assert self.shm is None and shm.capacity >= self.shm_min_capacity
        self.shm = shm

    def read_meta(self):
        self.meta.unpack_from(self.shm.buf, 0)

    def write_meta(self):
        self.meta.pack_into(self.shm.buf, 0)

    def read_msg(self, i):
        offset = self.msgs_offsets[i]
        msg = self.MSG_CLASS()
        msg.unpack_from(self.shm.buf, offset)
        return msg

    def write_msg(self, i, msg):
        offset = self.msgs_offsets[i]
        msg.pack_into(self.shm.buf, offset)

    def seal(self):
        self.shm.seal()

    def close(self):
        if self.is_closed:
            return
        with self.lock:
            self.read_meta()
            self.is_closed = True
            if not self.meta.is_closed:
                self.meta.is_closed = 1
                self.write_meta()
                # Notify only one waiting worker about closing of the queue, the woken up worker
                # will notify the next one. Avoid notify_all at this point, due to possible deadlock if
                # one of the notified workers exited abraptly when waiting on cv_empty without proper releasing
                # of the underlying semaphore.
                self.cv_empty.notify()

    def put(self, msgs):
        if self.is_closed or not msgs:
            return
        with self.lock:
            self.read_meta()
            assert self.meta.size + len(msgs) <= self.meta.capacity
            if self.meta.is_closed:
                self.is_closed = True
                return
            msgs_len = len(msgs)
            next_slot = (self.meta.begining + self.meta.size) % self.meta.capacity
            for msg in msgs:
                self.write_msg(next_slot, msg)
                next_slot = (next_slot + 1) % self.meta.capacity
            self.meta.size += msgs_len
            self.write_meta()
            self.cv_empty.notify()
        return msgs_len

    def _recv_samples(self, num_samples):
        num_take = self.meta.size
        if num_samples is not None and num_samples < num_take:
            num_take = num_samples
        recv = [self.read_msg((self.meta.begining + i) % self.meta.capacity) for i in range(num_take)]
        self.meta.size -= num_take
        self.meta.begining = (self.meta.begining + num_take) % self.meta.capacity
        self.write_meta()
        return recv

    def _wait_for_samples(self):
        waited = False
        self.read_meta()
        while not self.meta.size > 0 and not self.meta.is_closed:
            self.cv_empty.wait()
            waited = True
            self.read_meta()
        return waited

    def get(self, num_samples=1, get_if_waited=None):
        if self.is_closed:
            return
        with self.cv_empty:
            waited = self._wait_for_samples()
            if self.meta.is_closed:
                self.is_closed = True
                self.cv_empty.notify()
                return
            if waited and get_if_waited is not None and not get_if_waited():
                recv = []
            else:
                recv = self._recv_samples(num_samples)
            if self.meta.size > 0:
                self.cv_empty.notify()
        return recv


class Dispatcher:

    def __init__(self, queue):
        self.pending_cv = threading.Condition()
        self.pending = []
        self.queue = queue
        self.thread = None

    def close(self):
        if self.queue is not None:
            self.queue.close()
            self.queue = None
        if self.thread is not None:
            self.append(None)
            self.thread.join()
            self.thread = None

    def start_thread(self, worker):
        thread = threading.Thread(target=worker.dispatch_loop, daemon=True)
        thread.start()
        self.thread = thread

    def extend(self, msgs):
        with self.pending_cv:
            self.pending.extend(msgs)
            self.pending_cv.notify()

    def append(self, msg):
        with self.pending_cv:
            self.pending.append(msg)
            self.pending_cv.notify()


class DispatcherWorker:

    def __init__(self, pending_cv, pending, queue):
        self.pending_cv = pending_cv
        self.pending = pending
        self.queue = queue
        self.is_interrupted_by_sender = False
        self.is_queue_closed = False

    def dispatch_loop(self):
        try:
            while True:
                with self.pending_cv:
                    while len(self.pending) == 0:
                        self.pending_cv.wait()
                    msgs = list(self.pending)
                    self.pending.clear()
                if any(msg is None for msg in msgs):
                    self.is_interrupted_by_sender = True
                    break
                if self.send(msgs) is None:
                    self.is_queue_closed = True
                    break
        finally:
            self.close()

    def send(self, msgs):
        msgs = self.serialize_msgs(msgs)
        return self.queue.put(msgs)

    def close(self):
        raise NotImplementedError

    def serialize_msgs(self, msgs):
        raise NotImplementedError
