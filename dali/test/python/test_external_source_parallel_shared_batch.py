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


import os
import multiprocessing
import socket
from contextlib import closing
import numpy as np

from nvidia.dali._multiproc.shared_batch import BufShmChunk, SharedBatchWriter, SharedBatchMeta, deserialize_batch
from nvidia.dali._multiproc.shared_queue import ShmQueue
from nvidia.dali._multiproc.messages import ShmMessageDesc

from test_utils import RandomlyShapedDataIterator
from nose_utils import raises


def check_serialize_deserialize(batch):
    shm_chunk = BufShmChunk.allocate("chunk_0", 100)
    with closing(shm_chunk) as shm_chunk:
        writer = SharedBatchWriter(shm_chunk, batch)
        batch_meta = SharedBatchMeta.from_writer(writer)
        deserialized_batch = deserialize_batch(shm_chunk, batch_meta)
        assert len(batch) == len(
            deserialized_batch), "Lengths before and after should be the same"
        for i in range(len(batch)):
            np.testing.assert_array_equal(batch[i], deserialized_batch[i])


def test_serialize_deserialize():
    for shapes in [[(10)], [(10, 20)], [(10, 20, 3)], [(1), (2)], [(2), (2, 3)],
                   [(2, 3, 4), (2, 3, 5), (3, 4, 5)], []]:
        for dtype in [np.int8, np.float, np.int32]:
            yield check_serialize_deserialize, [np.full(s, 42, dtype=dtype) for s in shapes]


def test_serialize_deserialize_random():
    for max_shape in [(12, 200, 100, 3), (200, 300, 3), (300, 2)]:
        for dtype in [np.uint8, np.float]:
            rsdi = RandomlyShapedDataIterator(10, max_shape=max_shape, dtype=dtype)
            for i, batch in enumerate(rsdi):
                if i == 10:
                    break
                yield check_serialize_deserialize, batch


def worker(start_method, sock, task_queue, res_queue, worker_cb, worker_params):
    if start_method == "spawn":
        task_queue.open_shm(multiprocessing.reduction.recv_handle(sock))
        res_queue.open_shm(multiprocessing.reduction.recv_handle(sock))
        sock.close()
    while True:
        if worker_cb(task_queue, res_queue, **worker_params) is None:
            break


def setup_queue_and_worker(start_method, capacity, worker_cb, worker_params):
    mp = multiprocessing.get_context(start_method)
    task_queue = ShmQueue(mp, capacity)
    res_queue = ShmQueue(mp, capacity)
    if start_method == "spawn":
        socket_r, socket_w = socket.socketpair()
    else:
        socket_r = None
    proc = mp.Process(target=worker, args=(start_method, socket_r, task_queue, res_queue, worker_cb, worker_params))
    proc.start()
    if start_method == "spawn":
        pid = os.getppid()
        multiprocessing.reduction.send_handle(socket_w, task_queue.shm.handle, pid)
        multiprocessing.reduction.send_handle(socket_w, res_queue.shm.handle, pid)
    return proc, task_queue, res_queue


def _put_msgs(queue, msgs, one_by_one):
    if not one_by_one:
        queue.put(msgs)
    else:
        for msg in msgs:
            queue.put([msg])

def test_queue_full_assertion():
    for start_method in ("spawn", "fork"):
        for capacity in [1, 4]:
            for one_by_one in (True, False):
                mp = multiprocessing.get_context(start_method)
                queue = ShmQueue(mp, capacity)
                msgs = [ShmMessageDesc(i, i, i, i, i) for i in range(capacity + 1)]
                yield raises(RuntimeError, "The queue is full")(_put_msgs), queue, msgs, one_by_one


def copy_callback(task_queue, res_queue, num_samples):
    msgs = task_queue.get(num_samples=num_samples)
    if msgs is None:
        return
    assert len(msgs) > 0
    res_queue.put(msgs)
    return msgs


def _test_queue_recv(start_method, worker_params, capacity, send_msgs, recv_msgs, send_one_by_one):
    count = 0
    def next_i():
        nonlocal count
        count += 1
        return count
    proc, task_queue, res_queue = setup_queue_and_worker(start_method, capacity, copy_callback, worker_params)
    all_msgs = []
    received = 0
    for send_msg, recv_msg in zip(send_msgs, recv_msgs):
        msgs = [ShmMessageDesc(next_i(), -next_i(), -next_i(), next_i(), -next_i()) for i in range(send_msg)]
        all_msgs.extend(msgs)
        _put_msgs(task_queue, msgs, send_one_by_one)
        for _ in range(recv_msg):
            [recv_msg] = res_queue.get()
            msg_values = all_msgs[received].get_values()
            received += 1
            recv_msg_values = recv_msg.get_values()
            assert len(msg_values) == len(recv_msg_values)
            assert all(msg_value == recv_msg_value for msg_value, recv_msg_value in zip(msg_values, recv_msg_values))
    if not proc.exitcode:
        task_queue.close()
    proc.join()
    assert proc.exitcode == 0


def test_queue_recv():
    capacities = [1, 13, 20, 100]
    send_msgs = [(1, 1, 1), (7, 6, 5), (19, 5, 4, 9), (100, 100, 5)]
    recv_msgs = [(1, 1, 1), (5, 1, 12), (19, 1, 5, 12), (100, 95, 10)]
    for start_method in ("spawn", "fork"):
        for capacity, send_msg, recv_msg in zip(capacities, send_msgs, recv_msgs):
            for send_one_by_one in (True, False):
                for worker_params in ({'num_samples': 1}, {'num_samples': None}):
                    yield _test_queue_recv, start_method, worker_params, capacity, send_msg, recv_msg, send_one_by_one
