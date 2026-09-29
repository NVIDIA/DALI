# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pickle
import struct
import socket
from types import SimpleNamespace
from contextlib import closing, contextmanager
import numpy as np

from nvidia.dali._multiproc.shared_batch import (
    BufShmChunk,
    SampleMeta,
    SharedBatchWriter,
    SharedBatchMeta,
    deserialize_batch,
    deserialize_message,
    deserialize_samples_meta,
    read_shm_message,
    serialize_message,
    serialize_samples_meta,
    write_shm_message,
)
from nvidia.dali._multiproc.shared_queue import ShmQueue
from nvidia.dali._multiproc.messages import (
    CompletedTask,
    SampleRange,
    ScheduledTask,
    ShmMessageDesc,
    TaskArgs,
)
from nvidia.dali.types import BatchInfo

from test_utils import RandomlyShapedDataIterator
from nose_utils import raises


class UnsafePicklePayload:
    def __reduce__(self):
        return eval, ("1 + 1",)  # nosec B307


def test_message_deserialization_rejects_pickle():
    payload = pickle.dumps(UnsafePicklePayload())
    raises(RuntimeError, "Malformed shared memory message")(deserialize_message)(payload)
    raises(RuntimeError, "Malformed shared memory message")(deserialize_samples_meta)(payload)


def test_samples_meta_deserialization_rejects_malformed_data():
    too_deep = struct.pack("<Q", 1) + (b"\x01" + struct.pack("<Q", 1)) * 100
    raises(RuntimeError, "nesting is too deep")(deserialize_samples_meta)(too_deep)
    bad_dtype = struct.pack("<QBQQQ", 1, 0, 0, 1, 3) + b"xyz" + struct.pack("<I", 0)
    raises(RuntimeError, "invalid sample data type")(deserialize_samples_meta)(bad_dtype)
    object_dtype = struct.pack("<QBQQQ", 1, 0, 0, 8, 3) + b"|O8" + struct.pack("<I", 0)
    raises(RuntimeError, "object data type")(deserialize_samples_meta)(object_dtype)


def test_message_deserialization_rejects_malformed_data():
    task = ScheduledTask(0, 1, 2, TaskArgs.make_sample(SampleRange(0, 8, 0, 0)))
    serialized = serialize_message(task)
    for truncated in [b"", serialized[:1], serialized[:-1]]:
        raises(RuntimeError, "unexpected end of data")(deserialize_message)(truncated)
    raises(RuntimeError, "trailing data")(deserialize_message)(serialized + b"\0")
    raises(RuntimeError, "unexpected tag")(deserialize_message)(b"\xff" + serialized[1:])


def test_samples_meta_rejects_object_dtype():
    np_dtype = np.dtype(np.int32)
    serialized = serialize_samples_meta([SampleMeta(0, (2,), np_dtype, 8)])
    malformed = serialized.replace(np_dtype.str.encode(), b"|O\0")
    assert len(malformed) == len(serialized)
    raises(RuntimeError, "Malformed shared memory message")(deserialize_samples_meta)(malformed)


def _check_scheduled_task_roundtrip(task):
    decoded = deserialize_message(serialize_message(task))
    assert type(decoded) is ScheduledTask
    assert decoded.context_i == task.context_i
    assert decoded.scheduled_i == task.scheduled_i
    assert decoded.epoch_start == task.epoch_start
    assert decoded.task.minibatch_i == task.task.minibatch_i
    if task.task.is_sample_mode():
        assert decoded.task.is_sample_mode()
        expected = [vars(si) for si in task.task.sample_range]
        assert [vars(si) for si in decoded.task.sample_range] == expected
    else:
        assert not decoded.task.is_sample_mode()
        assert len(decoded.task.batch_args) == len(task.task.batch_args)
        for decoded_arg, arg in zip(decoded.task.batch_args, task.task.batch_args):
            assert type(decoded_arg) is type(arg)
            if isinstance(arg, BatchInfo):
                assert vars(decoded_arg) == vars(arg)
            else:
                assert decoded_arg == arg


def test_scheduled_task_roundtrip():
    sample_range = SampleRange(16, 32, 1, 3)
    for task_args in [
        TaskArgs.make_sample(sample_range),
        TaskArgs(2, sample_range=sample_range[4:12]),
        TaskArgs.make_batch(()),
        TaskArgs.make_batch((7,)),
        TaskArgs.make_batch((BatchInfo(5, 2),)),
    ]:
        yield _check_scheduled_task_roundtrip, ScheduledTask(3, 11, 2, task_args)


def test_scheduled_task_rejects_unsupported_batch_args():
    task = ScheduledTask(0, 0, 0, TaskArgs.make_batch(("not an int",)))
    raises(TypeError, "Unsupported batch argument type")(serialize_message)(task)


def test_completed_task_roundtrip():
    processed = SimpleNamespace(context_i=1, scheduled_i=2, minibatch_i=3)
    done = CompletedTask.done(4, processed, SharedBatchMeta(128, 256))
    decoded = deserialize_message(serialize_message(done))
    assert type(decoded) is CompletedTask
    assert (decoded.worker_id, decoded.context_i, decoded.scheduled_i, decoded.minibatch_i) == (
        4,
        1,
        2,
        3,
    )
    assert (decoded.batch_meta.meta_offset, decoded.batch_meta.meta_size) == (128, 256)
    assert not decoded.is_failed() and decoded.traceback_str is None


class CustomStopIteration(StopIteration):
    pass


class CustomError(Exception):
    pass


def _check_failed_task_roundtrip(exception, expected_type):
    processed = SimpleNamespace(
        context_i=0,
        scheduled_i=0,
        minibatch_i=0,
        exception=exception,
        traceback_str="Traceback: \u017c\u00f3\u0142w",
    )
    completed = CompletedTask.failed(0, processed)
    decoded = deserialize_message(serialize_message(completed))
    assert type(decoded.exception) is expected_type
    assert str(decoded.exception) == str(exception)
    assert decoded.traceback_str == processed.traceback_str
    assert decoded.batch_meta is None


def test_failed_task_roundtrip():
    for exception, expected_type in [
        (CustomStopIteration("done"), StopIteration),
        (StopIteration(), StopIteration),
        (CustomError("custom"), RuntimeError),
        (ValueError("value"), RuntimeError),
    ]:
        yield _check_failed_task_roundtrip, exception, expected_type


def test_shm_message_roundtrip():
    shm_chunk = BufShmChunk.allocate(0, 4096)
    with closing(shm_chunk) as shm_chunk:
        task = ScheduledTask(1, 2, 3, TaskArgs.make_sample(SampleRange(0, 4, 0, 0)))
        msg_desc = write_shm_message(-1, shm_chunk, task, 0, resize=False)
        decoded = read_shm_message(shm_chunk, msg_desc)
        assert type(decoded) is ScheduledTask
        assert (decoded.context_i, decoded.scheduled_i, decoded.epoch_start) == (1, 2, 3)


def check_serialize_deserialize(batch):
    shm_chunk = BufShmChunk.allocate("chunk_0", 100)
    with closing(shm_chunk) as shm_chunk:
        writer = SharedBatchWriter(shm_chunk, batch)
        batch_meta = SharedBatchMeta.from_writer(writer)
        deserialized_batch = deserialize_batch(shm_chunk, batch_meta)
        assert len(batch) == len(deserialized_batch), "Lengths before and after should be the same"
        for i in range(len(batch)):
            np.testing.assert_array_equal(batch[i], deserialized_batch[i])


def test_serialize_deserialize():
    for shapes in [
        [(10,)],
        [(10, 20)],
        [(10, 20, 3)],
        [(1), (2)],
        [(2), (2, 3)],
        [(2, 3, 4), (2, 3, 5), (3, 4, 5)],
        [],
    ]:
        for dtype in [np.int8, float, np.int32, np.bool_, np.float16, np.complex64, ">u2"]:
            yield check_serialize_deserialize, [np.full(s, 42, dtype=dtype) for s in shapes]


def test_serialize_deserialize_nested():
    batch = [
        (np.full((2, 3), 1, dtype=np.uint8), np.full((), 3, dtype=np.int64)),
        [np.full((0, 5), 4, dtype=np.float32), np.full((1,), 5.5)],
    ]
    shm_chunk = BufShmChunk.allocate(0, 100)
    with closing(shm_chunk) as shm_chunk:
        writer = SharedBatchWriter(shm_chunk, batch)
        deserialized_batch = deserialize_batch(shm_chunk, SharedBatchMeta.from_writer(writer))
        assert len(deserialized_batch) == len(batch)
        for sample, deserialized in zip(batch, deserialized_batch):
            assert type(deserialized) is type(sample)
            for part, deserialized_part in zip(sample, deserialized):
                assert part.dtype == deserialized_part.dtype
                np.testing.assert_array_equal(part, deserialized_part)


def test_serialize_deserialize_random():
    for max_shape in [(12, 200, 100, 3), (200, 300, 3), (300, 2)]:
        for dtype in [np.uint8, float]:
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


@contextmanager
def setup_queue_and_worker(start_method, capacity, worker_cb, worker_params):
    mp = multiprocessing.get_context(start_method)
    task_queue = ShmQueue(mp, capacity)
    res_queue = ShmQueue(mp, capacity)
    if start_method == "spawn":
        socket_r, socket_w = socket.socketpair()
    else:
        socket_r = None
    proc = mp.Process(
        target=worker,
        args=(start_method, socket_r, task_queue, res_queue, worker_cb, worker_params),
    )
    proc.start()
    try:
        if start_method == "spawn":
            pid = os.getppid()
            multiprocessing.reduction.send_handle(socket_w, task_queue.shm.handle, pid)
            multiprocessing.reduction.send_handle(socket_w, res_queue.shm.handle, pid)
        yield task_queue, res_queue
    finally:
        if not proc.exitcode:
            res_queue.close()
            task_queue.close()
        proc.join()
        assert proc.exitcode == 0


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

    with setup_queue_and_worker(start_method, capacity, copy_callback, worker_params) as (
        task_queue,
        res_queue,
    ):
        all_msgs = []
        received = 0
        for send_msg, recv_msg in zip(send_msgs, recv_msgs):
            msgs = [
                ShmMessageDesc(next_i(), -next_i(), next_i(), next_i(), next_i())
                for i in range(send_msg)
            ]
            all_msgs.extend(msgs)
            _put_msgs(task_queue, msgs, send_one_by_one)
            for _ in range(recv_msg):
                [recv_msg] = res_queue.get()
                msg_values = all_msgs[received].get_values()
                received += 1
                recv_msg_values = recv_msg.get_values()
                assert len(msg_values) == len(recv_msg_values)
                assert all(msg == recv_msg for msg, recv_msg in zip(msg_values, recv_msg_values))


def test_queue_recv():
    capacities = [1, 13, 20, 100]
    send_msgs = [(1, 1, 1), (7, 6, 5), (19, 5, 4, 9), (100, 100, 5)]
    recv_msgs = [(1, 1, 1), (5, 1, 12), (19, 1, 5, 12), (100, 95, 10)]
    for start_method in ("spawn", "fork"):
        for capacity, send_msg, recv_msg in zip(capacities, send_msgs, recv_msgs):
            for send_one_by_one in (True, False):
                for worker_params in ({"num_samples": 1}, {"num_samples": None}):
                    yield (
                        _test_queue_recv,
                        start_method,
                        worker_params,
                        capacity,
                        send_msg,
                        recv_msg,
                        send_one_by_one,
                    )


def _test_queue_large(start_method, msg_values):
    with setup_queue_and_worker(
        start_method, len(msg_values), copy_callback, {"num_samples": None}
    ) as (task_queue, res_queue):
        msg_instances = [ShmMessageDesc(*values) for values in msg_values]
        _put_msgs(task_queue, msg_instances, False)
        for values in msg_values:
            [recv_msg] = res_queue.get()
            recv_msg_values = recv_msg.get_values()
            assert len(values) == len(recv_msg_values)
            assert all(msg == recv_msg for msg, recv_msg in zip(values, recv_msg_values))


def test_queue_large():
    max_int32 = 2**31 - 1
    max_uint32 = 2**32 - 1
    max_uint64 = 2**64 - 1
    msgs = [
        (max_int32, max_int32, max_int32, max_int32, max_int32),
        (max_int32, max_int32, max_uint32, max_uint32, max_uint32),
        (max_int32, max_int32, max_uint64, max_uint64, max_uint64),
    ]
    for start_method in ("spawn", "fork"):
        for msg in msgs:
            yield _test_queue_large, start_method, [msg]


def test_queue_large_failure():
    max_int32 = 2**31 - 1
    max_uint32 = 2**32 - 1
    error_message = (
        "Failed to serialize object as C-like structure. " "Tried to populate following fields:"
    )
    for start_method in ("spawn", "fork"):
        yield raises(RuntimeError, error_message)(_test_queue_large), start_method, [
            (max_int32 + 1, 0, max_uint32, max_uint32, max_uint32)
        ]
        yield raises(RuntimeError, error_message)(_test_queue_large), start_method, [
            (max_int32, max_int32, -1, 0, 0)
        ]
