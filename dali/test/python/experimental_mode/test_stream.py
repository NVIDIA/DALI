# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali.backend as _backend
from ndd_utils import cuda_launch_host_func
from nose_utils import SkipTest, attr


def test_stream_comparison():
    s1 = ndd.stream(device_id=0)
    s2 = ndd.stream()
    assert s1.device_id == 0
    assert s2.device_id == 0
    assert s1 != s2
    assert s2 != s1
    assert s1 == s1.handle
    assert s2 == s2.handle
    assert s1 == s1
    assert s2 == s2
    assert ndd.stream(stream=1) == 1


@attr("multi_gpu")
def test_stream_device_id():
    if _backend.GetCUDADeviceCount() < 2:
        raise SkipTest("At least 2 devices needed for the test")
    try:
        _backend.SetCUDACurrentDevice(0)
        s0x = ndd.stream()
        assert s0x.device_id == 0, "The stream constructor doesn't respect current CUDA device"

        legacy0 = ndd.stream(stream=0)
        assert legacy0.device_id == 0
        s00 = ndd.stream(device_id=0)
        s01 = ndd.stream(device_id=1)
        assert s00.device_id == 0
        assert s01.device_id == 1

        _backend.SetCUDACurrentDevice(1)

        assert s0x.device_id == 0  # the device_id should not change, even if it wasn't explicit
        assert legacy0.device_id == 0  # the device_id should not change, even in a legacy stream

        assert s00.device_id == 0
        assert s01.device_id == 1

        s1x = ndd.stream()  # the default device is now 1
        assert s0x != s1x  # just for sanity
        assert s1x.device_id == 1, "The stream constructor doesn't respect current CUDA device"
        legacy1 = ndd.stream(stream=0)
        assert legacy0 != legacy1  # Legacy default streams on different devices are not equal

        s10 = ndd.stream(device_id=0)
        s11 = ndd.stream(device_id=1)
        assert s10.device_id == 0
        assert s11.device_id == 1

        _backend.SetCUDACurrentDevice(1)
        assert s0x.device_id == 0
        assert s1x.device_id == 1
        assert s00.device_id == 0
        assert s01.device_id == 1
        assert s10.device_id == 0
        assert s11.device_id == 1

    finally:
        _backend.SetCUDACurrentDevice(0)


@attr("pytorch")
def test_stream_torch():
    import torch

    ts = torch.cuda.Stream()
    ds = ndd.stream(stream=ts)
    ds2 = ndd.stream()
    assert ds == ts
    # assert ts == ds  # torch returns False instead of NotImplemented
    assert ds != ds2


def test_stream_synchronize():
    stream = ndd.stream()
    callback_started = threading.Event()
    allow_finish = threading.Event()
    sync_returned = False
    error = False

    def callback(_):
        nonlocal error
        callback_started.set()
        if not allow_finish.wait(1):
            error = True

    def worker():
        nonlocal error
        if not callback_started.wait(1):
            error = True
        if sync_returned:
            error = True
        allow_finish.set()

    cuda_launch_host_func(stream, callback)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    stream.synchronize()
    sync_returned = True

    thread.join()
    assert allow_finish.is_set()
    assert not error
