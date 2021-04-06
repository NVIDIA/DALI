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

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
from nvidia.dali.pipeline import Pipeline
from test_utils import check_batch

def build_src_pipe(device, layout = None):
    if layout is None:
        layout = "XY"
    batches = [[
        np.array([[1,2,3],[4,5,6]], dtype = np.float32),
        np.array([[10,20], [30,40], [50,60]], dtype = np.float32)
    ],
    [
        np.array([[9,10],[11,12]], dtype = np.float32),
        np.array([[100,200,300,400,500]], dtype = np.float32)
    ]]

    src_pipe = Pipeline(len(batches), 1, 0)
    src_pipe.set_outputs(fn.external_source(source=batches, device=device, cycle=True, layout=layout))
    src_pipe.build()
    return src_pipe, len(batches)

def _test_feed_input(device):
    src_pipe, batch_size = build_src_pipe(device)

    dst_pipe = Pipeline(batch_size, 1, 0, exec_async=False, exec_pipelined=False)
    dst_pipe.set_outputs(fn.external_source(name="ext", device=device))
    dst_pipe.build()
    for iter in range(3):
        out1 = src_pipe.run()
        dst_pipe.feed_input("ext", out1[0])
        out2 = dst_pipe.run()
        check_batch(out2[0], out1[0], batch_size, 0, 0, "XY")


def test_feed_input():
    for device in ["cpu", "gpu"]:
        yield _test_feed_input, device


def _test_callback(device, as_tensors, change_layout_to = None):
    src_pipe, batch_size = build_src_pipe(device)
    ref_pipe, batch_size = build_src_pipe(device, layout=change_layout_to)

    dst_pipe = Pipeline(batch_size, 1, 0)
    def get_from_src():
        tl = src_pipe.run()[0]
        return [tl[i] for i in range(len(tl))] if as_tensors else tl

    dst_pipe.set_outputs(fn.external_source(source=get_from_src, device=device, layout=change_layout_to))
    dst_pipe.build()

    for iter in range(3):
        ref = ref_pipe.run()
        out = dst_pipe.run()
        check_batch(out[0], ref[0], batch_size, 0, 0)

def test_callback():
    for device in ["cpu", "gpu"]:
        for as_tensors in [False, True]:
            for change_layout in [None, "AB"]:
                yield _test_callback, device, as_tensors, change_layout

def _test_scalar(device, as_tensors):
    """Test propagation of scalars from external source"""
    batch_size = 4
    src_pipe = Pipeline(batch_size, 1, 0)
    src_ext = fn.external_source(source=lambda i: [np.float32(i * 10 + i + 1) for i in range(batch_size)], device=device)
    src_pipe.set_outputs(src_ext)

    src_pipe.build()
    dst_pipe = Pipeline(batch_size, 1, 0, exec_async=False, exec_pipelined=False)
    dst_pipe.set_outputs(fn.external_source(name="ext", device=device))
    dst_pipe.build()

    for iter in range(3):
        src = src_pipe.run()
        data = src[0]
        if as_tensors:
            data = [data[i] for i in range(len(data))]
        dst_pipe.feed_input("ext", data)
        dst = dst_pipe.run()
        check_batch(src[0], dst[0], batch_size, 0, 0, "")

def test_scalar():
    for device in ["cpu", "gpu"]:
        for as_tensors in [False, True]:
            yield _test_scalar, device, as_tensors
