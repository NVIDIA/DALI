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
from nvidia.dali.pipeline import Pipeline, pipeline_def
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


def _generator_with_args(max_size, iters, seed=42):
    rng = np.random.default_rng(seed=seed)
    for _ in range(iters):
        size = rng.integers(1, max_size)
        yield np.array([rng.integers(0, 200, size=[size], dtype=np.int64)]), \
              np.array([size], dtype=np.int64)


@pipeline_def(num_threads=3, device_id=0)
def es_with_args_pipe(source_args, device, cycle):
    source_args['max_size'] = Pipeline.current().max_batch_size
    ret1, ret2 = fn.external_source(source=_generator_with_args, device=device, num_outputs=2,
                                    cycle=cycle, source_args=source_args)
    return ret1, ret2


def verify_pipeline(pipeline, device, iters, max_size):
    for ref in _generator_with_args(max_size, iters):
        val = pipeline.run()
        if device == "cpu":
            assert np.array_equal(val[0].as_array(), ref[0])
            assert np.array_equal(val[1].as_array(), ref[1])
        elif device == "gpu":
            assert np.array_equal(val[0].as_cpu().as_array(), ref[0])
            assert np.array_equal(val[1].as_cpu().as_array(), ref[1])
        else:
            assert False, "Unknown device"


def check_es_generator_with_args(device, cycle):
    iters = 5
    batch_size = 32
    source_args = {'iters': iters}
    pipe = es_with_args_pipe(source_args, device, cycle, batch_size=batch_size)
    pipe.build()

    verify_pipeline(pipe, device, iters, batch_size)
    if cycle:
        verify_pipeline(pipe, device, iters, batch_size)


def test_es_generator_with_args():
    devices = ["cpu", "gpu"]
    cycle = [True, False]
    for dv in devices:
        for cy in cycle:
            yield check_es_generator_with_args, dv, cy
