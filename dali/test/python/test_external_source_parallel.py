# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
from nose.tools import raises

from test_utils import compare_pipelines
from test_external_source_parallel_utils import *


def test_parallel_fork_cpu_only():
    pipeline_pairs = 4
    batch_size = 10
    iters = 4
    callback = ExtCallback((4, 5), iters * batch_size, np.int32)
    parallel_pipes = [create_pipe(callback, 'cpu', batch_size, py_workers_num=4,
                                  py_workers_init='fork', parallel=True, device_id=None)
                      for i in range(2 * pipeline_pairs)]
    for i in range(pipeline_pairs):
        parallel_pipes[2 * i].build()
        parallel_pipes[2 * i + 1].build()
        compare_pipelines(parallel_pipes[2 * i], parallel_pipes[2 * i + 1], batch_size, iters)


def test_parallel_fork():
    epoch_size = 250
    callback = ExtCallback((4, 5), epoch_size, np.int32)
    pipes = [(
        create_pipe(
            callback, 'cpu', batch_size, py_workers_num=workers_num, py_workers_init='fork',
            parallel=True),
        create_pipe(callback, 'cpu', batch_size, parallel=False),
        dtype, batch_size)
        for dtype in [np.float32, np.int16]
        for workers_num in [1, 3, 4] for batch_size in [1, 16, 150, 250]]
    for parallel_pipe, _, _, _ in pipes:
        parallel_pipe.start_py_workers()
    for parallel_pipe, pipe, dtype, batch_size in pipes:
        yield check_callback, parallel_pipe, pipe, epoch_size, batch_size, dtype
    # test that another pipline with forking initialization fails as there is CUDA contexts already initialized
    parallel_pipe = create_pipe(callback, 'cpu', 16, py_workers_num=4,
                                py_workers_init='fork', parallel=True)
    yield raises(RuntimeError)(build_and_run_pipeline), parallel_pipe, 1


def test_dtypes():
    yield from check_spawn_with_callback(ExtCallback)


def test_random_data():
    yield from check_spawn_with_callback(ExtCallback, shapes=[(100, 40, 3), (8, 64, 64, 3)], random_data=True)


def test_randomly_shaped_data():
    yield from check_spawn_with_callback(ExtCallback, shapes=[(100, 40, 3), (8, 64, 64, 3)], random_data=True, random_shape=True)


def test_num_outputs():
    yield from check_spawn_with_callback(ExtCallbackMultipleOutputs, ExtCallbackMultipleOutputs, num_outputs=2, dtypes=[np.uint8, np.float])


def test_tensor_cpu():
    yield from check_spawn_with_callback(ExtCallbackTensorCPU)


def test_exception_propagation():
    for raised, expected in [(StopIteration, StopIteration), (CustomException, Exception)]:
        callback = ExtCallback((4, 4), 250, np.int32, exception_class=raised)
        for workers_num in [1, 4]:
            for batch_size in [1, 15, 150]:
                pipe = create_pipe(
                    callback, 'cpu', batch_size, py_workers_num=workers_num,
                    py_workers_init='spawn', parallel=True)
                yield raises(expected)(build_and_run_pipeline), pipe, None, raised, expected


def test_stop_iteration_resume():
    callback = ExtCallback((4, 4), 250, 'int32')
    layout = "XY"
    for workers_num in [1, 4]:
        for batch_size in [1, 15, 150]:
            pipe = create_pipe(callback, 'cpu', batch_size, layout=layout,
                               py_workers_num=workers_num, py_workers_init='spawn', parallel=True)
            yield check_stop_iteration_resume, pipe, batch_size, layout


def test_layout():
    for layout, dims in zip(["X", "XY", "XYZ"], ((4,), (4, 4), (4, 4, 4))):
        callback = ExtCallback(dims, 1024, 'int32')
        for workers_num in [1, 4]:
            for batch_size in [1, 256, 600]:
                pipe = create_pipe(
                    callback, 'cpu', batch_size, layout=layout, py_workers_num=workers_num,
                    py_workers_init='spawn', parallel=True)
                yield check_layout, pipe, layout
