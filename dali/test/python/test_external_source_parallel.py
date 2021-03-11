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
from nose.tools import raises, with_setup

from test_pool_utils import *
from test_utils import compare_pipelines
from test_external_source_parallel_utils import *


# Test that we can launch several CPU-only pipelines by fork as we don't touch CUDA context.
@with_setup(setup_function, teardown_function)
def test_parallel_fork_cpu_only():
    global pipe_processes
    pipeline_pairs = 4
    batch_size = 10
    iters = 40
    callback = ExtCallback((4, 5), iters * batch_size, np.int32)
    parallel_pipes = [(create_pipe(callback, 'cpu', batch_size, py_num_workers=4,
                                   py_start_method='fork', parallel=True, device_id=None),
                       create_pipe(callback, 'cpu', batch_size, py_num_workers=4,
                                   py_start_method='fork', parallel=True, device_id=None))
                      for i in range(pipeline_pairs)]
    for pipe0, pipe1 in parallel_pipes:
        pipe0.build()
        pipe1.build()
        capture_processes(pipe0._py_pool)
        capture_processes(pipe1._py_pool)
        compare_pipelines(pipe0, pipe1, batch_size, iters)

@with_setup(setup_function, teardown_function)
def test_parallel_no_workers():
    batch_size = 10
    iters = 4
    callback = ExtCallback((4, 5), iters * batch_size, np.int32)
    parallel_pipe = create_pipe(callback, 'cpu', batch_size, py_num_workers=0,
                                py_start_method='spawn', parallel=True, device_id=None)
    parallel_pipe.build()
    capture_processes(parallel_pipe._py_pool)
    assert parallel_pipe._py_pool is None
    assert parallel_pipe._py_pool_started == False

@with_setup(setup_function, teardown_function)
def test_parallel_fork():
    epoch_size = 250
    callback = ExtCallback((4, 5), epoch_size, np.int32)
    pipes = [(
        create_pipe(
            callback, 'cpu', batch_size, py_num_workers=num_workers, py_start_method='fork',
            parallel=True),
        create_pipe(callback, 'cpu', batch_size, parallel=False),
        dtype, batch_size)
        for dtype in [np.float32, np.int16]
        for num_workers in [1, 3, 4] for batch_size in [1, 16, 150, 250]]
    for parallel_pipe, _, _, _ in pipes:
        parallel_pipe.start_py_workers()
    for parallel_pipe, pipe, dtype, batch_size in pipes:
        yield check_callback, parallel_pipe, pipe, epoch_size, batch_size, dtype
    # test that another pipline with forking initialization fails as there is CUDA contexts already initialized
    parallel_pipe = create_pipe(callback, 'cpu', 16, py_num_workers=4,
                                py_start_method='fork', parallel=True)
    yield raises(RuntimeError)(build_and_run_pipeline), parallel_pipe, 1

@with_setup(setup_function, teardown_function)
def test_dtypes():
    yield from check_spawn_with_callback(ExtCallback)

@with_setup(setup_function, teardown_function)
def test_random_data():
    yield from check_spawn_with_callback(ExtCallback, shapes=[(100, 40, 3), (8, 64, 64, 3)], random_data=True)

@with_setup(setup_function, teardown_function)
def test_randomly_shaped_data():
    yield from check_spawn_with_callback(ExtCallback, shapes=[(100, 40, 3), (8, 64, 64, 3)], random_data=True, random_shape=True)

@with_setup(setup_function, teardown_function)
def test_num_outputs():
    yield from check_spawn_with_callback(ExtCallbackMultipleOutputs, ExtCallbackMultipleOutputs, num_outputs=2, dtypes=[np.uint8, np.float])

@with_setup(setup_function, teardown_function)
def test_tensor_cpu():
    yield from check_spawn_with_callback(ExtCallbackTensorCPU)

@with_setup(setup_function, teardown_function)
def test_exception_propagation():
    for raised, expected in [(StopIteration, StopIteration), (CustomException, Exception)]:
        callback = ExtCallback((4, 4), 250, np.int32, exception_class=raised)
        for num_workers in [1, 4]:
            for batch_size in [1, 15, 150]:
                pipe = create_pipe(
                    callback, 'cpu', batch_size, py_num_workers=num_workers,
                    py_start_method='spawn', parallel=True)
                yield raises(expected)(build_and_run_pipeline), pipe, None, raised, expected

@with_setup(setup_function, teardown_function)
def test_stop_iteration_resume():
    callback = ExtCallback((4, 4), 250, 'int32')
    layout = "XY"
    for num_workers in [1, 4]:
        for batch_size in [1, 15, 150]:
            pipe = create_pipe(callback, 'cpu', batch_size, layout=layout,
                               py_num_workers=num_workers, py_start_method='spawn', parallel=True)
            yield check_stop_iteration_resume, pipe, batch_size, layout

@with_setup(setup_function, teardown_function)
def test_layout():
    for layout, dims in zip(["X", "XY", "XYZ"], ((4,), (4, 4), (4, 4, 4))):
        callback = ExtCallback(dims, 1024, 'int32')
        for num_workers in [1, 4]:
            for batch_size in [1, 256, 600]:
                pipe = create_pipe(
                    callback, 'cpu', batch_size, layout=layout, py_num_workers=num_workers,
                    py_start_method='spawn', parallel=True)
                yield check_layout, pipe, layout

class ext_cb():
    def __init__(self, name):
        pass
    def __call__(self, sinfo):
        return np.full([100,100,100], sinfo.idx_in_epoch, dtype=np.int32)

def test_vs_non_parallel():
    bs = 50
    pipe = dali.Pipeline(batch_size=bs, device_id=None, num_threads=5, py_num_workers=14, py_start_method='spawn')
    with pipe:
        ext_seq = dali.fn.external_source(ext_cb("cb 1"), batch=False, parallel=False)
        ext_par = dali.fn.external_source(ext_cb("cb 2"), batch=False, parallel=True)
        pipe.set_outputs(ext_seq, ext_par)
    pipe.build()
    for i in range(10):
        seq, par = pipe.run()
        for j in range(bs):
            s = seq.at(j)
            p = par.at(j)
            assert np.array_equal(s, p)
