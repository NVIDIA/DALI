# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali as dali
from nose_utils import with_setup
from test_pool_utils import capture_processes, teardown_function, setup_function
from test_utils import (
    compare_pipelines,
    check_batch,
    RandomDataIterator,
    RandomlyShapedDataIterator,
)


class ExtCallback:
    """Callable to generate specified data samples"""

    def __init__(
        self,
        dims,
        epoch_size,
        dtype,
        exception_class=StopIteration,
        random_data=False,
        random_shape=False,
    ):
        self.dims = dims
        self.epoch_size = epoch_size
        self.dtype = dtype
        self.exception_class = exception_class
        self.ds = {}
        self.random_data = random_data
        self.random_shape = random_shape
        self.data_iterator = None
        self.iterator_data_samples = []
        if random_data and not random_shape:
            self.data_iterator = iter(RandomDataIterator(1, shape=dims, dtype=dtype))
        if random_data and random_shape:
            self.data_iterator = iter(RandomlyShapedDataIterator(1, max_shape=dims, dtype=dtype))
        if not random_data and random_shape:
            raise ValueError("If random_shape is required the random_data is required to be True.")

    def __call__(self, sample_info):
        if sample_info.idx_in_epoch >= self.epoch_size:
            raise self.exception_class
        if self.data_iterator:
            while len(self.iterator_data_samples) <= sample_info.idx_in_epoch:
                batch = self.data_iterator.next()
                self.iterator_data_samples.append(batch[0])
            return self.iterator_data_samples[sample_info.idx_in_epoch]
        if sample_info.idx_in_epoch not in self.ds:
            self.ds[sample_info.idx_in_epoch] = np.full(
                self.dims, sample_info.idx_in_epoch, dtype=self.dtype
            )
        return self.ds[sample_info.idx_in_epoch]


class ExtCallbackTensorCPU(ExtCallback):
    def __call__(self, sample_info):
        return dali.tensors.TensorCPU(super().__call__(sample_info))


def create_pipe(
    callback,
    device,
    batch_size,
    num_outputs=None,
    layout=None,
    py_num_workers=None,
    py_start_method="fork",
    parallel=True,
    device_id=0,
    batch=False,
    num_threads=1,
    cycle=None,
    batch_info=None,
    prefetch_queue_depth=2,
    reader_queue_depth=None,
):
    pipe = dali.pipeline.Pipeline(
        batch_size,
        num_threads,
        device_id,
        py_num_workers=py_num_workers,
        py_start_method=py_start_method,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    with pipe:
        inputs = dali.fn.external_source(
            callback,
            num_outputs=num_outputs,
            device=device,
            layout=layout,
            batch=batch,
            parallel=parallel,
            cycle=cycle,
            batch_info=batch_info,
            prefetch_queue_depth=reader_queue_depth,
        )
        if num_outputs is None:
            pipe.set_outputs(inputs)
        else:
            pipe.set_outputs(*inputs)
    return pipe


def build_and_run_pipeline(pipe, iters=None, *args):
    pipe.build()
    capture_processes(pipe._py_pool)
    if iters is None:
        while True:
            pipe.run()
    else:
        for _ in range(iters):
            pipe.run()


# dtype is ignored but pass it so that is showed by nosetest
def check_callback(parallel_pipe, pipe, epoch_size, batch_size, dtype=None):
    iters_no = epoch_size // batch_size
    parallel_pipe.build()
    pipe.build()
    capture_processes(parallel_pipe._py_pool)
    compare_pipelines(parallel_pipe, pipe, batch_size, iters_no)


@with_setup(setup_function, teardown_function)
def _check_spawn_with_callback(
    callback, callback_ref, batch_size, num_outputs, layout, workers_num, epoch_size, dtype
):
    pipe_parallel = create_pipe(
        callback,
        "cpu",
        batch_size,
        py_num_workers=workers_num,
        py_start_method="spawn",
        parallel=True,
        num_outputs=num_outputs,
        layout=layout,
    )
    pipe = create_pipe(
        callback_ref, "cpu", batch_size, parallel=False, num_outputs=num_outputs, layout=layout
    )
    check_callback(pipe_parallel, pipe, epoch_size, batch_size, dtype)


def check_spawn_with_callback(
    callback_class,
    callback_ref_class=ExtCallback,
    num_outputs=None,
    layout=None,
    dtypes=[np.float32, np.int32, np.uint8],
    shapes=[(4, 5)],
    random_data=False,
    random_shape=False,
):
    epoch_size = 250
    for shape in shapes:
        for dtype in dtypes:
            callback = callback_class(
                shape, epoch_size, dtype, random_data=random_data, random_shape=random_shape
            )
            callback_ref = callback_ref_class(
                shape, epoch_size, dtype, random_data=random_data, random_shape=random_shape
            )
            for workers_num in [1, 4]:
                for batch_size in [1, 16, 150]:
                    yield (
                        _check_spawn_with_callback,
                        callback,
                        callback_ref,
                        batch_size,
                        num_outputs,
                        layout,
                        workers_num,
                        epoch_size,
                        dtype,
                    )


class ExtCallbackMultipleOutputs(ExtCallback):
    def __call__(self, sample_info):
        a = super().__call__(sample_info)
        return a, np.array([sample_info.idx_in_batch])


class CustomException(Exception):
    pass


def check_stop_iteration_resume(pipe, batch_size, layout):
    pipe.build()
    capture_processes(pipe._py_pool)
    outputs_epoch_1, outputs_epoch_2 = [], []
    for output in [outputs_epoch_1, outputs_epoch_2]:
        try:
            while True:
                (r,) = pipe.run()
                r = [np.copy(r.at(i)) for i in range(len(r))]
                output.append(r)
        except StopIteration:
            pipe.reset()
    assert len(outputs_epoch_1) == len(
        outputs_epoch_2
    ), "Epochs must have same number of iterations, " "but they have {} {} respectively".format(
        len(outputs_epoch_1), len(outputs_epoch_2)
    )
    for out_1, out_2 in zip(outputs_epoch_1, outputs_epoch_2):
        check_batch(out_1, out_2, batch_size, 0, None, expected_layout=layout, compare_layouts=True)


def check_layout(pipe, layout):
    pipe.build()
    capture_processes(pipe._py_pool)
    while True:
        try:
            (res,) = pipe.run()
            assert res.layout() == layout
        except StopIteration:
            break
