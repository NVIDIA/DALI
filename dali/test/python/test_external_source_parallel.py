import numpy as np
from nose.tools import raises

import nvidia.dali as dali

from test_utils import compare_pipelines, check_batch


class ExtCallback:

    def __init__(self, dims, epoch_size, dtype, exception_class=StopIteration):
        self.dims = dims
        self.epoch_size = epoch_size
        self.dtype = dtype
        self.exception_class = exception_class
        self.ds = {}

    def __call__(self, sample_info):
        if sample_info.idx_in_epoch >= self.epoch_size:
            raise self.exception_class
        if sample_info.idx_in_epoch not in self.ds:
            self.ds[sample_info.idx_in_epoch] = np.zeros(self.dims, dtype=self.dtype) + sample_info.idx_in_epoch
        return self.ds[sample_info.idx_in_epoch]


class ExtCallbackTensorCPU(ExtCallback):

    def __call__(self, sample_info):
        return dali.tensors.TensorCPU(super().__call__(sample_info))


class ExtCallbackTorch(ExtCallback):

    def __call__(self, sample_info):
        import torch
        return torch.tensor(super().__call__(sample_info))


class ExtCallbackMX(ExtCallback):

    def __call__(self, sample_info):
        from mxnet import ndarray as mxnd
        a = super().__call__(sample_info)
        return mxnd.array(a, dtype=a.dtype)


def create_pipe(callback, device, batch_size, num_outputs=None, layout=None, py_workers_num=None, py_workers_init="fork", parallel=True):
    pipe = dali.pipeline.Pipeline(batch_size, 1, 0, py_workers_num=py_workers_num, py_workers_init=py_workers_init)
    inputs = dali.fn.external_source(callback, num_outputs=num_outputs, device=device, layout=layout, batch=False, parallel=parallel)
    if num_outputs is None:
        pipe.set_outputs(inputs)
    else:
        pipe.set_outputs(*inputs)
    return pipe


def _build_and_run_pipeline(pipe, *args):
    pipe.build()
    while True:
        pipe.run()


def _test_callback(parallel_pipe, pipe, epoch_size, batch_size, dtype=None):  # dtype is ignored but pass it so that is showed by nosetest
    iters_no = epoch_size // batch_size
    parallel_pipe.build()
    pipe.build()
    compare_pipelines(parallel_pipe, pipe, batch_size, iters_no)
    parallel_pipe._py_pool.close()


def test_parallel_fork():
    epoch_size = 250
    callback = ExtCallback((4, 5), epoch_size, 'int32')
    pipes = [(
        create_pipe(callback, 'cpu', batch_size, py_workers_num=workers_num, py_workers_init='fork', parallel=True),
        create_pipe(callback, 'cpu', batch_size, parallel=False),
        dtype, batch_size)
        for (dtype, callback) in [(dtype, ExtCallback((4, 5), epoch_size, dtype)) for dtype in ['float64', 'int16']]
        for workers_num in [1, 3, 4]
        for batch_size in [1, 16, 150, 250]
    ]
    for parallel_pipe, _, _, _ in pipes:
        parallel_pipe.start_py_workers()
    for parallel_pipe, pipe, dtype, batch_size in pipes:
        yield _test_callback, parallel_pipe, pipe, epoch_size, batch_size, dtype
    # test that another pipline with forking initialization fails as there is CUDA contexts already initialized
    parallel_pipe = create_pipe(callback, 'cpu', 16, py_workers_num=4, py_workers_init='fork', parallel=True)
    yield raises(RuntimeError)(_build_and_run_pipeline), parallel_pipe


def _test_spawn_with_callback(
        callback_class, callback_ref_class=ExtCallback, num_outputs=None, layout=None,
        dtypes=['float64', 'int32', 'uint8']):
    epoch_size = 250
    for dtype, args in [(dtype, ((4, 5), epoch_size, dtype)) for dtype in dtypes]:
        callback = callback_class(*args)
        callback_ref = callback_ref_class(*args)
        for workers_num in [1, 4]:
            for batch_size in [1, 16, 150]:
                pipe_parallel = create_pipe(
                    callback, 'cpu', batch_size, py_workers_num=workers_num, py_workers_init='spawn',
                    parallel=True, num_outputs=num_outputs, layout=layout)
                pipe = create_pipe(callback_ref, 'cpu', batch_size, parallel=False, num_outputs=num_outputs, layout=layout)
                yield _test_callback, pipe_parallel, pipe, epoch_size, batch_size, dtype


def test_dtypes():
    yield from _test_spawn_with_callback(ExtCallback)


class ExtCallbackMultipleOutputs(ExtCallback):

    def __call__(self, sample_info):
        import torch
        a = super().__call__(sample_info)
        return a, np.array([sample_info.idx_in_batch])


def test_num_outputs():
    yield from _test_spawn_with_callback(ExtCallbackMultipleOutputs, ExtCallbackMultipleOutputs, num_outputs=2, dtypes=['uint8', 'float64'])


def test_tensor_cpu():
    yield from _test_spawn_with_callback(ExtCallbackTensorCPU)


def test_pytorch():
    yield from _test_spawn_with_callback(ExtCallbackTorch)


def test_mxnet():
    yield from _test_spawn_with_callback(ExtCallbackMX)


class CustomException(Exception):
    pass


def test_exception_propagation():
    for exception in [StopIteration, CustomException]:
        callback = ExtCallback((4, 4), 250, 'int32', exception_class=exception)
        for workers_num in [1, 4]:
            for batch_size in [1, 15, 150]:
                pipe = create_pipe(callback, 'cpu', batch_size, py_workers_num=workers_num, py_workers_init='spawn', parallel=True)
                yield raises(exception)(_build_and_run_pipeline), pipe, exception


def _test_stop_iteration_resume(pipe, batch_size, layout):
    pipe.build()
    outputs_epoch_1, outputs_epoch_2 = [], []
    for output in [outputs_epoch_1, outputs_epoch_2]:
        try:
            while True:
                (r, ) = pipe.run()
                output.append(r)
        except StopIteration:
            pipe.reset()
    assert len(outputs_epoch_1) == len(outputs_epoch_2), ("Epochs must have same number of iterations, "
        "but they have {} {} respectively".format(len(outputs_epoch_1), len(outputs_epoch_2)))
    for out_1, out_2 in zip(outputs_epoch_1, outputs_epoch_2):
        check_batch(out_1, out_2, batch_size, 0, None, expected_layout=layout, compare_layouts=True)


def test_stop_iteration_resume():
    callback = ExtCallback((4, 4), 250, 'int32')
    layout = "XY"
    for workers_num in [1, 4]:
        for batch_size in [1, 15, 150]:
            pipe = create_pipe(callback, 'cpu', batch_size, layout=layout, py_workers_num=workers_num, py_workers_init='spawn', parallel=True)
            yield _test_stop_iteration_resume, pipe, batch_size, layout


def _test_layout(pipe, layout):
    pipe.build()
    while True:
        try:
            (res, ) = pipe.run()
            assert res.layout() == layout
        except StopIteration:
            break


def test_layout():
    for layout, dims in zip(["X", "XY", "XYZ"], ((4,), (4, 4), (4, 4, 4))):
        callback = ExtCallback(dims, 1024, 'int32')
        for workers_num in [1, 4]:
            for batch_size in [1, 256, 600]:
                pipe = create_pipe(callback, 'cpu', batch_size, layout=layout, py_workers_num=workers_num,
                       py_workers_init='spawn', parallel=True)
                yield _test_layout, pipe, layout
