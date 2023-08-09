import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
from collections.abc import Iterable
from test_utils import compare_pipelines
from nose_utils import assert_raises

# TODO:
# - fn.saturation
# - fn.experimental.equalize
# - fn.reductions
# - fn.cast_like

# Test configurationu
batch_size = 8
test_data_shape = [25, 15, 3]
test_data_layout = "HWC"

def tensor_list_to_array(tensor_list):
    if isinstance(tensor_list, dali.backend_impl.TensorListGPU):
        tensor_list = tensor_list.as_cpu()
    return tensor_list.as_array()

# Check whether a given pipeline is stateless
def check_is_pipeline_stateless(pipeline_factory, iterations=10):
    pipe = pipeline_factory()
    for _ in range(iterations):
        pipe.run()
    # Compare a pipeline that was already used with a fresh one
    compare_pipelines(pipe, pipeline_factory(), batch_size, iterations)

# Provides the same random batch each time
class RandomBatch:
    def __init__(self):
        rng = np.random.default_rng(1234)
        self.batch = [rng.integers(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]

    def __call__(self):
        return self.batch

def check_single_input(op, batch=True, exec_async=True, exec_pipelined=True, **kwargs):
    def pipeline_factory():
        pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=None, exec_async=exec_async,
                      exec_pipelined=exec_pipelined)
        with pipe:
            data = fn.external_source(source=RandomBatch(), layout=test_data_layout, batch=batch)
            processed = op(data, **kwargs)
            if isinstance(processed, Iterable):
                pipe.set_outputs(*processed)
            else:
                pipe.set_outputs(processed)
        pipe.build()
        return pipe

    check_is_pipeline_stateless(pipeline_factory)

def test_stateful():
    assert_raises(AssertionError, check_single_input, fn.random.coin_flip)

def test_rotate_stateless():
    check_single_input(fn.rotate, angle=40)

def test_resize_stateless():
    check_single_input(fn.resize, resize_x=50, resize_y=50) 

def test_flip_stateless():
    check_single_input(fn.flip)

def test_crop_mirror_normalize_stateless():
    check_single_input(fn.crop_mirror_normalize)

def test_warp_affine_stateless():
    check_single_input(fn.warp_affine, matrix=(0.1, 0.9, 10, 0.8, -0.2, -20))
  
def test_saturation_stateless():
    check_single_input(fn.saturation)

def test_reductions_min_stateless():
    check_single_input(fn.reductions.min)

def test_reductions_max_stateless():
    check_single_input(fn.reductions.max)

def test_reductions_sum_stateless():
    check_single_input(fn.reductions.sum)

def test_equalize_stateless():
    check_single_input(fn.experimental.equalize)
 
def test_cast_like_cpu():
    def pipeline_factory():
        pipe = Pipeline(batch_size=batch_size, num_threads=3, device_id=None)
        out = fn.cast_like(np.array([1, 2, 3], dtype=np.int32), np.array([1.0], dtype=np.float32))
        pipe.set_outputs(out)
        pipe.build()
        return pipe
    
    check_is_pipeline_stateless(pipeline_factory)