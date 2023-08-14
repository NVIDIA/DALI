# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from collections.abc import Iterable
from test_utils import compare_pipelines
from nose_utils import assert_raises

# Test configuration
batch_size = 8
test_data_shape = [25, 15, 3]
test_data_layout = "HWC"


def tensor_list_to_array(tensor_list):
    if isinstance(tensor_list, dali.backend_impl.TensorListGPU):
        tensor_list = tensor_list.as_cpu()
    return tensor_list.as_array()


# Check whether a given pipeline is stateless
def check_is_pipeline_stateless(pipeline_factory, iterations=10):
    args = {
        'batch_size': batch_size,
        'num_threads': 4,
        'device_id': None,
        'exec_async': True,
        'exec_pipelined': True,
    }

    pipe = pipeline_factory(**args)
    pipe.build()
    for _ in range(iterations):
        pipe.run()

    # Compare a pipeline that was already used with a fresh one
    compare_pipelines(pipe, pipeline_factory(**args), batch_size, iterations)


# Provides the same random batch each time
class RandomBatch:
    def __init__(self):
        rng = np.random.default_rng(1234)
        self.batch = [rng.integers(255, size=test_data_shape, dtype=np.uint8)
                      for _ in range(batch_size)]

    def __call__(self):
        return self.batch


def check_single_input(op, **kwargs):
    @pipeline_def
    def pipeline_factory():
        data = fn.external_source(source=RandomBatch(), layout=test_data_layout, batch=True)
        return op(data, **kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def check_no_input(op, **kwargs):
    @pipeline_def
    def pipeline_factory():
        return op(**kwargs)

    check_is_pipeline_stateless(pipeline_factory)


def test_stateful():
    assert_raises(
        AssertionError, check_single_input, fn.random.coin_flip,
        glob='Mean error: *, Min error: *, Max error: *'
             'Total error count: *, Tensor size: *'
             'Index in batch: 0')


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


def test_transforms_crop_stateless():
    check_no_input(fn.transforms.crop)


def test_transforms_rotation_stateless():
    check_no_input(fn.transforms.rotation, angle=35)


def test_transforms_shear_stateless():
    check_no_input(fn.transforms.shear, shear=(2, 2))


def test_transforms_scale_stateless():
    check_no_input(fn.transforms.scale, scale=(3, 2))


def test_transforms_translation_stateless():
    check_no_input(fn.transforms.translation, offset=(4, 3))


def test_cast_like_stateless():
    @pipeline_def
    def pipeline_factory():
        return fn.cast_like(np.array([1, 2, 3], dtype=np.int32), np.array([1.0], dtype=np.float32))

    check_is_pipeline_stateless(pipeline_factory)
