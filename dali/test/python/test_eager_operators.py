# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from functools import reduce

import nvidia.dali.eager.experimental as eager
import nvidia.dali.fn as fn
import nvidia.dali.tensors as tensors
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nose_utils import raises
from test_utils import check_batch

rng = np.random.default_rng()

batch_size = 2
sample_shape = [20, 20, 3]
data = [[rng.integers(0, 255, size=sample_shape, dtype=np.uint8)
         for _ in range(batch_size)] for _ in range(10)]


def get_data(i):
    return data[i]


@pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
def single_op_pipe(op, kwargs):
    data = fn.external_source(source=get_data, layout="HWC")
    out = op(data, **kwargs)
    return out


def reduce_getattr(x, y):
    return getattr(x, y)


def compare_eager_with_pipeline(path, batch_size=batch_size, N_iterations=5, fn_op=None, eager_op=None,
                                **kwargs):
    import_path = path.split('.')
    if fn_op is None:
        fn_op = reduce(reduce_getattr, [fn] + import_path)
    if eager_op is None:
        eager_op = reduce(reduce_getattr, [eager] + import_path)

    pipe = single_op_pipe(fn_op, kwargs)
    pipe.build()

    for i in range(N_iterations):
        input_tl = tensors.TensorListCPU(np.array(get_data(i)), layout="HWC")
        out1, = pipe.run()
        out2 = eager_op(input_tl, **kwargs)

        out1_data = out1.as_cpu() if isinstance(out1, tensors.TensorListGPU) else out1
        out2_data = out2.as_cpu() if isinstance(out2, tensors.TensorListGPU) else out2

        check_batch(out1_data, out2_data, batch_size)


def test_rotate_cpu():
    compare_eager_with_pipeline('rotate', angle=25)


def test_brightness_contrast_cpu():
    compare_eager_with_pipeline('brightness_contrast')


def test_hue_cpu():
    compare_eager_with_pipeline('hue')


def test_brightness_cpu():
    compare_eager_with_pipeline('brightness')


def test_contrast_cpu():
    compare_eager_with_pipeline('contrast')


def test_hsv_cpu():
    compare_eager_with_pipeline('hsv')


def test_color_twist_cpu():
    compare_eager_with_pipeline('color_twist')


def test_saturation_cpu():
    compare_eager_with_pipeline('saturation')


def test_shapes_cpu():
    compare_eager_with_pipeline('shapes')


def test_crop_cpu():
    compare_eager_with_pipeline('crop', crop=(5, 5))


def test_color_space_coversion_cpu():
    compare_eager_with_pipeline('color_space_conversion',
                                image_type=types.BGR, output_type=types.RGB)


def test_cast_cpu():
    compare_eager_with_pipeline('cast', dtype=types.INT32)


def test_resize_cpu():
    compare_eager_with_pipeline('resize', resize_x=50, resize_y=50)


def test_gaussian_blur_cpu():
    compare_eager_with_pipeline('gaussian_blur', window_size=5)


def test_laplacian_cpu():
    compare_eager_with_pipeline('laplacian', window_size=5)


def test_crop_mirror_normalize_cpu():
    compare_eager_with_pipeline('crop_mirror_normalize')


def test_flip_cpu():
    compare_eager_with_pipeline('flip', horizontal=True)


def test_jpeg_compression_distortion_cpu():
    compare_eager_with_pipeline('jpeg_compression_distortion', quality=10)


def test_reshape_cpu():
    new_shape = sample_shape.copy()
    new_shape[0] //= 2
    new_shape[1] *= 2
    compare_eager_with_pipeline("reshape", shape=new_shape)


def test_reinterpret_cpu():
    compare_eager_with_pipeline("reinterpret", rel_shape=[0.5, 1, -1])


def test_water_cpu():
    compare_eager_with_pipeline("water")


def test_sphere_cpu():
    compare_eager_with_pipeline("sphere")


def test_erase_cpu():
    compare_eager_with_pipeline("erase", anchor=[0.3], axis_names="H",
                                normalized_anchor=True, shape=[0.1], normalized_shape=True)


def test_expand_dims_cpu():
    compare_eager_with_pipeline("expand_dims", axes=1, new_axis_names="Z")


def test_coord_transform_cpu():
    M = [0, 0, 1,
         0, 1, 0,
         1, 0, 0]
    compare_eager_with_pipeline("coord_transform", M=M, dtype=types.UINT8)


def test_grid_mask_cpu():
    compare_eager_with_pipeline("grid_mask", tile=51, ratio=0.38158387, angle=2.6810782)

import torch

def test_multi_paste_cpu():
    compare_eager_with_pipeline("multi_paste", in_ids=torch.tensor([0, 1], dtype=torch.int32, device='cuda'), output_size=sample_shape)


@raises(RuntimeError, glob=f"Argument '*' is not supported by eager operator 'crop'.")
def _test_disqualified_argument(key):
    tl = tensors.TensorListCPU(np.zeros((8, 256, 256, 3)))
    eager.crop(tl, crop=[64, 64], **{key: 0})


def test_disqualified_arguments():
    for arg in ['bytes_per_sample_hint', 'preserve', 'seed']:
        yield _test_disqualified_argument, arg
