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

import os

import numpy as np

from nvidia.dali import pipeline_def, fn, types
from test_utils import get_dali_extra_path, np_type_to_dali, check_batch
from nose2.tools import params

from filter_test_utils import filter_img_baseline

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')


def filter_enumerated(shape):
    n = np.prod(shape)
    weight_sum = n * (n + 1) / 2
    return np.float32(1 / weight_sum * np.arange(1, n + 1).reshape(shape))


def create_filter_anchor_source(shapes):
    rng = np.random.default_rng(42)

    def source(sample_info):
        shape_idx = sample_info.idx_in_batch % len(shapes)
        shape = shapes[shape_idx]
        anchor = np.int32(rng.uniform(0, shape))
        for dim in range(len(anchor)):
            if rng.uniform(0, 4) >= 3:
                anchor[dim] = -1
        anchor = np.array(anchor, dtype=np.int32)
        return filter_enumerated(shape), anchor

    return source


def create_sample_source(shapes, dtype):
    rng = np.random.default_rng(42)
    if not np.issubdtype(dtype, np.integer):
        low, high = 0, 1
    else:
        type_info = np.iinfo(dtype)
        low, high = type_info.min, type_info.max

    def source(sample_info):
        shape_idx = sample_info.idx_in_batch % len(shapes)
        shape = shapes[shape_idx]
        return dtype(rng.uniform(low, high, shape))

    return source


@pipeline_def
def images_pipeline(shapes, border, in_dtype, mode):
    images, _ = fn.readers.file(name="Reader", file_root=images_dir, prefetch_queue_depth=2,
                                random_shuffle=True, seed=42)
    images = fn.experimental.decoders.image(images, device="mixed", output_type=types.RGB,
                                            dtype=np_type_to_dali(in_dtype))
    filters, anchors = fn.external_source(source=create_filter_anchor_source(shapes), batch=False,
                                          num_outputs=2)
    fill_val_limit = 1 if not np.issubdtype(in_dtype, np.integer) else np.iinfo(in_dtype).max
    fill_values = fn.random.uniform(range=[0, fill_val_limit], dtype=np_type_to_dali(in_dtype))
    if border == "constant":
        convolved = fn.experimental.filter(images, filters, fill_values, anchor=anchors,
                                           border=border, mode=mode)
    else:
        convolved = fn.experimental.filter(images, filters, anchor=anchors, border=border,
                                           mode=mode)
    return convolved, images, filters, anchors, fill_values


@pipeline_def
def sample_pipeline(sample_shapes, sample_layout, filter_shapes, border, in_dtype, mode):
    samples = fn.external_source(source=create_sample_source(sample_shapes, in_dtype), batch=False,
                                 layout=sample_layout)
    filters, anchors = fn.external_source(source=create_filter_anchor_source(filter_shapes),
                                          batch=False, num_outputs=2)
    fill_val_limit = 1 if not np.issubdtype(in_dtype, np.integer) else np.iinfo(in_dtype).max
    rand_fill_dtype = in_dtype if in_dtype != np.float16 else np.float32
    fill_values = fn.random.uniform(range=[0, fill_val_limit],
                                    dtype=np_type_to_dali(rand_fill_dtype))
    fill_values = fn.cast_like(fill_values, samples)
    if border == "constant":
        convolved = fn.experimental.filter(samples.gpu(), filters, fill_values, anchor=anchors,
                                           border=border, mode=mode)
    else:
        convolved = fn.experimental.filter(samples.gpu(), filters, anchor=anchors, border=border,
                                           mode=mode)
    return convolved, samples, filters, anchors, fill_values


@params(
    (np.uint8, 16, "101", "same"),
    (np.uint8, 11, "clamp", "same"),
    (np.uint8, 4, "constant", "same"),
    (np.int8, 7, "1001", "same"),
    (np.int16, 8, "wrap", "same"),
    (np.int16, 1, "constant", "same"),
    (np.float32, 11, "constant", "same"),
    (np.float32, 13, "101", "same"),
    (np.uint8, 4, "constant", "valid"),
    (np.float32, 7, "101", "valid"),
)
def test_image_pipeline(dtype, batch_size, border, mode):
    shapes = [(3, 3), (8, 8), (31, 1), (1, 31), (1, 1), (51, 3), (3, 51), (2, 40), (2, 40), (2, 2),
              (27, 27)]
    num_iters = 2

    pipe = images_pipeline(batch_size=batch_size, num_threads=4, device_id=0, border=border,
                           in_dtype=dtype, shapes=shapes, mode=mode)
    pipe.build()
    atol = 1 if np.issubdtype(dtype, np.integer) else 1e-5
    for _ in range(num_iters):
        filtered_imgs, imgs, kernels, anchors, fill_values = pipe.run()
        filtered_imgs = [np.array(img) for img in filtered_imgs.as_cpu()]
        imgs = [np.array(img) for img in imgs.as_cpu()]
        kernels = [np.array(kernel) for kernel in kernels]
        anchors = [np.array(anchor) for anchor in anchors]
        fill_values = [np.array(fv) for fv in fill_values]
        assert len(filtered_imgs) == len(imgs) == len(kernels) == len(anchors) == len(fill_values)
        baseline = [
            filter_img_baseline(img, kernel, anchor, border, fill_value, mode)
            for img, kernel, anchor, fill_value in zip(imgs, kernels, anchors, fill_values)
        ]
        check_batch(filtered_imgs, baseline, max_allowed_error=atol)


@params(
    (np.float16, [(501, 127, 3), (600, 600, 1), (128, 256, 5),
                  (200, 500, 2)], "HWC", [(3, 3), (8, 5), (10, 4), (70, 1),
                                          (1, 70)], 8, "101", "same"), )
def test_samples(dtype, sample_shapes, sample_layout, filter_shapes, batch_size, border, mode):
    num_iters = 2

    pipe = sample_pipeline(batch_size=batch_size, num_threads=4, device_id=0,
                           sample_shapes=sample_shapes, sample_layout=sample_layout,
                           filter_shapes=filter_shapes, border=border, in_dtype=dtype, mode=mode)
    pipe.build()
    if dtype == np.float32:
        atol = 1e-5
    elif dtype == np.float16:
        atol = 1e-2
    else:
        assert np.issubdtype(dtype, np.integer)
        atol = 1
    for _ in range(num_iters):
        filtered_imgs, imgs, kernels, anchors, fill_values = pipe.run()
        filtered_imgs = [np.array(img) for img in filtered_imgs.as_cpu()]
        imgs = [np.array(img) for img in imgs]
        kernels = [np.array(kernel) for kernel in kernels]
        anchors = [np.array(anchor) for anchor in anchors]
        fill_values = [np.array(fv) for fv in fill_values]
        assert len(filtered_imgs) == len(imgs) == len(kernels) == len(anchors) == len(fill_values)
        baseline = [
            filter_img_baseline(img, kernel, anchor, border, fill_value, mode)
            for img, kernel, anchor, fill_value in zip(imgs, kernels, anchors, fill_values)
        ]
        check_batch(filtered_imgs, baseline, max_allowed_error=atol)
