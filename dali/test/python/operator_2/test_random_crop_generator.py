# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from nvidia.dali import fn, pipeline_def
from test_utils import get_dali_extra_path

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")


def test_random_crop_generator():
    np.random.seed(12345)
    batch_size = 8
    seed = 1234

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
    def pipe():
        encoded, _ = fn.readers.file(file_root=images_dir)
        images = fn.decoders.image(encoded)
        shapes = fn.peek_image_shape(encoded)
        images_crop1 = fn.decoders.image_random_crop(encoded, seed=seed)
        crop_anchor, crop_shape = fn.random_crop_generator(shapes, seed=seed)
        images_crop2 = fn.slice(images, start=crop_anchor, shape=crop_shape, axes=[0, 1])
        return images_crop1, images_crop2

    p = pipe()
    out0, out1 = p.run()
    for i in range(batch_size):
        np.testing.assert_array_equal(out0[i], out1[i])


def test_random_crop_generator_subcrop():
    np.random.seed(12345)
    batch_size = 8
    seed0 = 1234
    seed1 = 2345

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
    def pipe():
        encoded, _ = fn.readers.file(file_root=images_dir)

        # First: generate subrandom crop and crop only once
        shapes = fn.peek_image_shape(encoded)

        crop_anchor0_A, crop_shape0_A = fn.random_crop_generator(shapes, seed=seed0)
        rel_crop_anchor1_A, crop_shape1_A = fn.random_crop_generator(crop_shape0_A, seed=seed1)
        crop_anchor1_A = crop_anchor0_A + rel_crop_anchor1_A
        images_crop1_A = fn.decoders.image_slice(
            encoded, start=crop_anchor1_A, shape=crop_shape1_A, axes=[0, 1]
        )

        # Second: Crop twice
        images_crop0_B = fn.decoders.image_random_crop(encoded, seed=seed0)
        crop_anchor1_B, crop_shape1_B = fn.random_crop_generator(images_crop0_B.shape(), seed=seed1)
        images_crop1_B = fn.slice(
            images_crop0_B, start=crop_anchor1_B, shape=crop_shape1_B, axes=[0, 1]
        )

        return images_crop1_A, images_crop1_B

    p = pipe()
    out0, out1 = p.run()
    for i in range(batch_size):
        np.testing.assert_array_equal(out0[i], out1[i])
