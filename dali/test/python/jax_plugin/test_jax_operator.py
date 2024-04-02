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

import os

import numpy as np

import jax
from nvidia.dali import pipeline_def, fn, types
import nvidia.dali.plugin.jax as dax

from nose_utils import raises
from nose2.tools import params
from test_utils import get_dali_extra_path, check_batch
from jax_op_utils import jax_color_twist


test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, "db", "single", "jpeg")


@params(
    ("cpu", np.uint8),
    ("gpu", np.uint8),
    ("cpu", np.float32),
    ("gpu", np.float32),
)
def test_jit_vs_dali_op(device, dtype):
    assert device in ("cpu", "gpu")
    num_iters = 3

    @dax.fn.jax_function(output_layouts="HWC")
    @jax.jit
    @jax.vmap
    def dax_color_twist(img, bcs, hue):
        return jax_color_twist(img, bcs, hue)

    @pipeline_def(batch_size=8, device_id=0, num_threads=4, seed=42)
    def pipeline():
        img, _ = fn.readers.file(name="Reader", file_root=images_dir, random_shuffle=True, seed=42)
        img = fn.decoders.image(img, device="cpu" if device == "cpu" else "mixed")
        img = fn.resize(img, size=(224, 224))
        if dtype == np.float32:
            img = fn.cast(img, dtype=types.DALIDataType.FLOAT)
            img = img / 255
        bcs = fn.random.uniform(range=[0.1, 1.9], shape=3)
        hue = fn.random.uniform(range=[0, 180])
        dali_img = fn.color_twist(img, brightness=bcs[0], contrast=bcs[1], saturation=bcs[2], hue=hue)
        if device == "gpu":
            bcs, hue = bcs.gpu(), hue.gpu()
        dax_img = dax_color_twist(img, bcs, hue)
        return dali_img, dax_img

    p = pipeline()
    p.build()
    for _ in range(num_iters):
        dali_img, dax_img = p.run()
        check_batch(dali_img, dax_img, compare_layouts=True, max_allowed_error=1 if dtype != np.float32 else 1e-6)
