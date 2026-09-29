# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nose2.tools import cartesian_params


@pipeline_def(num_threads=4, device_id=0)
def zero_extent_color_space_conversion_pipe(data, device):
    images = fn.external_source(source=[data], cycle=True, layout="HWC")
    if device == "gpu":
        images = images.gpu()
    return fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY)


@cartesian_params(("cpu", "gpu"), (1, 4))
def test_color_space_conversion_zero_extent_samples(device, batch_size):
    data = [
        np.zeros((0, 8, 3), dtype=np.uint8),
        np.full((2, 3, 3), 255, dtype=np.uint8),
        np.zeros((4, 0, 3), dtype=np.uint8),
        np.zeros((0, 0, 3), dtype=np.uint8),
    ][:batch_size]
    pipe = zero_extent_color_space_conversion_pipe(data, device, batch_size=batch_size)
    (out,) = pipe.run()
    if device == "gpu":
        out = out.as_cpu()
    for i, ref in enumerate(data):
        sample = out.at(i)
        assert sample.shape == ref.shape[:2] + (1,), f"{sample.shape} vs {ref.shape}"
        if ref.size:
            np.testing.assert_array_equal(sample, np.full(ref.shape[:2] + (1,), 255, np.uint8))
