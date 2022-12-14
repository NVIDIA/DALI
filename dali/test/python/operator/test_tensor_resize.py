# Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.types as types
from nvidia.dali import pipeline_def, fn
from nose2.tools import params
import numpy as np
from test_utils import as_array


@params('cpu', 'gpu')
def test_resize_upsample_scales_nearest(device):
    data = np.array(
        [[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

    expected = np.array(
        [[[
            [1., 1., 1., 2., 2., 2.],
            [1., 1., 1., 2., 2., 2.],
            [3., 3., 3., 4., 4., 4.],
            [3., 3., 3., 4., 4., 4.],
        ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

    @pipeline_def(batch_size=1, num_threads=3, device_id=0)
    def pipe():
        input_data = types.Constant(data, device=device)
        return fn.experimental.tensor_resize(input_data, scales=scales, interp_type=types.INTERP_NN)
    p = pipe()
    p.build()
    out = p.run()
    np.testing.assert_allclose(expected, as_array(out[0][0]))
