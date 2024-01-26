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
from nvidia.dali import pipeline_def, fn

from nose2.tools import params


wrong_data = [
    np.full((100,), 0, dtype=np.float),  # Wrong type
    np.full((1, 1), 0, dtype=np.uint8),  # Wrong dimensionality
]


def peek_img(input):
    return fn.peek_image_shape(input)


@params(*wrong_data)
def test_error_regular(input):
    @pipeline_def(batch_size=4, device_id=0, num_threads=1)
    def test_pipe():
        return peek_img(input)

    pipe = test_pipe()
    pipe.build()
    print(pipe.run())


@params(*wrong_data)
def test_error_cond(input):
    @pipeline_def(batch_size=4, device_id=0, num_threads=1, enable_conditionals=True)
    def test_pipe():
        if fn.random.coin_flip():
            x = peek_img(input)
        else:
            x = input
        return x

    pipe = test_pipe()
    pipe.build()
    print(pipe.run())
