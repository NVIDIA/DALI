# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import fn, pipeline_def
from test_utils import RandomlyShapedDataIterator, to_array

import numpy as np

batch_size = 10


@pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
def copy_pipe(shape, layout, dev, dtype):
    min_shape = [s // 2 if s > 1 else 1 for s in shape]
    min_shape = tuple(min_shape)
    input = fn.external_source(
        source=RandomlyShapedDataIterator(
            batch_size, min_shape=min_shape, max_shape=shape, dtype=dtype
        ),
        layout=layout,
    )
    if dev == "gpu":
        input = input.gpu()
    output = fn.copy(input)
    return input, output


def check_copy(shape, layout, dev, dtype=np.uint8):
    pipe = copy_pipe(shape, layout, dev, dtype)
    for i in range(10):
        input, output = pipe.run()
        for i in range(batch_size):
            assert output[i].layout() == input[i].layout()
            expected = to_array(input[i])
            obtained = to_array(output[i])
            np.testing.assert_array_equal(expected, obtained)


def test_copy():
    for shape, layout in [([4, 2, 3], "HWC"), ([6, 1], "FX"), ([8, 10, 10, 3], "FHWC")]:
        for device in ["cpu", "gpu"]:
            for dtype in [np.uint8, np.float16, np.int32]:
                yield check_copy, shape, layout, device, dtype
