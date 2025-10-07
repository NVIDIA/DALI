# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.experimental.dali2 as D
import numpy as np
import gc
from nose_utils import SkipTest, attr
from nose2.tools import params
import nvidia.dali.backend as _b
from nose_utils import assert_raises
from test_tensor import asnumpy

def test_batch_subscript_per_sample():
    b = D.as_batch(
        [
            D.tensor([[1, 2, 3], [4, 5, 6]], dtype=D.int32),
            D.tensor([[7, 8, 9], [10, 11, 12]], dtype=D.int32),
        ]
    )
    # unzipped indices (1, 1), (0, 2)
    i = D.as_batch([
        1, 0
    ])
    j = D.as_batch([
        1, 2
    ])
    b11 = b.slice[i, j]
    assert isinstance(b11, D.Batch)
    assert asnumpy(b11.tensors[0]) == 5
    assert asnumpy(b11.tensors[1]) == 9
