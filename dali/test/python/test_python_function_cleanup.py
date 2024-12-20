# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import numpy as np


def f(x):
    return x


@pipeline_def(batch_size=256, num_threads=1, seed=0)
def pipeline():
    x = types.Constant(np.full((1,), 0))
    x = fn.python_function(x, function=f, num_outputs=1)
    y = types.Constant(np.full((1024, 720), 0))

    return x + y


p = pipeline(device_id=None)
p.run()
