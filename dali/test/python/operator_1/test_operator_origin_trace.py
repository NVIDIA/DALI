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
from test_utils import load_test_operator_plugin


def test_plugin_ops():
    load_test_operator_plugin()

    @pipeline_def(batch_size=2, num_threads=1, device_id=0, enable_conditionals=True)
    def pipe():
        for _ in range(1):
            if 0:
                return fn.origin_trace_dump()
            else:
                return fn.origin_trace_dump()

    p = pipe()
    p.build()
    (out,) = p.run()
    arr = np.array(out[0])
    print("\n\n")
    print(arr.view(f"S{arr.shape[0]}")[0].decode("utf-8"))
