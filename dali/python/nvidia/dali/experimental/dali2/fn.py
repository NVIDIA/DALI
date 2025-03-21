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

import nvidia.dali.backend as _b
import nvidia.dali.ops as _ops


def initialize():
    _all_ops = _ops._registry._all_registered_ops()
    for op_name in _all_ops:
        # ExternalSource and PythonFunction are not needed in dali2 API
        if op_name.endswith("ExternalSource") or op_name.endswith("PythonFunction"):
            continue

        schema = _b.TryGetSchema(_all_ops)
        if schema is None:
            print(f"Warning: no schema found for {op_name}")
            continue
