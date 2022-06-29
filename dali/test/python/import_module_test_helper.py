# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Used in test_external_source_parallel_custom_serialization to check if modules
# are properly imported during callback deserialization. Such test only makes sense
# if this module is not automatically imported when worker process starts, so don't
# import this file globally

import numpy as np


def cb(x):
    return np.full((10, 100), x.idx_in_epoch)
