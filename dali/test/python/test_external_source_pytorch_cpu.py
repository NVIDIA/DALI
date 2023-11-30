# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# it is enough to just import all functions from test_internals_operator_external_source
# nose will query for the methods available and will run them
# the test_internals_operator_external_source is 99% the same for cupy and numpy tests
# so it is better to store everything in one file and just call `use_cupy`
# to switch between the default numpy and cupy

from test_external_source_impl import *  # noqa: F403

use_torch(False)  # noqa: F405
