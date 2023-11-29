# Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    from nvidia.dali.backend_impl.tfrecord import *  # noqa: F403 F401
except ImportError:
    raise RuntimeError(
        "DALI was not compiled with TFRecord support."
        " Use BUILD_PROTO3=ON CMake option to enable TFRecord support"
    )
