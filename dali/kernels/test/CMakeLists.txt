# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

add_subdirectory(resampling_test)
add_subdirectory(warp_test)

# Get all the source files and dump test files, but not the headers
collect_sources(DALI_KERNEL_SRCS PARENT_SCOPE)
collect_test_sources(DALI_KERNEL_TEST_SRCS PARENT_SCOPE)
