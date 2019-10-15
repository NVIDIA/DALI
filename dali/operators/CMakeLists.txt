# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

add_subdirectory(color)
add_subdirectory(color_space)
add_subdirectory(crop)
add_subdirectory(decoder)
add_subdirectory(detection)
add_subdirectory(displacement)
add_subdirectory(expressions)
add_subdirectory(fused)
add_subdirectory(geometric)
if (BUILD_NVOF)
  add_subdirectory(optical_flow)
endif()
add_subdirectory(paste)
add_subdirectory(reader)
add_subdirectory(resize)
add_subdirectory(sequence)
add_subdirectory(support)
add_subdirectory(transpose)
add_subdirectory(util)
add_subdirectory(python_function/util)

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_SRCS PARENT_SCOPE)
collect_test_sources(DALI_TEST_SRCS PARENT_SCOPE)

set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS} PARENT_SCOPE)
