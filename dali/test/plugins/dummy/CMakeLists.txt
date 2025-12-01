# Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

file(GLOB SRCS *.cc *.cu)

set(lib_name "dali_customdummyplugin")
adjust_source_file_language_property("${SRCS}")
add_library(${lib_name} SHARED ${SRCS})
target_link_libraries(${lib_name} PRIVATE ${CUDART_LIB})
target_link_libraries(${lib_name} PUBLIC dali)

set_target_properties(${lib_name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})
