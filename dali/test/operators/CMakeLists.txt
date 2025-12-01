# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (BUILD_TEST)
  # Get all the test srcs, make it a part of gtest binary
  file(GLOB tmp *.cc *.cu *.h)
  adjust_source_file_language_property("${tmp}")
  set(DALI_TEST_SRCS ${DALI_TEST_SRCS} ${tmp} PARENT_SCOPE)

  # Additionally build the operators as loadable library, so it can be imported as plugin in Python
  set(lib_name "testoperatorplugin")
  add_library(${lib_name} SHARED ${tmp})
  target_link_libraries(${lib_name} PRIVATE ${CUDART_LIB})
  target_link_libraries(${lib_name} PUBLIC dali)
  if (BUILD_NVML)
    target_link_libraries(${lib_name} PRIVATE dynlink_nvml)
  endif(BUILD_NVML)

  set_target_properties(${lib_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})
endif()
