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

if (BUILD_FUZZING)
  collect_headers(DALI_INST_HDRS PARENT_SCOPE)
  set(DALI_FUZZING_SRCS
    "${PROJECT_SOURCE_DIR}/dali/test/dali_test_config.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/simple_fuzzing.cc"
  )

  add_executable(dali_fuzzing "${DALI_FUZZING_SRCS}")

  target_link_libraries(dali_fuzzing PRIVATE dali dali_operators ${DALI_LIBS})
  target_link_libraries(dali_fuzzing PRIVATE "-pie")
  set_target_properties(dali_fuzzing PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_fuzzing PROPERTIES OUTPUT_NAME "dali_fuzzing.bin")

  set_target_properties(dali_fuzzing PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${DALI_WHEEL_DIR}/test")

endif()
