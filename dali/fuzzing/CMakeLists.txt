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
  )

  # Image decoder fuzzing target
  add_executable(dali_decoder_fuzzing "${DALI_FUZZING_SRCS}" "${CMAKE_CURRENT_SOURCE_DIR}/image_decoder_target.cc")
  target_link_libraries(dali_decoder_fuzzing PRIVATE dali dali_operators ${DALI_LIBS})
  target_link_libraries(dali_decoder_fuzzing PRIVATE "-pie")
  set_target_properties(dali_decoder_fuzzing PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_decoder_fuzzing PROPERTIES OUTPUT_NAME "dali_image_decoder_fuzzing_target.bin")
  set_target_properties(dali_decoder_fuzzing PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${DALI_WHEEL_DIR}/test")

  # ResNet pipeline fuzzing target
  add_executable(dali_rn50_fuzzing "${DALI_FUZZING_SRCS}" "${CMAKE_CURRENT_SOURCE_DIR}/rn50_target.cc")
  target_link_libraries(dali_rn50_fuzzing PRIVATE dali dali_operators ${DALI_LIBS})
  target_link_libraries(dali_rn50_fuzzing PRIVATE "-pie")
  set_target_properties(dali_rn50_fuzzing PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_rn50_fuzzing PROPERTIES OUTPUT_NAME "dali_rn50_fuzzing_target.bin")
  set_target_properties(dali_rn50_fuzzing PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${DALI_WHEEL_DIR}/test")

endif()
