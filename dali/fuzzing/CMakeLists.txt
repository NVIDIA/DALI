# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


collect_headers(DALI_INST_HDRS PARENT_SCOPE)
set(DALI_FUZZING_SRCS
  "${PROJECT_SOURCE_DIR}/dali/test/dali_test_config.cc"
)

function(DALI_ADD_FUZZING_TARGET TARGET_NAME BINARY_NAME TARGET_SRC)
  add_executable(${TARGET_NAME} "${DALI_FUZZING_SRCS}" "${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_SRC}")
  target_link_libraries(${TARGET_NAME} PRIVATE dali dali_operators ${DALI_LIBS})
  target_link_libraries(${TARGET_NAME} PRIVATE "-pie")
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${BINARY_NAME}")
  set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${DALI_WHEEL_DIR}/test")
endfunction()

# Image decoder fuzzing target
dali_add_fuzzing_target(dali_decoder_fuzzing dali_image_decoder_fuzzing_target.bin image_decoder_target.cc)

# ResNet pipeline fuzzing target
dali_add_fuzzing_target(dali_rn50_fuzzing dali_rn50_fuzzing_target.bin rn50_target.cc)
