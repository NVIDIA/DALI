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

if (BUILD_NVDEC)
  add_subdirectory(nvdecode)
  list(APPEND DALI_INST_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/frames_decoder_gpu.h")
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/frames_decoder_gpu.cc")
  list(APPEND DALI_INST_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/video_loader_decoder_gpu.h")
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/video_loader_decoder_gpu.cc")
  if (BUILD_TEST)
    list(APPEND DALI_OPERATOR_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/frames_decoder_test.cc")
  endif()
endif(BUILD_NVDEC)

list(APPEND DALI_INST_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/frames_decoder.h")
list(APPEND DALI_INST_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/video_loader_decoder_cpu.h")
set(DALI_INST_HDRS ${DALI_INST_HDRS} PARENT_SCOPE)

list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/frames_decoder.cc")
list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/video_loader_decoder_cpu.cc")
set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  list(APPEND DALI_OPERATOR_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/video_test_base.cc")
  set(DALI_OPERATOR_TEST_SRCS ${DALI_OPERATOR_TEST_SRCS} PARENT_SCOPE)
endif()
