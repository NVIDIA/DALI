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

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)

set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/file_loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/coco_loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/sequence_loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/utils.cc")

if (BUILD_NVDEC)
  set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/video_loader.cc)
endif()

set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS} PARENT_SCOPE)

# we don't want to test Caffe2 reader if LMDB is not present
if (BUILD_TEST AND BUILD_LMDB)
  # get all the test srcs
  file(GLOB tmp *_test.cc file_loader.cc)
  set(DALI_OPERATOR_TEST_SRCS ${DALI_OPERATOR_TEST_SRCS} ${tmp} PARENT_SCOPE)
endif()
