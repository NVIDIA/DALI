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

if (BUILD_LIBTAR)
  add_subdirectory(webdataset)
endif()

# Get all the source files and dump test files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)

set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/filesystem.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/discover_files.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/file_label_loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/coco_loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/sequence_loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/numpy_loader.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/utils.cc")


if (BUILD_CUFILE)
  set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/numpy_loader_gpu.cc")
endif()

if (BUILD_CFITSIO)
  set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/fits_loader.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/fits_loader_gpu.cc")
endif()

if (BUILD_LIBSND)
  set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/nemo_asr_loader.cc")
endif()

set(DALI_OPERATOR_TEST_SRCS ${DALI_OPERATOR_TEST_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/loader_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/sequence_loader_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/filesystem_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/discover_files_test.cc")

if (BUILD_LIBSND)
  set(DALI_OPERATOR_TEST_SRCS ${DALI_OPERATOR_TEST_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/nemo_asr_loader_test.cc")
endif()

if (BUILD_LIBTAR)
  set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS}
     "${CMAKE_CURRENT_SOURCE_DIR}/webdataset_loader.cc")
endif()

if (BUILD_AWSSDK)
  set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/discover_files_s3.cc")
endif()

set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS} PARENT_SCOPE)
set(DALI_OPERATOR_TEST_SRCS ${DALI_OPERATOR_TEST_SRCS} PARENT_SCOPE)
