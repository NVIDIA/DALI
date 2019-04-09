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

add_subdirectory(cache)

# Get all the source files and dump test files
if (BUILD_NVJPEG)
  if (NVJPEG_DECOUPLED_API)
    list(APPEND DALI_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_allocator.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_decoupled_api.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_crop.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_slice.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_random_crop.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_cpu.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_cpu_crop.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_cpu_slice.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_cpu_random_crop.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_gpu.cu")
  else()
    list(APPEND DALI_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder.cu")
  endif()
  if (BUILD_TEST)
    if (NVJPEG_DECOUPLED_API)
      list(APPEND DALI_TEST_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_decoupled_api_test.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_crop_test.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_slice_test.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_random_crop_test.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_split_test.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_split_crop_test.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_split_slice_test.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_split_random_crop_test.cc")
    else()
      list(APPEND DALI_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/nvjpeg_decoder_test.cc")
    endif()
  endif()
endif()

list(APPEND DALI_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/host_decoder.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/host_decoder_crop.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/host_decoder_slice.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/host_decoder_random_crop.cc"
  )

set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)

if (BUILD_TEST)
  list(APPEND DALI_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/host_decoder_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_decoder_crop_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_decoder_slice_test.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_decoder_random_crop_test.cc"
    )
  set(DALI_TEST_SRCS ${DALI_TEST_SRCS} PARENT_SCOPE)
endif()
