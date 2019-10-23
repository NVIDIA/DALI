# Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

set(DALI_INST_HDRS ${DALI_INST_HDRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/crop_window.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/custream.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/half.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/image.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/local_file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/npp.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ocv.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/thread_safe_queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/type_conversion.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/user_stream.h")

set(DALI_SRCS ${DALI_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/custream.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/image.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/local_file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/npp.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/ocv.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/type_conversion.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/user_stream.cc")

set(DALI_TEST_SRCS ${DALI_TEST_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator_test.cc")


if(BUILD_NVML)
  set(DALI_INST_HDRS ${DALI_INST_HDRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/nvml.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/nvml_wrap.h")

  set(DALI_SRCS ${DALI_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/nvml_wrap.cc")
endif()


set(DALI_INST_HDRS ${DALI_INST_HDRS} PARENT_SCOPE)
set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)
set(DALI_TEST_SRCS ${DALI_TEST_SRCS} PARENT_SCOPE)
