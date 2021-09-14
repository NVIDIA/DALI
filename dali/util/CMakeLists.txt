# Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  "${CMAKE_CURRENT_SOURCE_DIR}/file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/image.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/mmaped_file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/std_file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/npp.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ocv.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/thread_safe_queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/user_stream.h")

set(DALI_SRCS ${DALI_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/image.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/mmaped_file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/std_file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/npp.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/ocv.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/user_stream.cc")

if (BUILD_CUFILE)
  set(DALI_INST_HDRS ${DALI_INST_HDRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/cufile.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/std_cufile.h")

  set(DALI_SRCS ${DALI_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/cufile.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/std_cufile.cc")
endif()

set(DALI_TEST_SRCS ${DALI_TEST_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator_test.cc")


if(BUILD_NVML)
  set(DALI_INST_HDRS ${DALI_INST_HDRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/nvml.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/nvml_wrap.h")

  if (NOT LINK_DRIVER)
    set(NVML_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvml_gen.cc")
    add_custom_command(
        OUTPUT ${NVML_GENERATED_STUB}
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py --unique_prefix=Nvml --
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvml.json" ${NVML_GENERATED_STUB}
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/nvml.h" "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    # for some reason QNX fails with 'too many errors emitted' is this is not set
                    "-ferror-limit=0"
                    ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/stub_codegen.py
                "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/nvml.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/stub_generator/nvml.json"
        COMMENT "Running nvml.h stub generator"
        VERBATIM)

    set_source_files_properties(${NVML_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_nvml OBJECT nvml_wrap.cc nvml.cc ${NVML_GENERATED_STUB})
  else()
    add_library(dynlink_nvml OBJECT nvml_wrap.cc nvml.cc)
  endif()
endif()

set(DALI_INST_HDRS ${DALI_INST_HDRS} PARENT_SCOPE)
set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)
set(DALI_TEST_SRCS ${DALI_TEST_SRCS} PARENT_SCOPE)

list(FILTER DALI_SRCS EXCLUDE REGEX ".*nvml.*\.cc")
