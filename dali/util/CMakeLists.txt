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

set(DALI_INST_HDRS ${DALI_INST_HDRS}
  "${CMAKE_CURRENT_SOURCE_DIR}/crop_window.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/image.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/mmaped_file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/std_file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/odirect_file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ocv.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/thread_safe_queue.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/numpy.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/user_stream.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/uri.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/s3_file.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/s3_filesystem.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/s3_client_manager.h")

set(DALI_SRCS ${DALI_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/image.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/mmaped_file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/std_file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/odirect_file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/ocv.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/user_stream.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/numpy.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/uri.cc")

if (BUILD_CUFILE)
  set(DALI_INST_HDRS ${DALI_INST_HDRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/cufile.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cufile_helper.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/std_cufile.h")

  set(DALI_SRCS ${DALI_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/cufile.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/std_cufile.cc")
endif()

if (BUILD_CFITSIO)
  set(DALI_SRCS ${DALI_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/fits.cc")

  set(DALI_TEST_SRCS ${DALI_TEST_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/fits_test.cc")
endif()


if (BUILD_AWSSDK)
  set(DALI_SRCS ${DALI_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/s3_file.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/s3_filesystem.cc")
endif()

set(DALI_TEST_SRCS ${DALI_TEST_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/random_crop_generator_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/numpy_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/uri_test.cc")

# transform a list of paths into a list of include directives
DETERMINE_GCC_SYSTEM_INCLUDE_DIRS("c++" "${CMAKE_CXX_COMPILER}" "${CMAKE_CXX_FLAGS}" INFERED_COMPILER_INCLUDE)
set(DEFAULT_COMPILER_INCLUDE)
foreach(incl_dir ${INFERED_COMPILER_INCLUDE})
  set(DEFAULT_COMPILER_INCLUDE "${DEFAULT_COMPILER_INCLUDE} -I${incl_dir}")
endforeach(incl_dir)
separate_arguments(DEFAULT_COMPILER_INCLUDE UNIX_COMMAND  "${DEFAULT_COMPILER_INCLUDE}")

# CUDA 13 include add a cccl subdir, that is detected by CMake and CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
# becomes a list instead of a single dir. In our code we relay on it being a single entry, so this
# code makes sure we extract the first path from the list and use it further (assuming it is the main dir)
# So we check if it is a list (; inside) and act accordingly
string(FIND "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" ";" SEMICOLON_POS)
if(NOT "${SEMICOLON_POS}" EQUAL -1)
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE)
    foreach(incl_dir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
       set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE} -I${incl_dir}")
    endforeach(incl_dir)
else()
    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()
separate_arguments(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE UNIX_COMMAND  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE}")

if(BUILD_NVML)
  set(DALI_INST_HDRS ${DALI_INST_HDRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/nvml.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/nvml_wrap.h")

  if (NOT LINK_DRIVER)
    set(NVML_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvml_gen.cc")
    add_custom_command(
        OUTPUT ${NVML_GENERATED_STUB}
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py --unique_prefix=Nvml --
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/nvml.json" ${NVML_GENERATED_STUB}
                    "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/nvml.h"
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE}
                    # for some reason QNX fails with 'too many errors emitted' is this is not set
                    "-ferror-limit=0"
                    # let clang know which architecutre we compile for
                    "--target=${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
                    ${DEFAULT_COMPILER_INCLUDE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py
                "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/nvml.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/nvml.json"
        COMMENT "Running nvml.h stub generator"
        VERBATIM)

    set_source_files_properties(${NVML_GENERATED_STUB} PROPERTIES GENERATED TRUE)
    add_library(dynlink_nvml STATIC nvml_wrap.cc nvml.cc ${NVML_GENERATED_STUB})
  else()
    add_library(dynlink_nvml STATIC nvml_wrap.cc nvml.cc)
  endif()
endif()

set(DALI_INST_HDRS ${DALI_INST_HDRS} PARENT_SCOPE)
set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)
set(DALI_TEST_SRCS ${DALI_TEST_SRCS} PARENT_SCOPE)

list(FILTER DALI_SRCS EXCLUDE REGEX ".*nvml.*\.cc")
set(DALI_SRCS ${DALI_SRCS} PARENT_SCOPE)
