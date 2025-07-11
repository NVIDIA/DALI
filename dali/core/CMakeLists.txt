# Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(dali_core CUDA CXX C)

add_subdirectory(mm)
add_subdirectory(os)
add_subdirectory(exec)

# Get all the source files
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_CORE_SRCS)
collect_test_sources(DALI_CORE_TEST_SRCS)

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

if (NOT LINK_DRIVER)
  set(CUDA_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_cuda_gen.cc")
  add_custom_command(
      OUTPUT ${CUDA_GENERATED_STUB}
      COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py --unique_prefix=Cuda --
                  "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/cuda.json" ${CUDA_GENERATED_STUB}
                  "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/cuda.h"
                  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE}
                  # for some reason QNX fails with 'too many errors emitted' is this is not set
                  "-ferror-limit=0"
                  # let clang know which architecutre we compile for
                  "--target=${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
                  ${DEFAULT_COMPILER_INCLUDE}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py
              "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/cuda.h"
              "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/cuda.json"
      COMMENT "Running cuda.h stub generator"
      VERBATIM)

  set_source_files_properties(${CUDA_GENERATED_STUB} PROPERTIES GENERATED TRUE)
  add_library(dynlink_cuda STATIC dynlink_cuda.cc ${CUDA_GENERATED_STUB})
else()
  add_library(dynlink_cuda STATIC dynlink_cuda.cc)
endif()

list(FILTER DALI_CORE_SRCS EXCLUDE REGEX ".*dynlink_cuda.cc")

if (BUILD_CUFILE)
  set(CUFILE_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_cufile_gen.cc")
  add_custom_command(
      OUTPUT ${CUFILE_GENERATED_STUB}
      COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py --unique_prefix=Cufile --
                  "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/cufile.json" ${CUFILE_GENERATED_STUB}
                  "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/cufile.h"
                  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE}
                  # for some reason QNX fails with 'too many errors emitted' is this is not set
                  "-ferror-limit=0"
                  # let clang know which architecutre we compile for
                  "--target=${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
                  ${DEFAULT_COMPILER_INCLUDE}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py
              "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/cufile.h"
              "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/cufile.json"
      COMMENT "Running cufile.h stub generator"
      VERBATIM)

  set_source_files_properties(${CUFILE_GENERATED_STUB} PROPERTIES GENERATED TRUE)
  add_library(dynlink_cufile STATIC dynlink_cufile.cc ${CUFILE_GENERATED_STUB})
endif()

list(FILTER DALI_CORE_SRCS EXCLUDE REGEX ".*dynlink_cufile.cc")

if (BUILD_NVCOMP)
  set(NVCOMP_GENERATED_STUB "${CMAKE_CURRENT_BINARY_DIR}/dynlink_nvcomp_gen.cc")
  add_custom_command(
      OUTPUT ${NVCOMP_GENERATED_STUB}
      COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py --unique_prefix=nvComp --
                  "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/nvcomp.json" ${NVCOMP_GENERATED_STUB}
                  "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/nvcomp/lz4.h"
                  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES_DIRECTIVE}
                  # for some reason QNX fails with 'too many errors emitted' is this is not set
                  "-ferror-limit=0"
                  # let clang know which architecutre we compile for
                  "--target=${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
                  ${DEFAULT_COMPILER_INCLUDE}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/stub_codegen.py
              "${CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY}/nvcomp/lz4.h"
              "${CMAKE_CURRENT_SOURCE_DIR}/../../internal_tools/stub_generator/nvcomp.json"
      COMMENT "Running nvcomp/lz4.hstub generator"
      VERBATIM)

  set_source_files_properties(${NVCOMP_GENERATED_STUB} PROPERTIES GENERATED TRUE)
  add_library(dynlink_nvcomp STATIC dynlink_nvcomp.cc ${NVCOMP_GENERATED_STUB})
endif()

list(FILTER DALI_CORE_SRCS EXCLUDE REGEX ".*dynlink_nvcomp.cc")

adjust_source_file_language_property("${DALI_CORE_SRCS}")
add_library(dali_core ${LIBTYPE} ${DALI_CORE_SRCS})
target_include_directories(dali_core SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(dali_core PRIVATE dynlink_cuda ${CUDART_LIB})
if (BUILD_CUFILE)
    target_link_libraries(dali_core PRIVATE dynlink_cufile)
  endif()
target_link_libraries(dali_core PUBLIC ${DALI_SYSTEM_LIBS})
target_link_libraries(dali_core PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
set_target_properties(dali_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DALI_LIBRARY_OUTPUT_DIR}")

set(lib_exports "libdali_core.map")
configure_file("${DALI_ROOT}/cmake/${lib_exports}.in" "${CMAKE_BINARY_DIR}/${lib_exports}")
target_link_libraries(dali_core PRIVATE -Wl,--version-script=${CMAKE_BINARY_DIR}/${lib_exports})
if (BUILD_NVML)
  target_link_libraries(dali_core PRIVATE dynlink_nvml)
endif(BUILD_NVML)

if (BUILD_TEST)
  adjust_source_file_language_property("${DALI_CORE_TEST_SRCS}")
  add_executable(dali_core_test "${DALI_CORE_TEST_SRCS}")
  target_link_libraries(dali_core_test PUBLIC dali_core)
  target_link_libraries(dali_core_test PRIVATE gtest dynlink_cuda ${DALI_LIBS})
  if (BUILD_CUFILE)
    target_link_libraries(dali_core_test PRIVATE dynlink_cufile)
  endif()
  target_link_libraries(dali_core_test PRIVATE "-Wl,--exclude-libs,${exclude_libs}")
  target_link_libraries(dali_core_test PRIVATE "-pie")
  set_target_properties(dali_core_test PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dali_core_test PROPERTIES OUTPUT_NAME "dali_core_test.bin")

  set_target_properties(dali_core_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIR})

  add_check_gtest_target("check-core-gtest" dali_core_test ${TEST_BINARY_DIR})
endif()
