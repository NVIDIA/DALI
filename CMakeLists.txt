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

cmake_minimum_required(VERSION 3.11)

project(DALI CXX)

# Build options
option(BUILD_TEST "Build googletest test suite" ON)
option(BUILD_BENCHMARK "Build benchmark suite" ON)
option(BUILD_NVTX "Build with NVTX profiling enabled" OFF)
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_LMDB "Build LMDB readers" OFF)
option(BUILD_TENSORFLOW "Build TensorFlow plugin" OFF)
option(BUILD_JPEG_TURBO "Build with libjpeg-turbo" ON)
option(BUILD_NVJPEG "Build with nvJPEG" ON)

option(WERROR "Threat all warnings as errors" OFF)

include(cmake/Utils.cmake)

get_dali_version(${CMAKE_SOURCE_DIR}/VERSION DALI_VERSION)

# Default to release build
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type from [Debug, Release]. For perf testing, build Release" FORCE)
endif()

# Cmake path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Dependencies
include(cmake/Dependencies.cmake)

# add more flags after they are populated by find_package from Dependencies.cmake

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -DDALI_DEBUG=1")
# Generate only line info for device as -G disables all optimizations and causes unit tests to fail
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -g -lineinfo -DDALI_DEBUG=1")

# DevDebug flags - Use the "-G" for proper debug info for device code
set(CMAKE_CXX_FLAGS_DEVDEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE "-lineinfo" "-G" CUDA_NVCC_FLAGS_DEVDEBUG "${CUDA_NVCC_FLAGS_DEBUG}")

# Release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DDALI_DEBUG=0")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -DDALI_DEBUG=0")

# Release with debug info flags
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g3")
set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "${CUDA_NVCC_FLAGS_RELWITHDEBINFO} -g -lineinfo")


# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unused-variable -Wno-unused-function -fno-strict-aliasing -fPIC -fvisibility=hidden")

if (WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register -Wsign-compare")

  # CUDA does not support current clang as host compiler, we need use gcc
  # CUDA_HOST_COMPILER variable operates on paths
  if (${CUDA_HOST_COMPILER} MATCHES "clang")
    message(STATUS "CUDA_HOST_COMPILER is set to ${CMAKE_C_COMPILER} - setting CUDA_HOST_COMPILER to gcc")
    execute_process(COMMAND which gcc OUTPUT_VARIABLE PATH_TO_GCC OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT PATH_TO_GCC)
      message(FATAL_ERROR "gcc was not found in PATH")
    else()
      set(CUDA_HOST_COMPILER ${PATH_TO_GCC})
    endif()
  endif()
endif()

# Discover what architectures does nvcc support
include(cmake/CUDA_utils.cmake)
CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
list(SORT CUDA_TARGET_ARCHS_SORTED)
CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
if (NOT CUDA_targeted_archs)
  message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) is supported by nvcc, use one or more of: ${CUDA_supported_archs}")
endif()

CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
message(STATUS "Generated gencode flags: ${CUDA_gencode_flags}")

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

# Project dir
include_directories(BEFORE ${PROJECT_SOURCE_DIR})
include_directories(BEFORE ${PROJECT_BINARY_DIR})
cuda_include_directories(${PROJECT_SOURCE_DIR})

# Project build
add_subdirectory(dali)

# HACK: Add __init__.pys as needed
file(WRITE ${CMAKE_BINARY_DIR}/dali/__init__.py "")

add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/lint.cmake)
