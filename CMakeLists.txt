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

cmake_minimum_required(VERSION 3.12)
# allow usage of check_symbol_exists() macro
include(CheckCXXSymbolExists)

if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

project(DALI CUDA CXX C)
set(DALI_ROOT ${PROJECT_SOURCE_DIR})
set(CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# Build options
option(DYNAMIC_CUDA "Link to CUDA dynamically" OFF)
option(BUILD_TEST "Build googletest test suite" ON)
option(BUILD_BENCHMARK "Build benchmark suite" ON)
# if BUILD_NVTX is empty remove it and let is be default
if ("${BUILD_NVTX}" STREQUAL "")
  unset(BUILD_NVTX CACHE)
endif()
# starting from CUDA 10.0 we enabling NVTX by default as it doesn't have any linkage dependency
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "10.0")
  option(BUILD_NVTX "Build with NVTX profiling enabled" ON)
else()
  option(BUILD_NVTX "Build with NVTX profiling enabled" OFF)
endif()
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_LMDB "Build LMDB readers" OFF)
option(BUILD_JPEG_TURBO "Build with libjpeg-turbo support" ON)
option(BUILD_LIBTIFF "Build with libtiff support" ON)
option(BUILD_NVJPEG "Build with nvJPEG support" ON)
option(BUILD_NVOF "Build with NVIDIA OPTICAL FLOW SDK support" ON)
option(BUILD_NVDEC "Build with NVIDIA NVDEC support" ON)
option(BUILD_LIBSND "Build with suport for libsnd library" ON)
option(BUILD_NVML "Build with NVIDIA Management Library (NVML) support" ON)
option(BUILD_FFTS "Build with ffts support" ON)
option(VERBOSE_LOGS "Adds verbose loging to DALI" OFF)

option(WERROR "Threat all warnings as errors" OFF)

# FFmpeg is required when we are using NVDEC for video reader
set(BUILD_FFMPEG ${BUILD_NVDEC})

include(cmake/Utils.cmake)
include(cmake/CUDA_utils.cmake)

propagate_option(BUILD_NVTX)
propagate_option(BUILD_PYTHON)
propagate_option(BUILD_LMDB)
propagate_option(BUILD_JPEG_TURBO)
propagate_option(BUILD_LIBTIFF)
propagate_option(BUILD_NVJPEG)
propagate_option(BUILD_NVOF)
propagate_option(BUILD_NVDEC)
propagate_option(BUILD_LIBSND)
propagate_option(BUILD_NVML)
propagate_option(BUILD_FFTS)
propagate_option(BUILD_FFMPEG)

get_dali_version(${PROJECT_SOURCE_DIR}/VERSION DALI_VERSION)

get_dali_extra_version(${PROJECT_SOURCE_DIR}/DALI_EXTRA_VERSION DALI_EXTRA_VERSION)
configure_file("${PROJECT_SOURCE_DIR}/dali/test/dali_test_info.h.in"
               "${PROJECT_BINARY_DIR}/dali/test/dali_test_info.h")
set(DALI_INST_HDRS ${DALI_INST_HDRS} "${PROJECT_BINARY_DIR}/dali/test/dali_test_info.h")

# Default to release build
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type from [Debug, DevDebug, Release, RelWithDebInfo]. For perf testing, build Release" FORCE)
endif()
message(STATUS "Build configuration: " ${CMAKE_BUILD_TYPE})

# Cmake path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Dependencies
if(${ARCH} MATCHES "aarch64-linux")
  message("Target set to aarch64-linux")
  include(cmake/Dependencies.aarch64-linux.cmake)
elseif(${ARCH} MATCHES "aarch64-qnx")
  message("Target set to aarch64-qnx")
  include(cmake/Dependencies.aarch64-qnx.cmake)
else()
  include(cmake/Dependencies.cmake)
endif()

# add more flags after they are populated by find_package from Dependencies.cmake

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -DDALI_DEBUG=1")
# Generate only line info for device as -G disables all optimizations and causes unit tests to fail
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -lineinfo -DDALI_DEBUG=1")

# DevDebug flags - Use the "-G" for proper debug info for device code
set(CMAKE_CXX_FLAGS_DEVDEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
string(REPLACE "-lineinfo" "-G" CMAKE_CUDA_FLAGS_DEVDEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")

# Release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DDALI_DEBUG=0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -DDALI_DEBUG=0")

# Release with debug info flags
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -g -lineinfo")

# Use -std=c++14 (and not gnu++14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-variable -Wno-unused-function -fno-strict-aliasing -fPIC -fvisibility=hidden")

if (WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if (BUILD_WITH_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register -Wsign-compare")

  # CUDA does not support current clang as host compiler, we need use gcc
  # CMAKE_CUDA_HOST_COMPILER variable operates on paths
  set(CUDA_UNSUPPORTED_COMPILER 0)
  if ("${CMAKE_CUDA_HOST_COMPILER}" MATCHES "clang")
      set(CUDA_UNSUPPORTED_COMPILER 1)
  elseif (CMAKE_CUDA_HOST_COMPILER STREQUAL "")
      set(CUDA_UNSUPPORTED_COMPILER 1)
  endif()

  if(${CUDA_UNSUPPORTED_COMPILER})
    message(STATUS "CMAKE_CUDA_HOST_COMPILER is set to ${CMAKE_C_COMPILER} - setting CMAKE_CUDA_HOST_COMPILER to gcc")
    execute_process(COMMAND which gcc OUTPUT_VARIABLE PATH_TO_GCC OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT PATH_TO_GCC)
      message(FATAL_ERROR "gcc was not found in PATH")
    else()
      set(CMAKE_CUDA_HOST_COMPILER ${PATH_TO_GCC})
    endif()
  endif()
endif()

# Discover what architectures does nvcc support
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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_gencode_flags} --compiler-options -fvisibility=hidden")

# Include directories
include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}")

add_custom_target("check")
add_custom_target("check-gtest")
add_dependencies("check" "check-gtest")

# Project build
add_subdirectory(dali)

# HACK: Add __init__.pys as needed
file(WRITE ${CMAKE_BINARY_DIR}/dali/__init__.py "")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/lint.cmake)
