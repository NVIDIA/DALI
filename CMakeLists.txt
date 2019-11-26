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
# allow usage of check_symbol_exists() macro
include(CheckSymbolExists)

if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

project(DALI CXX)
set(DALI_ROOT ${PROJECT_SOURCE_DIR})

# Build options
option(BUILD_TEST "Build googletest test suite" ON)
option(BUILD_BENCHMARK "Build benchmark suite" ON)
option(BUILD_NVTX "Build with NVTX profiling enabled" OFF)
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

# Use -std=c++14 (and not gnu++14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

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

# Include directories
include_directories(BEFORE
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}")
cuda_include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/include")

add_custom_target("check")
add_custom_target("check-gtest")
add_dependencies("check" "check-gtest")

# Project build
add_subdirectory(dali)

# HACK: Add __init__.pys as needed
file(WRITE ${CMAKE_BINARY_DIR}/dali/__init__.py "")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/lint.cmake)
