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

cmake_minimum_required(VERSION 3.25.2)

# allow usage of check_symbol_exists() macro
include(CheckCXXSymbolExists)
include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

include(cmake/Utils.cmake)

if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()
if (POLICY CMP0175)
  cmake_policy(SET CMP0175 NEW)
endif()

project(DALI CUDA CXX C)

set(DALI_ROOT ${PROJECT_SOURCE_DIR})
set(CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# more recent CMake puts both cuda/targets/x86_64-linux/include and cuda/targets/x86_64-linux/include/cccl into the path
# while we usually need only the former
string(FIND "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" ";" SEMICOLON_POS)
if(NOT "${SEMICOLON_POS}" EQUAL -1)
    list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY)
else()
    set(CUDA_TOOLKIT_INCLUDE_MAJOR_DIRECTORY ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

parse_cuda_version(${CUDA_VERSION} CUDA_VERSION_MAJOR CUDA_VERSION_MINOR CUDA_VERSION_PATCH CUDA_VERSION_SHORT CUDA_VERSION_SHORT_DIGIT_ONLY)
# CUDA_utils.cmake uses CUDA_VERSION_MAJOR value to determine the content of CUDA_known_archs and CUDA_TARGET_ARCHS
include(cmake/CUDA_utils.cmake)

check_cxx_compiler_flag(-fopenmp      CXX_HAVE_OMP)
check_cxx_compiler_flag(-fopenmp-simd CXX_HAVE_OMP_SIMD)

# Build options
option(BUILD_DALI_NODEPS "Disable components that require extra external libraries to be present in the system. Effectively, it builds only the DALI core and kernel libraries")
option(LINK_DRIVER "Links directly with libcuda.so instead of dlopen it at runtime" OFF)
option(WITH_DYNAMIC_CUDA_TOOLKIT "Links CUDA toolkit libraries dynamically (NPP, nvJPEG, cuFFT)" OFF)

option(WITH_DYNAMIC_NVIMGCODEC "Links nvimgcodec library dynamically during runtime" ON)

option(BUILD_WITH_ASAN "Build with ASAN" OFF)
option(BUILD_WITH_LSAN "Build with LSAN" OFF)
option(BUILD_WITH_UBSAN "Build with UBSAN" OFF)


# Tests use OpenCV...
cmake_dependent_option(BUILD_TEST "Build googletest test suite" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_BENCHMARK "Build benchmark suite" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
option(BUILD_FUZZING "Build fuzzing suite" OFF)

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

# Third party library dependencies
cmake_dependent_option(BUILD_PYTHON "Build Python bindings" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_SHM_WRAPPER "Build shared memory bindings, needs BUILD_PYTHON=ON" ON
                      "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_LMDB "Build LMDB readers" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_JPEG_TURBO "Build with libjpeg-turbo support" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_LIBTIFF "Build with libtiff support" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_LIBSND "Build with support for libsnd library" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_LIBTAR "Build with support for libtar library" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
option(BUILD_FFTS "Build with ffts support" ON)  # Built from thirdparty sources
cmake_dependent_option(BUILD_CFITSIO "Build with cfitsio support"  ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_NVIMAGECODEC "Build with support for nvimagecodec library" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_AWSSDK "Build with support for AWS SKD library" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
set(NVIMGCODEC_DEFAULT_INSTALL_PATH "/opt/nvidia/nvimgcodec_cuda${CUDA_VERSION_MAJOR}" CACHE STRING
    "Path of the nvimagecodec installation")

cmake_dependent_option(BUILD_CVCUDA "Build with CV-CUDA" ON
                       "NOT STATIC_LIBS" OFF)  # Built from thirdparty sources; doesn't support static libs build

set(KERNEL_SRCS_PATTERN "" CACHE STRING
    "Glob expression to build a subset of files for the kernel lib")
set(KERNEL_SRCS_PATTERN_EXCLUDE "" CACHE STRING
    "Glob expression to exclude from KERNEL_SRCS_PATTERN")
set(KERNEL_TEST_SRCS_PATTERN "" CACHE STRING
    "Glob expression to build a subset of files for the kernel test binary")
set(KERNEL_TEST_SRCS_PATTERN_EXCLUDE "" CACHE STRING
    "Glob expression to exclude from KERNEL_SRCS_PATTERN")

set(OPERATOR_SRCS_PATTERN "" CACHE STRING
    "Glob expression to build a subset of files for the operators lib")
set(OPERATOR_SRCS_PATTERN_EXCLUDE "" CACHE STRING
    "Glob expression to exclude from OPERATOR_SRCS_PATTERN")
set(OPERATOR_TEST_SRCS_PATTERN "" CACHE STRING
    "Glob expression to build a subset of files for the operators test binary")
set(OPERATOR_TEST_SRCS_PATTERN_EXCLUDE "" CACHE STRING
    "Glob expression to exclude from OPERATOR_TEST_SRCS_PATTERN")

# NVIDIA libraries
cmake_dependent_option(BUILD_NVDEC "Build with NVIDIA NVDEC support" ON
                       "NOT BUILD_DALI_NODEPS" OFF)  # Video support requires ffmpeg as well
set(BUILD_FFMPEG ${BUILD_NVDEC})

cmake_dependent_option(BUILD_NVJPEG "Build with nvJPEG support" ON
                       "NOT BUILD_DALI_NODEPS" OFF)

cmake_dependent_option(BUILD_NVJPEG2K "Build with nvJPEG2K support" ON
                        "NOT BUILD_DALI_NODEPS" OFF)

cmake_dependent_option(BUILD_NVOF "Build with NVIDIA OPTICAL FLOW SDK support" ON
                       "NOT BUILD_DALI_NODEPS" OFF)
cmake_dependent_option(BUILD_NVML "Build with NVIDIA Management Library (NVML) support" ON
                       "NOT BUILD_DALI_NODEPS" OFF)

cmake_dependent_option(WITH_DYNAMIC_NVJPEG "Dynamicly load nvJPEG" ON
                      "WITH_DYNAMIC_CUDA_TOOLKIT;BUILD_NVJPEG" OFF)
cmake_dependent_option(WITH_DYNAMIC_CUFFT "Dynamicly load cuFFT" ON
                      "WITH_DYNAMIC_CUDA_TOOLKIT" OFF)
cmake_dependent_option(WITH_DYNAMIC_NPP "Dynamicly load npp" ON
                      "WITH_DYNAMIC_CUDA_TOOLKIT" OFF)

if((${CUDA_VERSION} VERSION_GREATER_EQUAL "12.2") OR NOT (${ARCH} MATCHES "aarch64"))
  cmake_dependent_option(BUILD_CUFILE "Build with cufile (GPU Direct Storage) support" OFF
                         "NOT BUILD_DALI_NODEPS" OFF)
else()
  # make sure that even if set by -DBUILD_CUFILE, it will be unset as not supported
  unset(BUILD_CUFILE CACHE)
endif()

cmake_dependent_option(BUILD_NVCOMP "Build with nvCOMP support" OFF
                       "NOT BUILD_DALI_NODEPS" OFF)
if (BUILD_NVCOMP)
  cmake_dependent_option(WITH_DYNAMIC_NVCOMP "Dynamicly load nvcomp" ON
                        "WITH_DYNAMIC_CUDA_TOOLKIT" OFF)
else()
  set(WITH_DYNAMIC_NVCOMP OFF)
endif()

if (BUILD_DALI_NODEPS)
  set(BUILD_OPENCV OFF)
  set(BUILD_PROTOBUF OFF)
else()
  set(BUILD_OPENCV ON)
  set(BUILD_PROTOBUF ON)
endif()

# DALI modules
# Note dali_core is always enabled
set(BUILD_DALI_KERNELS ON)
if (BUILD_DALI_KERNELS AND NOT BUILD_DALI_NODEPS)
  set(BUILD_DALI_PIPELINE ON)
  set(BUILD_DALI_OPERATORS ON)
else()
  set(BUILD_DALI_PIPELINE OFF)
  set(BUILD_DALI_OPERATORS OFF)
endif()

# Experimental, only enabled for BUILD_DALI_NODEPS=ON
cmake_dependent_option(STATIC_LIBS "Build static libraries instead of shared-object libraries" OFF
                       "BUILD_DALI_NODEPS" OFF)

option(VERBOSE_LOGS "Adds verbose loging to DALI" OFF)
option(WERROR "Treat all warnings as errors" OFF)
option(RELWITHDEBINFO_CUDA_DEBUG "Add device side debug info for RelWithDebInfo build configuration" OFF)
option(FORMAT_PYTHON_STUB_FILES "Format *.pyi API stub files automatically" ON)

cmake_dependent_option(DALI_CLANG_ONLY "Compile DALI using only Clang. Suitable only for developement."
    OFF "CMAKE_CXX_COMPILER_ID STREQUAL Clang" OFF)

if (DALI_CLANG_ONLY AND BUILD_NVDEC)
  message(STATUS "NVDEC is not supportet when compiling only with Clang. Setting BUILD_NVDEC to OFF.")
  set(BUILD_NVDEC OFF)
endif()

message(STATUS "DALI_CLANG_ONLY -- ${DALI_CLANG_ONLY}")

if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  message(FATAL_ERROR "Expected CUDA compiler to be set to nvcc. Clang-only build is supported via DALI_CLANG_ONLY
      which requires setting Clang as C and CXX compilers and leaving nvcc as CUDA compiler.")
endif()

if (NOT BUILD_PYTHON AND PYTHON_VERSIONS)
  message(WARNING "The BUILD_PYTHON option is set to off and the PYTHON_VERSIONS variable is
          provided. The latter will be ignored.")
elseif (BUILD_PYTHON AND "${PYTHON_VERSIONS}" STREQUAL "")
  set(PYTHON_VERSIONS "3.9;3.10;3.11;3.12;3.13")
endif ()

if (BUILD_PYTHON)
  message(STATUS "Building DALI for Python versions: ${PYTHON_VERSIONS}")
  # Python binary can be used directly, otherwise we just run the python_stub_generator.py
  # and python3 is selected via shebang
  message(STATUS "Generating python stubs using interpreter: ${PYTHON_STUBGEN_INTERPRETER}")
  set(PYBIND11_FINDPYTHON OFF)
endif ()

cmake_dependent_option(PREBUILD_DALI_LIBS "Use prebuilt DALI libs to compile python bindings" OFF
                       "BUILD_PYTHON" OFF)

if (STATIC_LIBS)
  message (STATUS "Building static libraries")
  set(LIBTYPE STATIC)
else()
  message (STATUS "Building shared-object libraries")
  set(LIBTYPE SHARED)
endif()


foreach(rpath IN ITEMS
  "$ORIGIN"
  "$ORIGIN/../cufft/lib"
  "$ORIGIN/../npp/lib"
  "$ORIGIN/../nvjpeg/lib"
  "$ORIGIN/../nvimgcodec"
  "${NVIMGCODEC_DEFAULT_INSTALL_PATH}/lib64")
  message(STATUS "Add to rpath: ${rpath}")
  list(APPEND CMAKE_BUILD_RPATH ${rpath})
  list(APPEND CMAKE_INSTALL_RPATH ${rpath})
endforeach()

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

# Discover what architectures does nvcc support
if (DALI_CLANG_ONLY)
  set(USED_CUDA_COMPILER ${CMAKE_CXX_COMPILER})
else()
  set(USED_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})
endif()


message(STATUS "CUDA .cu files compiler: ${USED_CUDA_COMPILER}")

CUDA_find_supported_arch_values(CUDA_supported_archs ${USED_CUDA_COMPILER} ${CUDA_known_archs})
message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
list(SORT CUDA_TARGET_ARCHS_SORTED COMPARE NATURAL)
CUDA_find_supported_arch_values(CUDA_targeted_archs ${USED_CUDA_COMPILER} ${CUDA_TARGET_ARCHS_SORTED})
message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
if (NOT CUDA_targeted_archs)
  message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}})"
      " is supported by ${USED_CUDA_COMPILER}, use one or more of: ${CUDA_supported_archs}")
endif()


# Add gpu-arch and toolkit flags for clang when compiling cuda (if used)
if (DALI_CLANG_ONLY)
  CUDA_get_gencode_args(CUDA_gencode_flags_clang ${USED_CUDA_COMPILER} ${CUDA_targeted_archs})
  message(STATUS "Generated gencode flags for clang: ${CUDA_gencode_flags_clang}")
  CUDA_get_toolkit_from_compiler(CUDA_TOOLKIT_PATH_VAR)
  message(STATUS "Used CUDA toolkit: ${CUDA_TOOLKIT_PATH_VAR}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-cuda-version --cuda-path=${CUDA_TOOLKIT_PATH_VAR} ${CUDA_gencode_flags_clang}")
  # in CTK 13 cuda/std/utility has been moved to cccl/cuda/std/utility and clang doesn't know about this (nor CUTLASS yet)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_PATH_VAR}/include/cccl")
endif()

CUDA_get_cmake_cuda_archs(CMAKE_CUDA_ARCHITECTURES ${CUDA_targeted_archs})
message(STATUS "Generated CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"-fvisibility=hidden -Wno-free-nonheap-object\" --Wno-deprecated-gpu-targets -Xfatbin -compress-all")

if ("${CUDA_VERSION_MAJOR}" GREATER_EQUAL "13")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compress-mode=size")
endif()

# Suppress a few warnings from nvcc:
# calling a __host__ function(...) from a __host__ __device__ function(...) is not allowed
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=20011")
# 'long double' is treated as 'double' in device code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=20208")
# unrecognized #pragma in device code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=20199")
# overloaded virtual function "dali::Operator<dali::CPUBackend>::RunImpl" is only partially overridden in class "..."
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=611")

# Dependencies
if(${ARCH} MATCHES "aarch64-linux")
  message("Target set to aarch64-linux")
  include(cmake/Dependencies.aarch64-linux.cmake)
else()
  include(cmake/Dependencies.cmake)
endif()

propagate_option(BUILD_NVTX)
propagate_option(BUILD_PYTHON)
propagate_option(BUILD_SHM_WRAPPER)
propagate_option(BUILD_LMDB)
propagate_option(BUILD_JPEG_TURBO)
propagate_option(BUILD_LIBTIFF)
propagate_option(BUILD_LIBSND)
propagate_option(BUILD_LIBTAR)
propagate_option(BUILD_FFTS)
propagate_option(BUILD_CFITSIO)
propagate_option(BUILD_CVCUDA)
propagate_option(BUILD_NVJPEG)
propagate_option(BUILD_NVJPEG2K)
propagate_option(BUILD_NVOF)
propagate_option(BUILD_NVDEC)
propagate_option(BUILD_FFMPEG)
propagate_option(BUILD_NVCOMP)
propagate_option(BUILD_NVML)
propagate_option(BUILD_CUFILE)
propagate_option(BUILD_NVIMAGECODEC)
propagate_option(BUILD_AWSSDK)
propagate_option(LINK_DRIVER)
propagate_option(WITH_DYNAMIC_NVJPEG)
propagate_option(WITH_DYNAMIC_CUFFT)
propagate_option(WITH_DYNAMIC_NPP)
propagate_option(WITH_DYNAMIC_NVIMGCODEC)
propagate_option(WITH_DYNAMIC_NVCOMP)

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
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -g")
if (RELWITHDEBINFO_CUDA_DEBUG)
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} " -lineinfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 17)

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-free-nonheap-object -Wno-unused-variable -Wno-unused-function -Wno-strict-overflow -Wno-maybe-uninitialized -Wno-unknown-warning-option -fno-strict-aliasing -fPIC -fvisibility=hidden")

if (WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if (BUILD_WITH_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope")
endif()

if (BUILD_WITH_LSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
endif()

if (BUILD_WITH_UBSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
endif()

if ("${ARCH}" STREQUAL "aarch64-linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register -Wsign-compare -Wno-deprecated-declarations")

  # Settings specific to builds where Clang is used for compiling CUDA code
  if (DALI_CLANG_ONLY)
    # Enable SFINAE with ellipsis in device code
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fcuda-allow-variadic-functions")

    # TODO(klecki): Plethora of warnings that should be adressed as a followup
    # std::abs have no effect on unsigned value in a templated call
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-absolute-value")
    # convert.h when the bigger values are not representable
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-int-float-conversion")
    # Some aggregate constructors with inner object suggest double braces
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
    # Reductions do not cover all enum values
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
    # TYPE_SWITCH over bool exists in the wild
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-bool")
    # CUDA flags are passed to .cc files and are ignored
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
    # Ignore warnings coming from cutlass
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --system-header-prefix=cutlass/")
  endif()

  # CUDA does not support current Clang as host compiler, we need use gcc
  # CMAKE_CUDA_HOST_COMPILER variable operates on paths
  set(CUDA_UNSUPPORTED_COMPILER 0)
  if (NOT DALI_CLANG_ONLY)
    if ("${CMAKE_CUDA_HOST_COMPILER}" MATCHES "clang")
        set(CUDA_UNSUPPORTED_COMPILER 1)
    elseif (CMAKE_CUDA_HOST_COMPILER STREQUAL "")
        set(CUDA_UNSUPPORTED_COMPILER 1)
    endif()

    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})

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


message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

# OpenMP SIMD support
if(CXX_HAVE_OMP_SIMD AND NOT DALI_CLANG_ONLY)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-simd")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}")

add_custom_target("check")
add_custom_target("check-gtest")
add_dependencies("check" "check-gtest")

CUDA_move_toolkit_include_dirs_to_end()

# Project build
add_subdirectory(dali)

# HACK: Add __init__.pys as needed
file(WRITE ${CMAKE_BINARY_DIR}/dali/__init__.py "")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/lint.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Install.cmake)

add_library(DALI::dali_core ALIAS dali_core)

if (BUILD_DALI_KERNELS)
  add_library(DALI::dali_kernels ALIAS dali_kernels)
endif()

if (BUILD_DALI_PIPELINE)
  add_library(DALI::dali ALIAS dali)
endif()

if (BUILD_DALI_OPERATORS)
  add_library(DALI::dali_operators ALIAS dali_operators)
endif()
