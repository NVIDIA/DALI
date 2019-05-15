# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

# Check QNX
if(NOT DEFINED ENV{QNX_TARGET} OR NOT DEFINED ENV{QNX_HOST})
  message("QNX_TARGET and QNX_HOST not exported")
endif()

set(CMAKE_SYSROOT $ENV{QNX_TARGET})

if(NOT IS_DIRECTORY ${CMAKE_SYSROOT}/aarch64le)
  message(FATAL_ERROR"[ERROR] Please set $QNX_TARGET like below.\n $ export QNX_TARGET=/PATH/TO/qnx700/target/qnx7\n")
endif()

if(NOT EXISTS ${CMAKE_C_COMPILER})
  message(FATAL_ERROR "[ERROR] Please set $QNX_HOST like below.\n $ export QNX_HOST=/ PATH/TO/qnx700/host/linux/x86_64\n")
endif()

set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOST})
set(CUDA_TOOLKIT_TARGET_DIR ${CUDA_TARGET})
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF) # "QNX does not have librt.so"

find_package(CUDA 10.0 REQUIRED)

# find_library(CUDA_CUDART_LIBRARY cudart
#   PATHS ${CUDA_TOOLKIT_TARGET_DIR}
#   PATH_SUFFIXES lib64 lib)

message(STATUS "Found cudart at ${CUDA_LIBRARIES}")

# set(CUDA_TOOLKIT_ROOT_DIR_INTERNAL ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "Cuda toolkit internal root location")
# set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL ${CUDA_TOOLKIT_TARGET_DIR} CACHE PATH "Cuda toolkit target location")

# set(CUDA_LIBRARIES ${CUDA_TOOLKIT_TARGET_DIR}/lib)
# set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "QNX does not have librt.so")

include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${CUDA_LIBRARIES})

# NVIDIA NPPC library
find_cuda_helper_libs(nppc_static)
find_cuda_helper_libs(nppicom_static)
find_cuda_helper_libs(nppicc_static)
find_cuda_helper_libs(nppig_static)
list(APPEND DALI_LIBS ${CUDA_nppicom_static_LIBRARY}
  ${CUDA_nppicc_static_LIBRARY}
  ${CUDA_nppig_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppicom_static.a
  libnppicc_static.a
  libnppig_static.a)
list(APPEND DALI_LIBS ${CUDA_nppc_static_LIBRARY})
list(APPEND DALI_EXCLUDES libnppc_static.a)

include_directories(${CUDA_TOOLKIT_TARGET_DIR}/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CUDA_LIBRARIES} -L${CUDA_LIBRARIES}/stubs -lcudart -lnppc_static -lnppicom_static -lnppicc_static -lnppig_static -lnpps -lnppc -lculibos")

# NVTX for profiling
if (BUILD_NVTX)
  find_cuda_helper_libs(nvToolsExt)
  list(APPEND DALI_LIBS ${CUDA_nvToolsExt_LIBRARY})
  add_definitions(-DDALI_USE_NVTX)
endif()

##################################################################
# Common dependencies
##################################################################
include(cmake/Dependencies.common.cmake)

##################################################################
# Protobuf
##################################################################
set(Protobuf_CROSS YES)
set(Protobuf_USE_STATIC_LIBS YES)
find_package(Protobuf 2.0 REQUIRED)
if(${Protobuf_VERSION} VERSION_LESS "3.0")
  message(STATUS "TensorFlow TFRecord file format support is not available with Protobuf 2")
else()
  message(STATUS "Enabling TensorFlow TFRecord file format support")
  add_definitions(-DDALI_BUILD_PROTO3=1)
  set(BUILD_PROTO3 ON CACHE STRING "Build proto3")
endif()

include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${PROTOBUF_LIBRARY})
list(APPEND DALI_LIBS ${Protobuf_LIBRARY} ${Protobuf_PROTOC_LIBRARIES} ${Protobuf_LITE_LIBRARIES})
list(APPEND DALI_EXCLUDES libprotobuf.a;libprotobuf-lite.a;libprotoc.a)
