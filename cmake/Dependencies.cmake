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

##################################################################
# CUDA Toolkit libraries (including NVJPEG)
##################################################################

CUDA_find_library(CUDART_LIB cudart_static)
list(APPEND DALI_EXCLUDES libcudart_static.a)

list(APPEND DALI_LIBS ${CUDART_LIB})

set(CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# For NVJPEG
if (BUILD_NVJPEG)
  find_package(NVJPEG 9.0 REQUIRED)
  if(${CUDA_VERSION} VERSION_LESS ${NVJPEG_VERSION})
    message(WARNING "Using nvJPEG ${NVJPEG_VERSION} together with CUDA ${CUDA_VERSION} "
                    "requires NVIDIA drivers compatible with CUDA ${NVJPEG_VERSION} or later")
  endif()
  include_directories(SYSTEM ${NVJPEG_INCLUDE_DIR})
  list(APPEND DALI_LIBS ${NVJPEG_LIBRARY})
  list(APPEND DALI_EXCLUDES libnvjpeg_static.a)
  add_definitions(-DDALI_USE_NVJPEG)

  if (${NVJPEG_LIBRARY_0_2_0})
    add_definitions(-DNVJPEG_LIBRARY_0_2_0)
  endif()

  if (${NVJPEG_DECOUPLED_API})
    add_definitions(-DNVJPEG_DECOUPLED_API)
  endif()
endif()

# NVIDIA NPPI library
list(APPEND DALI_LIBS nppicc_static)
list(APPEND DALI_EXCLUDES libnppicc_static.a)
# NVIDIA NPPC library
list(APPEND DALI_LIBS nppc_static)
list(APPEND DALI_EXCLUDES libnppc_static.a)

# CULIBOS needed when using static CUDA libs
list(APPEND DALI_LIBS culibos)
list(APPEND DALI_EXCLUDES libculibos.a)

# NVTX for profiling
if (BUILD_NVTX)
  list(APPEND DALI_LIBS nvToolsExt)
  add_definitions(-DDALI_USE_NVTX)
endif()

if (VERBOSE_LOGS)
  add_definitions(-DDALI_VERBOSE_LOGS)
endif()


include(cmake/Dependencies.common.cmake)

##################################################################
# protobuf
##################################################################
# link statically
if(NOT DEFINED Protobuf_USE_STATIC_LIBS)
set(Protobuf_USE_STATIC_LIBS YES)
endif(NOT DEFINED Protobuf_USE_STATIC_LIBS)
find_package(Protobuf 2.0 REQUIRED)
if(${Protobuf_VERSION} VERSION_LESS "3.0")
  message(STATUS "TensorFlow TFRecord file format support is not available with Protobuf 2")
else()
  message(STATUS "Enabling TensorFlow TFRecord file format support")
  add_definitions(-DDALI_BUILD_PROTO3=1)
  set(BUILD_PROTO3 ON CACHE STRING "Build proto3")
endif()

include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})
list(APPEND DALI_LIBS ${Protobuf_LIBRARY})
# hide things from the protobuf, all we export is only is API generated from our proto files
list(APPEND DALI_EXCLUDES libprotobuf.a)


##################################################################
# Exclude stdlib
##################################################################
list(APPEND DALI_EXCLUDES libsupc++.a;libstdc++.a;libstdc++_nonshared.a;)


##################################################################
# Turing Optical flow API
##################################################################
if(BUILD_NVOF)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/turing_of)
endif()
