# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#############################
#        CUDA TOOLKIT
#############################

# TODO(klecki): Setting them directly from command line does not work
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOST})
set(CUDA_TOOLKIT_TARGET_DIR ${CUDA_TARGET})

set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${CUDA_TARGET}/lib")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CUDA_TARGET}/include")

list(APPEND DALI_LIBS ${CUDA_LIBRARIES})

CUDA_find_library(CUDART_LIB cudart_static)
list(APPEND DALI_EXCLUDES libcudart_static.a)

# NVIDIA NPPC library
if (NOT WITH_DYNAMIC_NPP)
  CUDA_find_library(CUDA_nppicc_LIBRARY nppicc_static)
  CUDA_find_library(CUDA_nppig_LIBRARY nppig_static)
  CUDA_find_library(CUDA_nppc_LIBRARY nppc_static)
  list(APPEND DALI_LIBS ${CUDA_nppicc_LIBRARY})
  list(APPEND DALI_EXCLUDES libnppicc_static.a)
  list(APPEND DALI_LIBS ${CUDA_nppig_LIBRARY})
  list(APPEND DALI_EXCLUDES libnppig_static.a)
  list(APPEND DALI_LIBS ${CUDA_nppc_LIBRARY})
  list(APPEND DALI_EXCLUDES libnppc_static.a)
endif ()

# cuFFT library
if (NOT WITH_DYNAMIC_CUFFT)
  CUDA_find_library(CUDA_cufft_LIBRARY cufft_static)
  list(APPEND DALI_EXCLUDES libcufft_static.a)
endif ()

# CULIBOS needed when using static CUDA libs
if (NOT WITH_DYNAMIC_CUDA_TOOLKIT)
  CUDA_find_library(CUDA_culibos_LIBRARY culibos)
  list(APPEND DALI_LIBS ${CUDA_culibos_LIBRARY})
  list(APPEND DALI_EXCLUDES libculibos.a)
endif ()

# TODO(klecki): Do we need host includes?
include_directories(${CUDA_TOOLKIT_TARGET_DIR}/include)

# NVTX for profiling
if (NVTX_ENABLED)
  if(${CUDA_VERSION} VERSION_LESS "10.0")
     CUDA_find_library(CUDA_nvToolsExt_LIBRARY nvToolsExt)
     list(APPEND DALI_LIBS ${CUDA_nvToolsExt_LIBRARY})
  endif()
endif()

##################################################################
# Common dependencies
##################################################################
include(cmake/Dependencies.common.cmake)

##################################################################
# protobuf
##################################################################
set(Protobuf_CROSS YES)
set(Protobuf_USE_STATIC_LIBS YES)
# deliberatelly use protobuf instead of Protobuf to use protobuf provided cmake configuration file
# then use Protobuf to utilize our FindProtobuf.cmake to discover the rest
find_package(protobuf REQUIRED CONFIG)
find_package(Protobuf 3.6.1 REQUIRED)
message(STATUS "Enabling TensorFlow TFRecord file format support")
add_definitions(-DDALI_BUILD_PROTO3=1)
set(BUILD_PROTO3 ON CACHE STRING "Build proto3")

include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})
list(APPEND DALI_LIBS protobuf::libprotobuf)

set(DALI_SYSTEM_LIBS rt pthread m dl)
list(APPEND DALI_LIBS ${CUDART_LIB} ${DALI_SYSTEM_LIBS})

list(APPEND DALI_EXCLUDES libprotobuf.a;libprotobuf-lite.a;libprotoc.a)
# find all the libraries that protobuf::libprotobuf depends on
get_link_libraries(PROTO_LIB_DEPS protobuf::libprotobuf)
# libutf8_validity.a is a result of a generator expression, add it manually as it is
# hard/impossible to learn its name during the configuration phase
list(APPEND DALI_EXCLUDES ${PROTO_LIB_DEPS} libutf8_validity.a)

##################################################################
# pybind 11
##################################################################
set(PYBIND11_USE_CROSSCOMPILING ON)