# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set(PLUGIN_NAME "video")
project(dali_plugin_${PLUGIN_NAME} LANGUAGES CUDA CXX C)

include(GNUInstallDirs)
include(FetchContent)
include(common.cmake)

# Silence DOWNLOAD_EXTRACT_TIMESTAMP warning in CMake 3.24
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

set(PYTHON_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
if(NOT EXISTS ${PYTHON_EXECUTABLE})
  message(FATAL_ERROR "Python executable not set\n")
endif()

set(PYNVVIDEOCODEC_SOURCE_URL $ENV{PYNVVIDEOCODEC_SOURCE_URL})
set(PYNVVIDEOCODEC_SOURCE_SHA256 $ENV{PYNVVIDEOCODEC_SOURCE_SHA256})
if ("${PYNVVIDEOCODEC_SOURCE_URL}" STREQUAL "")
  set(PYNVVIDEOCODEC_SOURCE_URL    "https://files.pythonhosted.org/packages/23/3b/df68633705b2d777edb4d9f3391f106cfd8cd629a760973074493b065340/PyNvVideoCodec-1.0.2.tar.gz")
  set(PYNVVIDEOCODEC_SOURCE_SHA256 "02093d826eeece53928102e99e413f6c6d84e1b3c186062f2b5676d21945ac62")
endif()
message(STATUS "PYNVVIDEOCODEC_SOURCE_URL=${PYNVVIDEOCODEC_SOURCE_URL}")

FetchContent_Declare(
    pynvvideocodec
    URL      ${PYNVVIDEOCODEC_SOURCE_URL}
    URL_HASH SHA256=${PYNVVIDEOCODEC_SOURCE_SHA256}
    PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/CUDA-13.0.patch
)
FetchContent_Populate(pynvvideocodec)

include(ffmpeg.cmake)

message(STATUS "pynvvideocodec_SOURCE_DIR=${pynvvideocodec_SOURCE_DIR}")
include_directories(SYSTEM "${pynvvideocodec_SOURCE_DIR}/src/")
set(USE_PKGCONFIG OFF)
set(ENV{FFMPEG_DIR} ${FFMPEG_DIR})
add_subdirectory("${pynvvideocodec_SOURCE_DIR}/src/VideoCodecSDKUtils/")

set(VIDEO_PLUGIN_SOURCES
    src/decoder/video_decoder_mixed.cc
    src/decoder/color_space.cu
)

add_library(dali_${PLUGIN_NAME} SHARED ${VIDEO_PLUGIN_SOURCES})
set_target_properties(dali_${PLUGIN_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(dali_${PLUGIN_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
target_include_directories(dali_${PLUGIN_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_directories(dali_${PLUGIN_NAME} PUBLIC ${DALI_LIB_DIR})
target_link_libraries(dali_${PLUGIN_NAME} PUBLIC dali dali_core dali_kernels VideoCodecSDKUtils)

# Need to find the libcuda.so stub so that the shared-object library can be loaded
# to generate the Python signature files
find_library(cuda_LIBRARY cuda
             PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
             PATH_SUFFIXES lib/stubs lib64/stubs)

if(NOT $ENV{LD_PRELOAD} STREQUAL "")
    set(NEW_LD_PRELOAD "$ENV{LD_PRELOAD} ${cuda_LIBRARY}")
else()
    set(NEW_LD_PRELOAD "${cuda_LIBRARY}")
endif()

# it allows injecting extra preload that is not suppose to be system wide
if(NOT $ENV{LD_EXTRA_PRELOAD} STREQUAL "")
    set(NEW_LD_PRELOAD "${NEW_LD_PRELOAD} $ENV{LD_EXTRA_PRELOAD}")
endif()

# Generates .pyi stubs
# Note: Needs to preload libcuda.so stub so that the shared-object library can be loaded
add_custom_target(dali_${PLUGIN_NAME}_generate_stubs ALL
DEPENDS dali_${PLUGIN_NAME}
BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/fn/plugin/video/
COMMAND /bin/bash -c
    "LD_PRELOAD=\"${NEW_LD_PRELOAD}\" \
    ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/generate_plugin_stubs.py \
    ${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_BINARY_DIR}/libdali_${PLUGIN_NAME}.so"
)

install(TARGETS dali_${PLUGIN_NAME}
    RUNTIME DESTINATION nvidia/dali/plugin/${PLUGIN_NAME}
    LIBRARY DESTINATION nvidia/dali/plugin/${PLUGIN_NAME}
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fn/plugin/${PLUGIN_NAME}/
    DESTINATION nvidia/dali/fn/plugin/${PLUGIN_NAME}
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ops/plugin/${PLUGIN_NAME}/
    DESTINATION nvidia/dali/ops/plugin/${PLUGIN_NAME}
)
