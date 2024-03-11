# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.21)
set(PLUGIN_NAME "video")
project(dali_plugin_${PLUGIN_NAME} LANGUAGES CUDA CXX C)

# Silence DOWNLOAD_EXTRACT_TIMESTAMP warning in CMake 3.24
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

include(GNUInstallDirs)
include(FetchContent)
include(common.cmake)

set(FFMPEG_DIR $ENV{FFMPEG_DIR})
find_package(PkgConfig REQUIRED)

set(PYTHON_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
if(NOT EXISTS ${PYTHON_EXECUTABLE})
  message(FATAL_ERROR "Python executable not set\n")
endif()

list(APPEND CMAKE_BUILD_RPATH "$ORIGIN")          # current directory
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN")        # current directory
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../..")  # DALI dir is ../../ from plugin/${PLUGIN_NAME}

set(FFMPEG_RPATH "")
if (NOT "${FFMPEG_DIR}" STREQUAL "")
  set(CMAKE_PREFIX_PATH ${FFMPEG_DIR})
  message(STATUS "Using CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
endif()
pkg_check_modules(
    FFMPEG
    IMPORTED_TARGET
    # Ubuntu 20.04 versions
    libavfilter>=7.57.100
    libavformat>=58.29.100
    libavcodec>=58.54.100
    libswresample>=3.5.100
    libavutil>=56.31.100
)
if (NOT "${FFMPEG_FOUND}" STREQUAL "1")
  message(WARNING
    "FFmpeg not found in the system or the dir pointed by FFMPEG_DIR environment variable. "
    "Will download and build a minimal version.")

  set(FFMPEG_SOURCE_URL    https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n6.1.1.tar.gz)
  set(FFMPEG_SOURCE_URL    file:///opt/src/plugins/n6.1.1.tar.gz)
  set(FFMPEG_SOURCE_SHA512 a84209fe36a2a0262ebc34b727e7600b12d4739991a95599d7b4df533791b12e2e43586ccc6ff26aab2f935a3049866204e322ec0c5e49e378fc175ded34e183)
  FetchContent_Declare(
      ffmpeg
      URL ${FFMPEG_SOURCE_URL}
      URL_HASH SHA512=${FFMPEG_SOURCE_SHA512}
  )
  FetchContent_Populate(ffmpeg)

  set(FFMPEG_SRC ${CMAKE_CURRENT_BINARY_DIR}/_deps/ffmpeg-src)
  set(FFMPEG_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/ffmpeg)
  set(ENV{FFMPEG_DIR} ${FFMPEG_DIR})
  set(CMAKE_PREFIX_PATH ${FFMPEG_DIR})
  file(MAKE_DIRECTORY ${FFMPEG_DIR})
  execute_process(
    COMMAND_ERROR_IS_FATAL ANY
    WORKING_DIRECTORY ${FFMPEG_SRC}
    COMMAND /bin/bash -ex ${CMAKE_CURRENT_SOURCE_DIR}/build_ffmpeg.sh ${FFMPEG_DIR})
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps/ffmpeg/lib/
    DESTINATION nvidia/dali/plugin/${PLUGIN_NAME}/deps/ffmpeg
    FILES_MATCHING PATTERN "*.so*"
  )
  list(APPEND CMAKE_BUILD_RPATH "$ORIGIN/_deps/ffmpeg/lib")
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/deps/ffmpeg")
else()
list(APPEND CMAKE_BUILD_RPATH "${FFMPEG_LIBRARY_DIRS}")
list(APPEND CMAKE_INSTALL_RPATH "${FFMPEG_LIBRARY_DIRS}")
endif()

FetchContent_Declare(
    pynvvideocodec
    URL      https://api.ngc.nvidia.com/v2/resources/nvidia/py_nvvideocodec/versions/0.0.9/zip
    URL_HASH SHA512=b73d06f2239bc50a6a9971949a3579df56d8a6c223d64f5fcbebb3531b9adff2e9f25d23b1922bc42616e65fc850d60c085cadfd7daecbfc562f9959995f28a5
)
FetchContent_Populate(pynvvideocodec)
message(STATUS "pynvvideocodec_SOURCE_DIR=${pynvvideocodec_SOURCE_DIR}")
include_directories(SYSTEM "${pynvvideocodec_SOURCE_DIR}/src/")
add_subdirectory("${pynvvideocodec_SOURCE_DIR}/src/VideoCodecSDKUtils/")

set(VIDEO_PLUGIN_SOURCES
    src/decoder/video_decoder_mixed.cc
)

add_library(dali_${PLUGIN_NAME} SHARED ${VIDEO_PLUGIN_SOURCES})
set_target_properties(dali_${PLUGIN_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(dali_${PLUGIN_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
target_include_directories(dali_${PLUGIN_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_directories(dali_${PLUGIN_NAME} PUBLIC ${DALI_LIB_DIR})
target_link_libraries(dali_${PLUGIN_NAME} PUBLIC dali dali_core dali_kernels VideoCodecSDKUtils)

# Generates .pyi stubs
# Need to add the _deps/ffmpeg/lib dir so that the plugin can be loaded
add_custom_target(dali_${PLUGIN_NAME}_generate_stubs ALL
  DEPENDS dali_${PLUGIN_NAME}
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/fn/plugin/video/
  COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/generate_plugin_stubs.py
      ${CMAKE_CURRENT_BINARY_DIR}
      ${PROJECT_BINARY_DIR}/libdali_${PLUGIN_NAME}.so
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
