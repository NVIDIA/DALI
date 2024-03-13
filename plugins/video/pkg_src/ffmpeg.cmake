# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

option(BUILD_FFMPEG "Whether we want to always build FFmpeg from source" $ENV{BUILD_FFMPEG})
message(STATUS "env(FFMPEG_DIR) : $ENV{FFMPEG_DIR}")
message(STATUS "opt(BUILD_FFMPEG) : ${BUILD_FFMPEG}")

set(FFMPEG_SOURCE_URL $ENV{FFMPEG_SOURCE_URL})
set(FFMPEG_SOURCE_SHA512 $ENV{FFMPEG_SOURCE_SHA512})
if ("${FFMPEG_SOURCE_URL}" STREQUAL "")
  set(FFMPEG_SOURCE_URL    https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n6.1.1.tar.gz)
  set(FFMPEG_SOURCE_SHA512 a84209fe36a2a0262ebc34b727e7600b12d4739991a95599d7b4df533791b12e2e43586ccc6ff26aab2f935a3049866204e322ec0c5e49e378fc175ded34e183)
endif()

# Look for it first in $ENV{FFMPEG_DIR}, then in the prebuilt version found in pynvvideocodec
if (NOT BUILD_FFMPEG)
  # Check ENV{FFMPEG_DIR} first
  find_path(
      FFMPEG_DIR
      NAMES "lib/libavformat.so"
            "lib/${CMAKE_HOST_SYSTEM_PROCESSOR}/libavformat.so"
      PATHS $ENV{FFMPEG_DIR}
      NO_DEFAULT_PATH
  )
  set(BUNDLE_FFMPEG_LIBS OFF)
  if (${FFMPEG_DIR} STREQUAL "FFMPEG_DIR-NOTFOUND")
    # Check pynvvvideocodec FFmpeg now
    find_path(
        FFMPEG_DIR
        NAMES "lib/libavformat.so"
              "lib/${CMAKE_HOST_SYSTEM_PROCESSOR}/libavformat.so"
        PATHS ${pynvvideocodec_SOURCE_DIR}/ffmpeg
        NO_DEFAULT_PATH
    )
    if (${FFMPEG_DIR} STREQUAL "FFMPEG_DIR-NOTFOUND")
      set(BUILD_FFMPEG ON)  # Force build from source
      message(WARNING
        "FFmpeg not found in the system or the dir pointed by FFMPEG_DIR environment variable. "
        "Will download and build a minimal version.")
    else()
      set(BUNDLE_FFMPEG_LIBS ON)
      message(STATUS "Will bundle FFmpeg libs")
    endif()
  endif()
endif()

if (NOT BUILD_FFMPEG)
  find_path(
      FFMPEG_LIBRARY_DIR
      NAMES "libavformat.so"
      PATHS ${FFMPEG_DIR}/lib ${FFMPEG_DIR}/lib/${CMAKE_HOST_SYSTEM_PROCESSOR}
      NO_DEFAULT_PATH
  )
  message(STATUS "FFMPEG_DIR=${FFMPEG_DIR}")
  message(STATUS "FFMPEG_LIBRARY_DIR=${FFMPEG_LIBRARY_DIR}")

  if (BUNDLE_FFMPEG_LIBS)
    install(
      DIRECTORY ${FFMPEG_LIBRARY_DIR}
      DESTINATION nvidia/dali/plugin/${PLUGIN_NAME}/deps/ffmpeg
      FILES_MATCHING PATTERN "*.so*"
    )
  endif()
endif()

list(APPEND CMAKE_BUILD_RPATH "$ORIGIN")          # current directory
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN")        # current directory
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../..")  # DALI dir is ../../ from plugin/${PLUGIN_NAME}

if (BUILD_FFMPEG)
  message(STATUS "Building from ${FFMPEG_SOURCE_URL}")

  FetchContent_Declare(
      ffmpeg
      URL ${FFMPEG_SOURCE_URL}
      URL_HASH SHA512=${FFMPEG_SOURCE_SHA512}
  )
  FetchContent_Populate(ffmpeg)

  set(FFMPEG_SRC ${CMAKE_CURRENT_BINARY_DIR}/_deps/ffmpeg-src)
  set(FFMPEG_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/ffmpeg)
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
  set(BUNDLE_FFMPEG_LIBS ON)
  set(FFMPEG_LIBRARY_DIR ${FFMPEG_DIR}/lib)
endif()

list(APPEND CMAKE_BUILD_RPATH "${FFMPEG_LIBRARY_DIR}")
if (BUNDLE_FFMPEG_LIBS)
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/deps/ffmpeg")
else()
  list(APPEND CMAKE_INSTALL_RPATH "${FFMPEG_LIBRARY_DIR}")
endif()