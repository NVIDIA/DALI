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
  set(BUNDLE_FFMPEG_LIBS OFF)
  # Check ENV{FFMPEG_DIR} first
  if (DEFINED ENV{FFMPEG_DIR})
    message(STATUS "Looking for FFMPEG in ENV{FFMPEG_DIR}=$ENV{FFMPEG_DIR} or in the system")
    find_path(
      FFMPEG_DIR
      NAMES "lib/libavformat.so"
            "lib/${CMAKE_HOST_SYSTEM_PROCESSOR}/libavformat.so"
      PATHS $ENV{FFMPEG_DIR}
      NO_DEFAULT_PATH
    )
    if (${FFMPEG_DIR} STREQUAL "FFMPEG_DIR-NOTFOUND")
      find_path(
        FFMPEG_DIR
        NAMES "lib/libavformat.so"
              "lib/${CMAKE_HOST_SYSTEM_PROCESSOR}/libavformat.so"
        PATHS ${pynvvideocodec_SOURCE_DIR}/external/ffmpeg
        NO_DEFAULT_PATH
      )

      if (${FFMPEG_DIR} STREQUAL "FFMPEG_DIR-NOTFOUND")
        message(FATAL_ERROR
          "Could not find ffmpeg libs in provided ENV{FFMPEG_DIR}=$ENV{FFMPEG_DIR}. "
          "Make sure the FFMPEG_DIR points to the ffmpeg root path. "
          "Alternatively, set environment variable BUILD_FFMPEG to force building from source.")
      else()
        set(BUNDLE_FFMPEG_LIBS ON)
        set(FFMPEG_LIBRARY_DIR ${FFMPEG_DIR}/lib)
        install(
          DIRECTORY ${FFMPEG_LIBRARY_DIR}
          DESTINATION nvidia/dali/plugin/${PLUGIN_NAME}/deps/ffmpeg
          FILES_MATCHING PATTERN "*.so*"
        )
      endif()
    endif()
  else()
    message(STATUS "ENV{FFMPEG_DIR} not set. Looking for ffmpeg in the system")
    set(FFMPEG_LIBS_FOUND ON)
    macro(find_av_component lib_name)
      find_path(${lib_name}_INCLUDE_DIR NAMES "${lib_name}/${lib_name}.h" "lib${lib_name}/${lib_name}.h")
      find_library(${lib_name}_LIBRARY NAMES "${lib_name}")
      if (NOT ${lib_name}_INCLUDE_DIR OR NOT ${lib_name}_LIBRARY)
        set(FFMPEG_LIBS_FOUND OFF)
      else()
        message(STATUS "Found ${lib_name}: ${${lib_name}_LIBRARY}")
      endif()
    endmacro()
    find_av_component(avfilter)
    find_av_component(avformat)
    find_av_component(avcodec)
    find_av_component(swresample)
    find_av_component(avutil)

    if (NOT FFMPEG_LIBS_FOUND)
      message(FATAL_ERROR
            "Could not find ffmpeg libs in the system."
            "You can use FFMPEG_DIR environment variable to point to the ffmpeg root path."
            "Alternatively, you can set the environment variable BUILD_FFMPEG to force building from source.")
    endif()
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