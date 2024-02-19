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
#set(CMAKE_CUDA_ARCHITECTURES "50;60;70;80;90")
project(dali_video LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(PYTHON_EXECUTABLE "Python executable" OFF)
if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
    if (NOT PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE "python")
    endif()
endif()
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")

option(DALI_COMPILE_FLAGS "DALI compile flags" OFF)
if (NOT DALI_COMPILE_FLAGS)
    set(DALI_COMPILE_FLAGS $ENV{DALI_COMPILE_FLAGS})
    if (NOT DALI_COMPILE_FLAGS)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import nvidia.dali as dali; compile_flags=' '.join(dali.sysconfig.get_compile_flags()); print(compile_flags, end='')"
            OUTPUT_VARIABLE DALI_COMPILE_FLAGS
            RESULT_VARIABLE COMPILE_FLAGS_RESULT)

        if (${COMPILE_FLAGS_RESULT} EQUAL "1")
            message(FATAL_ERROR "Failed to get compile flags for DALI. Make sure that DALI is installed.")
        endif()
    endif()
endif()
message(STATUS "DALI_COMPILE_FLAGS=${DALI_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} -Wall -fPIC -fvisibility=hidden")


option(DALI_LIB_DIR "DALI library dir" OFF)
if (NOT DALI_LIB_DIR)
    set(DALI_LIB_DIR $ENV{DALI_LIB_DIR})
    if (NOT DALI_LIB_DIR)
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir(), end='')"
            OUTPUT_VARIABLE DALI_LIB_DIR
            RESULT_VARIABLE LIB_DIR_RESULT)

        if (${LIB_DIR_RESULT} EQUAL "1")
            message(FATAL_ERROR "Failed to get library paths for DALI. Make sure that DALI is installed.")
        endif()
    endif()
endif()
message(STATUS "DALI_LIB_DIR=${DALI_LIB_DIR}")

set(VIDEO_PLUGIN_SOURCES
    loader/frames_decoder.cc
    loader/frames_decoder_gpu.cc
    loader/video_loader_decoder_cpu.cc
    loader/video_loader_decoder_gpu.cc
    input/video_input_cpu.cc
    input/video_input_mixed.cc
    decoder/video_decoder_cpu.cc
    decoder/video_decoder_mixed.cc
)

add_library(dali_video SHARED ${VIDEO_PLUGIN_SOURCES})
set_target_properties(dali_video PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dali_video PUBLIC ${dali_video_SRC_DIR})
target_link_directories(dali_video PUBLIC ${DALI_LIB_DIR})
target_link_libraries(dali_video PUBLIC dali dali_core dali_kernels VideoCodecSDKUtils)

find_package(PkgConfig REQUIRED)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(
        LIBAV
        REQUIRED
        IMPORTED_TARGET
        # Ubuntu 20.04 system packages
        libavfilter>=7.57.100
        libavformat>=58.29.100
        libavcodec>=58.54.100
        libswresample>=3.5.100
        libavutil>=56.31.100
    )
    target_link_libraries(dali_video PUBLIC PkgConfig::LIBAV)
else()
    set(NV_FFMPEG_INCLUDE_DIR "" CACHE PATH "Where to find ffmpeg includes")
    set(NV_FFMPEG_LIBRARY_DIR "" CACHE PATH "Where to find ffmpeg libraries")
    message(STATUS "NV_FFMPEG_INCLUDE_DIR=\"${FFMPEG_DIR}/include\"")
    message(STATUS "FFMPEG_DIR=\"${FFMPEG_DIR}\"")
    message(STATUS "NV_FFMPEG_LIBRARY_DIR=\"${FFMPEG_DIR}/lib\"")

    macro(link_av_component target lib_name)
        find_path(
            ${lib_name}_INCLUDE_DIR
            NAMES "${lib_name}/${lib_name}.h" "lib${lib_name}/${lib_name}.h"
            HINTS ${NV_FFMPEG_INCLUDE_DIR} "${FFMPEG_DIR}/include"
        )
        find_library(
            ${lib_name}_LIBRARY
            NAMES "${lib_name}"
            HINTS ${NV_FFMPEG_LIBRARY_DIR} "${FFMPEG_DIR}/lib"
        )
        message(STATUS "${lib_name}: library=${${lib_name}_LIBRARY}, include dir=${${lib_name}_INCLUDE_DIR}")
        if(NOT ${lib_name}_LIBRARY)
            message(
                FATAL_ERROR
                "Could not find ${lib_name}_LIBRARY! Please set FFMPEG_DIR to indicate its location!"
            )
        endif()
        if(NOT ${lib_name}_INCLUDE_DIR)
            message(
                FATAL_ERROR
                "Could not find ${lib_name}_INCLUDE_DIR! Please set FFMPEG_DIR to indicate its location!"
            )
        endif()
        target_link_libraries(dali_video PUBLIC ${${lib_name}_LIBRARY})
        target_include_directories(dali_video PUBLIC ${${lib_name}_INCLUDE_DIR})
    endmacro()
    link_av_component(dali_video avfilter)
    link_av_component(dali_video avformat)
    link_av_component(dali_video avcodec)
    link_av_component(dali_video swresample)
    link_av_component(dali_video avutil)
endif()

include(GNUInstallDirs)

install(TARGETS dali_video
    RUNTIME DESTINATION dali_video
    LIBRARY DESTINATION dali_video
)
