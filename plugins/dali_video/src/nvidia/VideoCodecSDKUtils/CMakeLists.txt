# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2010-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

message(STATUS "FFMPEG_DIR=$ENV{FFMPEG_DIR}")
cmake_minimum_required(VERSION 3.20)

# This must be set before enabling CUDA. Otherwise it will be set to the default ARCH by nvcc (e.g. 52 by nvcc 11.7)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Define Volta / Turing / Ampere / Ada if not set by the user
    set(CMAKE_CUDA_ARCHITECTURES
        60
        70
        72
        75
        80
        86
        #89 # Require CUDA 11.8 for CC 89 or cmake 3.23 for CMAKE_CUDA_ARCHITECTURES_{ALL,ALL_MAJOR}
    )
endif()

project(VideoCodecSDKUtils LANGUAGES CXX CUDA)

set(FFMPEG_DIR $ENV{FFMPEG_DIR})
set(USE_NVTX $ENV{USE_NVTX})

set(CODEC_SOURCES
 helper_classes/NvCodec/NvDecoder/NvDecoder.cpp
 helper_classes/NvCodec/NvEncoder/NvEncoder.cpp
)
set(CODEC_HDRS
 helper_classes/NvCodec/NvDecoder/NvDecoder.h
 helper_classes/NvCodec/NvEncoder/NvEncoder.h
 helper_classes/Utils/NvCodecUtils.h
 helper_classes/Utils/FFmpegDemuxer.h
 helper_classes/Utils/ColorSpace.h
 helper_classes/Utils/FFmpegStreamer.h
 helper_classes/Utils/Logger.h
 helper_classes/Utils/NvEncoderCLIOptions.h
 helper_classes/Utils/cuvidFunctions.h
 helper_classes/Utils/cuvid_function_pointer.h
 helper_classes/Utils/cuvid_dlopen.h
 helper_classes/Utils/cuvid_dlopen_unix.cpp
 helper_classes/Utils/cuvid_dlopen_windows.cpp
 helper_classes/Utils/
 Interface/cuviddec.h
 Interface/nvcuvid.h
 Interface/nvEncodeAPI.h
)

set(CODEC_CUDA_UTILS
 helper_classes/Utils/ColorSpace.cu
)

if(WIN32)
    set(NV_FFMPEG_HDRS ${FFMPEG_DIR}/include)
endif()

find_package(CUDAToolkit 11.2 REQUIRED)
include(GenerateExportHeader)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=\"sm_50,compute_50\")
# CUDA 11.7 and later is compiled with /MT option (staticaly linked with C runtime), 
# since we are statically linking cudart_static.lib with the app, 
# we need to make sure that all .obj files linked are compiled with /MT to avoid linker warning LNK4098
if (WIN32 AND CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 7)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler /MT)
endif()
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if (WIN32 AND CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 7)
    target_compile_options(${PROJECT_NAME} PRIVATE "/MT")
endif()

add_library(
    VideoCodecSDKUtils
    ${CODEC_SOURCES}
    ${CODEC_HDRS}
    ${NV_FFMPEG_HDRS}
    ${CODEC_CUDA_UTILS}
)

set(TC_VERSION_MAJOR 1)
set(TC_VERSION_MINOR 0)

generate_export_header(VideoCodecSDKUtils)

target_link_libraries(
    VideoCodecSDKUtils
    PUBLIC
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::npps
        CUDA::nppig
        CUDA::nppial
        CUDA::nppicc
        CUDA::nppidei
)


target_compile_features(VideoCodecSDKUtils PRIVATE cxx_std_17)
set_property(
    TARGET VideoCodecSDKUtils
    PROPERTY
        # required for shared Python modules in case library is build statically on Unix
        POSITION_INDEPENDENT_CODE
        ON
)

set(VIDEO_CODEC_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Interface"
    CACHE PATH "Path to Video Codec SDK interface headers"
)
target_include_directories(
    VideoCodecSDKUtils
    PUBLIC
        helper_classes/NvCodec
        helper_classes/Utils
        ${VIDEO_CODEC_INTERFACE_DIR} # TODO: check if it can be made private!
        ${CMAKE_CURRENT_BINARY_DIR}
)

if(NOT EXISTS ${VIDEO_CODEC_INTERFACE_DIR}/nvEncodeAPI.h)
    message(
        FATAL_ERROR
        "Could not find nvEncodeAPI.h! "
        "Please set VIDEO_CODEC_INTERFACE_DIR=\"${VIDEO_CODEC_INTERFACE_DIR}\" to a location that contains nvEncodeAPI.h!"
    )
endif()

if(UNIX)
    target_link_libraries(VideoCodecSDKUtils PUBLIC pthread)
endif(UNIX)

find_package(PkgConfig)
if(PKG_CONFIG_FOUND AND (NOT WIN32)) # pkgconfig works on Windows, but can not handle runtime dependencies
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
    target_link_libraries(VideoCodecSDKUtils PUBLIC PkgConfig::LIBAV)
    #try_compile(HAS_BSF ${PROJECT_BINARY_DIR} SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/has_bsf_compile_test.c LINK_LIBRARIES PkgConfig::LIBAV)
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
        target_link_libraries(VideoCodecSDKUtils PUBLIC ${${lib_name}_LIBRARY})
        target_include_directories(VideoCodecSDKUtils PUBLIC ${${lib_name}_INCLUDE_DIR})
    endmacro()
    link_av_component(VideoCodecSDKUtils avfilter)
    link_av_component(VideoCodecSDKUtils avformat)
    link_av_component(VideoCodecSDKUtils avcodec)
    link_av_component(VideoCodecSDKUtils swresample)
    link_av_component(VideoCodecSDKUtils avutil)
    find_path(
        BSF_INCLUDE_DIR
        NAMES "libavcodec/bsf.h"
        HINTS ${TC_FFMPEG_INCLUDE_DIR} "${FFMPEG_DIR}/include"
    )
    if(BSF_INCLUDE_DIR)
        set(HAS_BSF TRUE)
    else()
        set(HAS_BSF FALSE)
        message(WARNING "Could not find \"libavcodec/bsf.h\" while other ffmpeg includes could be found."
            "This likely means that your FFMPEG installation is old. Still trying to compile VPF!")
    endif()
endif()

if(HAS_BSF)
    target_compile_definitions(VideoCodecSDKUtils PUBLIC -DHAS_BSF=1)
endif()

option(USE_NVTX "Use NVTX for profiling" FALSE)

if(USE_NVTX)
    option(FETCHCONTENT_QUIET OFF)
    include(FetchContent)
    fetchcontent_declare(
    nvtx_cpp
    URL https://github.com/NVIDIA/NVTX/archive/v3.1.0.tar.gz # Oct 5 2022
    URL_HASH
    SHA256=dc4e4a227d04d3da46ad920dfee5f7599ac8d6b2ee1809c9067110fb1cc71ced
    SOURCE_SUBDIR c
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
    fetchcontent_makeavailable(nvtx_cpp)
endif()

if(USE_NVTX)
   target_link_libraries(VideoCodecSDKUtils PRIVATE nvtx3-cpp)
   target_compile_definitions(VideoCodecSDKUtils PRIVATE -DUSE_NVTX=1)
endif()

message(STATUS "HAS_BSF=${HAS_BSF}")
#install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${PROJECT_NAME}/bin)
