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
project(dali_plugin_video LANGUAGES CUDA CXX C)

include(common.cmake)

include(FetchContent)
FetchContent_Declare(
    pynvvideocodec
    URL      https://api.ngc.nvidia.com/v2/resources/nvidia/py_nvvideocodec/versions/0.0.9/zip
    URL_HASH SHA512=b73d06f2239bc50a6a9971949a3579df56d8a6c223d64f5fcbebb3531b9adff2e9f25d23b1922bc42616e65fc850d60c085cadfd7daecbfc562f9959995f28a5
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Populate(pynvvideocodec)
message(STATUS "pynvvideocodec_SOURCE_DIR=${pynvvideocodec_SOURCE_DIR}")
include_directories(SYSTEM "${pynvvideocodec_SOURCE_DIR}/src/")
add_subdirectory("${pynvvideocodec_SOURCE_DIR}/src/VideoCodecSDKUtils/")

set(VIDEO_PLUGIN_SOURCES
    src/decoder/video_decoder_mixed.cc
)

set(PLUGIN_NAME "video")

add_library(dali_${PLUGIN_NAME} SHARED ${VIDEO_PLUGIN_SOURCES})
set_target_properties(dali_${PLUGIN_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(dali_${PLUGIN_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
target_include_directories(dali_${PLUGIN_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_directories(dali_${PLUGIN_NAME} PUBLIC ${DALI_LIB_DIR})
target_link_libraries(dali_${PLUGIN_NAME} PUBLIC dali dali_core dali_kernels VideoCodecSDKUtils)

include(GNUInstallDirs)

install(TARGETS dali_${PLUGIN_NAME}
    RUNTIME DESTINATION nvidia/dali/plugin/${PLUGIN_NAME}
    LIBRARY DESTINATION nvidia/dali/plugin/${PLUGIN_NAME}
)
