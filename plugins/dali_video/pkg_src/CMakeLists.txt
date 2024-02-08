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
project(dali_video)

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

add_subdirectory(src/nvidia)
