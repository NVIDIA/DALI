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

option(FFMPEG_DIR "setting FFMPEG dir" FALSE)
set(FFMPEG_DIR $ENV{FFMPEG_DIR})

find_package(CUDAToolkit 11.2 REQUIRED)

find_path(
    FFMPEG_DIR
    NAMES "include/libavcodec/avcodec.h" 
)

set(dali_video_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(VideoCodecSDKUtils)
add_subdirectory(dali_video)
