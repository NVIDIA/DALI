# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND DALI_INST_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/video_decoder_base.h")

if (BUILD_FFMPEG)
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/video_decoder_cpu.cc")
endif (BUILD_FFMPEG)

if (BUILD_NVDEC)
  list(APPEND DALI_OPERATOR_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/video_decoder_mixed.cc")
endif (BUILD_NVDEC)

set(DALI_OPERATOR_SRCS ${DALI_OPERATOR_SRCS} PARENT_SCOPE)
set(DALI_INST_HDRS ${DALI_INST_HDRS} PARENT_SCOPE)
