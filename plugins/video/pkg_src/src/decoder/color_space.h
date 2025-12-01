// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PLUGINS_VIDEO_PKG_SRC_SRC_DECODER_COLOR_SPACE_H_
#define PLUGINS_VIDEO_PKG_SRC_SRC_DECODER_COLOR_SPACE_H_

#include <stdint.h>

void yuv_to_rgb(
    uint8_t *yuv,
    int yuv_pitch,
    uint8_t *rgb,
    int rgb_pitch,
    int width,
    int height,
    bool full_range,
    cudaStream_t stream);

#endif  // PLUGINS_VIDEO_PKG_SRC_SRC_DECODER_COLOR_SPACE_H_
