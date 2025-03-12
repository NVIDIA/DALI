// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_VIDEO_COLOR_SPACE_H_
#define DALI_OPERATORS_VIDEO_COLOR_SPACE_H_

#include <stdint.h>

namespace dali {

enum VideoColorSpaceConversionType {
    VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB,
    VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_FULL_RANGE,
    VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_UPSAMPLE,
};

void VideoColorSpaceConversion(uint8_t *out, int out_pitch, const uint8_t* yuv, int yuv_pitch, int height, int width,
                               VideoColorSpaceConversionType conversion_type, bool normalized_range,
                               cudaStream_t stream);

void VideoColorSpaceConversion(float *out, int out_pitch, const uint8_t* yuv, int yuv_pitch, int height, int width,
                               VideoColorSpaceConversionType conversion_type, bool normalized_range,
                               cudaStream_t stream);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_COLOR_SPACE_H_
