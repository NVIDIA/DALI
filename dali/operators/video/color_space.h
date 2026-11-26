// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <stdint.h>
#include "dali/core/api_helper.h"

namespace dali {

enum VideoColorSpaceConversionType {
    // ITU-R BT.601 matrix, limited range
    VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB,
    // ITU-R BT.601 matrix, full range
    VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_FULL_RANGE,
    VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_UPSAMPLE,
    // ITU-R BT.709 matrix, limited range
    VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_BT709,
    // ITU-R BT.709 matrix, full range
    VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB_BT709_FULL_RANGE,
};

/**
 * @brief Converts an NV12 frame (`height` rows of luma, then the interleaved chroma) to RGB or
 *        upsampled YCbCr.
 *
 * Odd sizes are as NVDEC lays them out: the chroma plane starts after the luma rows rounded up to
 * an even number, and it has ceil(width / 2) samples per row and ceil(height / 2) rows.
 *
 * @param chroma_height The rows of the chroma plane that hold data, -1 for all of them
 *                      (ceil(height / 2)). The ones below are taken as copies of the last one.
 */
DLL_PUBLIC void ConvertVideoColorSpace(uint8_t *out, int out_pitch, const uint8_t* yuv, int yuv_pitch, int height, int width,
                                       VideoColorSpaceConversionType conversion_type,
                                       bool normalized_range, cudaStream_t stream,
                                       int chroma_height = -1);

DLL_PUBLIC void ConvertVideoColorSpace(float *out, int out_pitch, const uint8_t* yuv, int yuv_pitch, int height, int width,
                                       VideoColorSpaceConversionType conversion_type,
                                       bool normalized_range, cudaStream_t stream,
                                       int chroma_height = -1);

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_COLOR_SPACE_H_
