// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_NVJPEG_PERMUTE_LAYOUT_H_
#define DALI_OPERATORS_DECODER_NVJPEG_PERMUTE_LAYOUT_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include "dali/pipeline/data/types.h"

namespace dali {

/**
 * @brief Permute data in planar layout to interleaved layout.
 * @param npixels - volume of a single component in number of pixels
 * @param comp_count - number of components (channels)
 */
template <typename Output, typename Input>
void PlanarToInterleaved(Output *output, const Input *input, int64_t npixels, int64_t comp_count,
                         DALIImageType out_img_type, DALIDataType pixel_type, cudaStream_t stream);

/**
 * @brief Permute data in RGB planar layout to grayscale.
 * @param npixels - volume of a single component in number of pixels
 */
template <typename Output, typename Input>
void PlanarRGBToGray(Output *output, const Input *input,
                     int64_t npixels, DALIDataType pixel_type, cudaStream_t stream);

/**
 * @brief Convert interleaved RGB data to interleaved YCbCr .
 */
template <typename Output, typename Input>
void Convert_RGB_to_YCbCr(Output *output, const Input *input, int64_t npixels, cudaStream_t stream);


}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_NVJPEG_PERMUTE_LAYOUT_H_
