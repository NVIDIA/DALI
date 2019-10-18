// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_IMAGE_TRANSFORM_H_
#define DALI_IMAGE_TRANSFORM_H_

#include <string>
#include <utility>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

/**
 * @brief Performs resize, crop, & random mirror on the input image on the CPU. Input
 * data is assumed to be stored in HWC layout in memory.
 *
 * This method takes in an optional 'workspace' buffer. Because we perform the resize
 * and mirror in separate steps, and intermediate buffer is needed to store the
 * resutls of the resize before mirroring into the output buffer. Rather than
 * always allocate memory inside the function, we provide the option to pass in
 * this temporary workspace pointer to avoid extra memory allocation. The size
 * of the memory pointed to by 'workspace' should be rsz_h*rsz_w*C bytes
 *
 * Note: We leave the calculate of the resize dimensions & the decision of whether
 * to mirror the image or not external to the function. With the GPU version of
 * this function, these params will need to have been calculated before-hand
 * and, in the case of a batched call, copied to the device. Separating these
 * parameters from this function will make the API consistent across the CPU
 * & GPU versions.
 */
DLL_PUBLIC DALIError_t ResizeCropMirrorHost(const uint8 *img, int H, int W, int C,
    int rsz_h, int rsz_w, const std::pair<int, int> &crop, int crop_h, int crop_w,
    int mirror, uint8 *out_img, DALIInterpType type = DALI_INTERP_LINEAR,
    uint8 *workspace = nullptr);

/**
 * @brief Performs resize, crop, & random mirror on the input image on the CPU. Input
 * data is assumed to be stored in HWC layout in memory.
 *
 * 'Fast' ResizeCropMirrorHost does not perform the full image resize. Instead, it
 * takes advantage of the fact that we are going to crop, and backprojects the crop
 * into the input image. We then resize the backprojected crop region to the crop
 * dimensions (crop_w/crop_h), avoiding a significant amount of work on data that
 * would have been cropped away immediately.
 *
 * This method takes in an optional 'workspace' buffer. Because we perform the resize
 * and mirror in separate steps, and intermediate buffer is needed to store the
 * resutls of the resize before mirroring into the output buffer. Rather than
 * always allocate memory inside the function, we provide the option to pass in
 * this temporary workspace pointer to avoid extra memory allocation. The size
 * of the memory pointed to by 'workspace' should be crop_h*crop_w*C bytes
 */
DLL_PUBLIC DALIError_t FastResizeCropMirrorHost(const uint8 *img, int H, int W, int C,
    int rsz_h, int rsz_w, const std::pair<int, int> &crop, int crop_h, int crop_w,
    int mirror, uint8 *out_img, DALIInterpType type = DALI_INTERP_LINEAR,
    uint8 *workspace = nullptr);

DLL_PUBLIC void CheckParam(const Tensor<CPUBackend> &input, const std::string &pOperator);

DLL_PUBLIC DALIError_t MakeColorTransformation(const uint8 *img, int H, int W, int C,
                                               const float *matrix, uint8 *out_img);

}  // namespace dali

#endif  // DALI_IMAGE_TRANSFORM_H_
