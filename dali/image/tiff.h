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

#ifndef DALI_IMAGE_TIFF_H_
#define DALI_IMAGE_TIFF_H_

#include <opencv2/opencv.hpp>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

/// TIFF header format documentation
/// http://www.fileformat.info/format/tiff/corion.htm

/**
 * @brief Returns 'true' if input compressed image is a tiff
 */
extern bool CheckIsTiff(const unsigned char *tiff);


/**
 * @brief Get dimensions of tiff encoded image
 */
extern DALIError_t GetTiffImageDims(const unsigned char *tiff, int size, int *h, int *w);


/**
 * @brief Decodes 'tiff' into the buffer pointed to by 'image'
 */
extern DALIError_t
DecodeTiffHost(const unsigned char *tiff, int size, DALIImageType image_type, Tensor<CPUBackend> *output);

}  // namespace dali

#endif //DALI_IMAGE_TIFF_H_
