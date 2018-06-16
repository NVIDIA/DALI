// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_IMAGE_PNG_H_
#define DALI_IMAGE_PNG_H_

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

/**
 * @brief Returns 'true' if input compressed image is a png
 */
bool CheckIsPNG(const uint8 *png, int size);

/**
 * @brief Get dimensions of png encoded image
 */
DALIError_t GetPNGImageDims(const uint8 *png, int size, int *h, int *w);

/**
 * @brief Decodes 'png' into the buffer pointed to by 'image'
 */
DALIError_t DecodePNGHost(const uint8 *png, int size,
    DALIImageType image_type, Tensor<CPUBackend>* output);

}  // namespace dali

#endif  // DALI_IMAGE_PNG_H_
