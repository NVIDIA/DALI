// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_IMAGE_PNG_H_
#define NDLL_IMAGE_PNG_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

/**
 * @brief Returns 'true' if input compressed image is a png
 */
bool CheckIsPNG(const uint8 *png, int size);

/**
 * @brief Get dimensions of png encoded image
 */
NDLLError_t GetPNGImageDims(const uint8 *png, int size, int *h, int *w);

/**
 * @brief Decodes 'png' into the buffer pointed to by 'image'
 */
NDLLError_t DecodePNGHost(const uint8 *png, int size,
    NDLLImageType image_type, Tensor<CPUBackend>* output);

}  // namespace ndll

#endif  // NDLL_IMAGE_PNG_H_
