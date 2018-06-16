// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_IMAGE_JPEG_H_
#define DALI_IMAGE_JPEG_H_

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

/**
 * @brief Returns 'true' if input compressed image is a jpeg
 */
bool CheckIsJPEG(const uint8 *jpeg, int size);

/**
 * @brief Gets the dimensions of the jpeg encoded image
 */
DALIError_t GetJPEGImageDims(const uint8 *jpeg, int size, int *h, int *w);

/**
 * @brief Decodes `jpeg` into the the buffer pointed to by `image`
 */
DALIError_t DecodeJPEGHost(const uint8 *jpeg, int size,
    DALIImageType image_type, Tensor<CPUBackend>* output);

}  // namespace dali

#endif  // DALI_IMAGE_JPEG_H_
