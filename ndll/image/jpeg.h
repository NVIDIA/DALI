// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_IMAGE_JPEG_H_
#define NDLL_IMAGE_JPEG_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Returns 'true' if input compressed image is a jpeg
 */
bool CheckIsJPEG(const uint8 *jpeg, int size);

/**
 * @brief Gets the dimensions of the jpeg encoded image
 */
NDLLError_t GetJPEGImageDims(const uint8 *jpeg, int size, int *h, int *w);

/**
 * @brief Decodes `jpeg` into the the buffer pointed to by `image`
 */
NDLLError_t DecodeJPEGHost(const uint8 *jpeg, int size,
    NDLLImageType image_type, int h, int w, uint8 *image);

}  // namespace ndll

#endif  // NDLL_IMAGE_JPEG_H_
