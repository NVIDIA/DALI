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
NDLLError_t DecodeJPEGHost(const uint8 *jpeg, int size, bool color,
    int h, int w, uint8 *image);

// Note: The only jpeg decoder that we can make support jpeg parser caching is
// the hybrid one. To do this with CPU decode we'd need an idct, de-quant & yuv->rgb
// method to complete the decoder. Do we want to provide this api if it will be a
// bit of an anomaly?

} // namespace ndll

#endif // NDLL_IMAGE_JPEG_H_
