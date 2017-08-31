#ifndef NDLL_IMAGE_JPEG_H_
#define NDLL_IMAGE_JPEG_H_

namespace ndll {

/**
 * @brief Returns 'true' if input compressed image is a jpeg
 */
bool CheckIsJPEG(const uint8 *jpeg, int size) {
  if ((jpeg[0] == 255) && jpeg[1] == 216) {
    return true;
  }
  return false;
}

/**
 * @brief Gets the dimensions of the jpeg encoded image
 */
void GetJPEGImageDims(const uint8 *jpeg, int size, int *h, int *w) {
  // Note: For now we use turbo-jpeg header decompression. This
  // may be more expensive than using the hacky method MXNet has.
  // Worth benchmarking this at a later point
}

/**
 * @brief Decodes `jpeg` into the the buffer pointed to by `image`
 */
void DecodeJPEGHost(const uint8 *jpeg, int size, bool color, uint8 *image) {
  
}

// Note: The only jpeg decoder that we can make support jpeg parser caching is
// the hybrid one. To do this with CPU decode we'd need an idct, de-quant & yuv->rgb
// method to complete the decoder. Do we want to provide this api if it will be a
// bit of an anomaly?

} // namespace ndll

#endif // NDLL_IMAGE_JPEG_H_
