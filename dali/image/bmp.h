#ifndef DALI_BMP_H
#define DALI_BMP_H

#include "generic_image.h"

namespace dali {

/**
 * BMP image decoding is performed using OpenCV, thus it's the same as Generic decoding
 */
class BmpImage : public GenericImage {
 public:
  BmpImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);

 private:
  Image::ImageDims PeekDims(const uint8_t *bmp, size_t length) override;
};

} // namespace dali

#endif //DALI_BMP_H
