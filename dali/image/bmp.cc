#include "bmp.h"

namespace dali {

BmpImage::BmpImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        GenericImage(encoded_buffer, length, image_type) {
}


Image::ImageDims BmpImage::PeekDims(const uint8_t *bmp, size_t length) {

  DALI_ENFORCE(bmp);

  // https://en.wikipedia.org/wiki/BMP_file_format#DIB_header_(bitmap_information_header)
  unsigned header_size = bmp[14] | bmp[15] << 8 | bmp[16] << 16 | bmp[17] << 24;
  size_t h = 0;
  size_t w = 0;
  // BITMAPCOREHEADER: | 32u header | 16u width | 16u height | ...
  if (length >= 22 && header_size == 12) {
    w = (unsigned int) (bmp[18] | bmp[19] << 8) & 0xFFFF;
    h = (unsigned int) (bmp[20] | bmp[21] << 8) & 0xFFFF;
    // BITMAPINFOHEADER and later: | 32u header | 32s width | 32s height | ...
  } else if (length >= 26 && header_size >= 40) {
    w = static_cast<int>(bmp[18] | bmp[19] << 8 | bmp[20] << 16 | bmp[21] << 24);
    h = abs(static_cast<int>(bmp[22] | bmp[23] << 8 | bmp[24] << 16 | bmp[25] << 24));
  }
  return std::make_tuple(h, w, 0); // TODO fill channels
}


} // namespace dali