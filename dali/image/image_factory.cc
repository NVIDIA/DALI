#include "image_factory.h"

namespace dali {

namespace {

bool CheckIsJPEG(const uint8 *jpeg, int) {
  if ((jpeg[0] == 255) && (jpeg[1] == 216)) {
    return true;
  }
  return false;
}

bool CheckIsPNG(const uint8_t *png, int size) {
  DALI_ASSERT(png);
  // first bytes should be: 89 50 4E 47 0D 0A 1A 0A (hex)
  //                        137 80 78 71 13 10 26 10 (decimal)
  return (size >= 8 && png[0] == 137 && png[1] == 80 && png[2] == 78 && png[3] == 71 &&
          png[4] == 13 && png[5] == 10 && png[6] == 26 && png[7] == 10);
}

bool CheckIsGIF(const uint8_t *gif, int size) {
  DALI_ASSERT(gif);
  return (size >= 10 && gif[0] == 'G' && gif[1] == 'I' && gif[2] == 'F' && gif[3] == '8' &&
          (gif[4] == '7' || gif[4] == '9') && gif[5] == 'a');
}

bool CheckIsBMP(const uint8_t *bmp, int size) {
  return (size > 2 && bmp[0] == 'B' && bmp[1] == 'M');
}

} // namespace

std::unique_ptr<Image>
ImageFactory::CreateImage(const uint8_t *encoded_image, size_t length, DALIImageType image_type) {
  return std::unique_ptr<Image>(new GenericImage(encoded_image, length, image_type));
}

} // namespace dali