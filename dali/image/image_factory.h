#ifndef DALI_IMAGE_FACTORY_H
#define DALI_IMAGE_FACTORY_H

#include "image.h"

namespace dali {

class ImageFactory {
 public:
  DLL_PUBLIC static std::unique_ptr<Image>
  CreateImage(const uint8_t *encoded_image, size_t length, DALIImageType image_type=DALI_RGB); // TODO default argument
};

} // namespace dali

#endif //DALI_IMAGE_FACTORY_H
