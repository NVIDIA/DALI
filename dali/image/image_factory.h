#ifndef DALI_IMAGE_FACTORY_H
#define DALI_IMAGE_FACTORY_H

#include "generic_image.h"
#include "png.h"

namespace dali {



class ImageFactory {
 public:
  static std::unique_ptr<Image>
  CreateImage(const uint8_t *encoded_image, size_t length, DALIImageType image_type=DALI_RGB) // TODO default argument
  {
    return std::unique_ptr<Image>(new GenericImage(encoded_image, length, image_type));
  }
};

} // namespace dali

#endif //DALI_IMAGE_FACTORY_H
