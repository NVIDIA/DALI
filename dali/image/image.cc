#include "image.h"

namespace dali {


Image::Image(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        encoded_image_(encoded_buffer),
        length_(length),
        image_type_(image_type) {
}


void Image::Decode() {
  auto decoded = DecodeImpl(image_type_, encoded_image_, length_);
  decoded_image_ = decoded.first;
  dims_ = decoded.second;
  decoded_ = true;
}


std::shared_ptr<uint8_t> Image::GetImage() {
  DALI_ENFORCE(decoded_, "Image not decoded. Run Decode()");
  return decoded_image_;
}


std::tuple<size_t, size_t, size_t> Image::GetImageDims() {
  if (decoded_) {
    return dims_;
  }
  return PeekDims(encoded_image_, length_);
}

size_t Image::dims_multiply() {
  // There's no elegant way in C++11
  return std::get<0>(dims_) * std::get<1>(dims_) * std::get<2>(dims_);
}


} // namespace dali