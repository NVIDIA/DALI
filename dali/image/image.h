#ifndef DALI_IMAGE_H
#define DALI_IMAGE_H

#include "dali/common.h"
#include <opencv2/opencv.hpp>
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"
#include <memory>


namespace dali {


class Image {
 public:
  void Decode();

  uint8_t *GetImage();

  std::tuple<size_t, size_t, size_t> GetImageDims();

  virtual ~Image() = default;

 protected:
  using ImageDims = std::tuple<size_t, size_t, size_t>; /// (height, width, channels)
  Image(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);

 private:
  virtual std::pair<uint8_t *, ImageDims>
  DecodeImpl(DALIImageType image_type, const uint8_t *encoded_buffer, size_t length) = 0;

  virtual ImageDims PeekDims(const uint8_t *encoded_buffer, size_t length) = 0;

  const uint8_t *encoded_image_;
  const size_t length_;
  const DALIImageType image_type_;
  bool decoded_ = false;
  ImageDims dims_;
  uint8_t *decoded_image_ = nullptr;

};





} // namespace dali

#endif //DALI_IMAGE_H
