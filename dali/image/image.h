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
  /**
   * Perform image decoding. Actual implementation is defined
   * by DecodeImpl template method
   */
  void Decode();

  /**
   * Returns pointer to decoded image. Decode(...) has to be called
   * prior to calling this function
   */
  std::shared_ptr<uint8_t> GetImage();


  /**
   * Populates given data buffer with decoded image.
   * User is responsible for allocating `dst` buffer.
   */
  template<typename DstType>
  void GetImage(DstType *dst) {
    DALI_ENFORCE(decoded_image_ && decoded_, "Image hasn't been decoded, call Decode(...)");
    std::memcpy(dst, decoded_image_.get(), dims_multiply() * sizeof(DstType));
  }


  /**
   * Returns image dimensions. If image hasn't been decoded,
   * reads the dims without decoding the image.
   * @return [height, width, depth (channels)]
   */
  std::tuple<size_t, size_t, size_t> GetImageDims();

  virtual ~Image() = default;

 protected:
  using ImageDims = std::tuple<size_t, size_t, size_t>; /// (height, width, channels)

  /**
   * Template method, that implements actual decoding.
   * @param image_type
   * @param encoded_buffer encoded image data
   * @param length length of the encoded buffer
   * @return [ptr to decoded image, ImageDims]
   */
  virtual std::pair<std::shared_ptr<uint8_t>, ImageDims>
  DecodeImpl(DALIImageType image_type, const uint8_t *encoded_buffer, size_t length) = 0; //TODO shared_ptr

  /**
   * Template method. Reads image dimensions, without decoding the image
   * @param encoded_buffer encoded image data
   * @param length length of the encoded buffer
   * @return [height, width, depth]
   */
  virtual ImageDims PeekDims(const uint8_t *encoded_buffer, size_t length) = 0;
  Image(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);

 private:

  size_t dims_multiply();

  const uint8_t *encoded_image_;
  const size_t length_;
  const DALIImageType image_type_;
  bool decoded_ = false;
  ImageDims dims_;
  std::shared_ptr<uint8_t> decoded_image_ = nullptr;

};


} // namespace dali

#endif //DALI_IMAGE_H
