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
  void Decode() {
    auto decoded = DecodeImpl(image_type_, encoded_image_, length_);
    decoded_image_ = decoded.first;
    dims_ = decoded.second;
    decoded_ = true;
  }


  uint8_t *GetImage() {
    DALI_ENFORCE(decoded_, "Image not decoded. Run Decode()");
    return decoded_image_;
  }


  std::tuple<size_t, size_t, size_t> GetImageDims() {
    if (decoded_) {
      return dims_;
    }
    return PeekDims(encoded_image_, length_);
  }


  virtual ~Image() = default;

 protected:
  using ImageDims = std::tuple<size_t, size_t, size_t>; /// (height, width, channels)
  Image(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
          encoded_image_(encoded_buffer),
          length_(length),
          image_type_(image_type) {
  }


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

class GenericImage : public Image {
 public:
  GenericImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
          Image(encoded_buffer, length, image_type) {
  }


  inline std::pair<uint8_t *, ImageDims>
  DecodeImpl(DALIImageType image_type, const uint8_t *encoded_buffer, size_t length) override {

    // Decode image to tmp cv::Mat
    cv::Mat tmp = cv::imdecode(
            cv::Mat(1, length, CV_8UC1, (void *) (encoded_buffer)),
            IsColor(image_type) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    // if RGB needed, permute from BGR
    if (image_type == DALI_RGB) {
      cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
    }

    auto c = IsColor(image_type) ? 3 : 1;
    // Resize actual storage
    const int W = tmp.cols;
    const int H = tmp.rows;

    return std::make_pair(tmp.ptr(), std::make_tuple(H, W, c));

  }


  inline ImageDims PeekDims(const uint8_t *encoded_buffer, size_t length) override {
    throw std::runtime_error("Cannot peek dims for image of unknown format");
  }

};

class ImageFactory {
 public:
  static std::unique_ptr<Image>
  CreateImage(const uint8_t *encoded_image, size_t length, DALIImageType image_type) {
    return std::unique_ptr<Image>(new GenericImage(encoded_image, length, image_type));
  }
};


/*
class PngImage : public Image {
 private:

  // Assume chunk points to a 4-byte value
  int ReadIntFromPNG(const uint8_t *chunk) {
    // reverse the bytes, cast
    return (unsigned int)(chunk[0] << 24 | chunk[1] << 16 | chunk[2] << 8 | chunk[3]);
  }

  inline void AbstractDecode(const uint8_t * encoded_image, size_t encoded_image_length, DALIImageType image_type) override {

  }

  inline std::pair<size_t, size_t> PeekDims(const uint8_t * encoded_png, size_t encoded_image_length, DALIImageType image_type) override {
    DALI_ENFORCE(encoded_png);
    DALI_ENFORCE(encoded_image_length >= 16);

      // IHDR needs to be the first chunk
      const uint8_t *IHDR = encoded_png + 8;
      const uint8_t *png_dimens = IHDR;
      if (IHDR[4] != 'I' || IHDR[5] != 'H' || IHDR[6] != 'D' || IHDR[7] != 'R') {
        // no IHDR, older PNGs format
        png_dimens = encoded_png;
      }

      if (size >= png_dimens - png + 16) {
        // Layout:
        // 4 bytes: chunk size (should be 13 bytes for IHDR)
        // 4 bytes: Chunk Identifier (should be "IHDR")
        // 4 bytes: Width
        // 4 bytes: Height
        // 1 byte : Bit Depth
        // 1 byte : Color Type
        // 1 byte : Compression method
        // 1 byte : Filter method
        // 1 byte : Interlace method
        return std::make_pair(ReadIntFromPNG(png_dimens + 8), ReadIntFromPNG(png_dimens + 12));
      }

  }
};
*/



} // namespace dali

#endif //DALI_IMAGE_H
