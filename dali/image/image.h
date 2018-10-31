// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_IMAGE_IMAGE_H_
#define DALI_IMAGE_IMAGE_H_

#include <memory>
#include <tuple>
#include <utility>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"


namespace dali {


class Image {
 public:
  /**
   * Perform image decoding. Actual implementation is defined
   * by DecodeImpl template method
   */
  DLL_PUBLIC void Decode();

  /**
   * Returns pointer to decoded image. Decode(...) has to be called
   * prior to calling this function
   */
  std::shared_ptr<uint8_t> GetImage() const;


  /**
   * Populates given data buffer with decoded image.
   * User is responsible for allocating `dst` buffer.
   */
  template<typename DstType>
  void GetImage(DstType *dst) const {
    DALI_ENFORCE(decoded_image_ && decoded_, "Image hasn't been decoded, call Decode(...)");
    std::memcpy(dst, decoded_image_.get(), dims_multiply() * sizeof(DstType));
  }


  /**
   * Returns image dimensions. If image hasn't been decoded,
   * reads the dims without decoding the image.
   * @return [height, width, depth (channels)]
   */
  DLL_PUBLIC std::tuple<size_t, size_t, size_t> GetImageDims() const;

  virtual ~Image() = default;
  DISABLE_COPY_MOVE_ASSIGN(Image);

 protected:
  using ImageDims = std::tuple<size_t, size_t, size_t>;  /// (height, width, channels)

  /**
   * Template method, that implements actual decoding.
   * @param image_type
   * @param encoded_buffer encoded image data
   * @param length length of the encoded buffer
   * @return [ptr to decoded image, ImageDims]
   */
  virtual std::pair<std::shared_ptr<uint8_t>, ImageDims>
  DecodeImpl(DALIImageType image_type, const uint8_t *encoded_buffer, size_t length) const = 0;

  /**
   * Template method. Reads image dimensions, without decoding the image
   * @param encoded_buffer encoded image data
   * @param length length of the encoded buffer
   * @return [height, width, depth]
   */
  virtual ImageDims PeekDims(const uint8_t *encoded_buffer, size_t length) const = 0;

  Image(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);

 private:
  inline size_t dims_multiply() const {
    // There's no elegant way in C++11
    return std::get<0>(dims_) * std::get<1>(dims_) * std::get<2>(dims_);
  }


  const uint8_t *encoded_image_;
  const size_t length_;
  const DALIImageType image_type_;
  bool decoded_ = false;
  ImageDims dims_;
  std::shared_ptr<uint8_t> decoded_image_ = nullptr;
};


}  // namespace dali

#endif  // DALI_IMAGE_IMAGE_H_
