// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <functional>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/util/crop_window.h"
#include "dali/core/tensor_shape.h"

namespace dali {

class Image {
 public:
  using Shape = TensorShape<3>;

  /**
   * Perform image decoding. Actual implementation is defined
   * by DecodeImpl template method
   */
  DLL_PUBLIC void Decode();

  /**
   * Returns pointer to decoded image. Decode(...) has to be called
   * prior to calling this function
   */
  DLL_PUBLIC std::shared_ptr<uint8_t> GetImage() const;


  /**
   * Populates given data buffer with decoded image.
   * User is responsible for allocating `dst` buffer.
   */
  template<typename DstType>
  void GetImage(DstType *dst) const {
    DALI_ENFORCE(decoded_image_ && decoded_, "Image hasn't been decoded, call Decode(...)");
    std::memcpy(dst, decoded_image_.get(), volume(shape_) * sizeof(DstType));
  }


  /**
   * Returns the decoded image dimensions.
   * It will fail if image hasn't been decoded,
   * @return [height, width, depth channels]
   */
  DLL_PUBLIC Shape GetShape() const;

  /**
   * Reads the original image dimensions without decoding the image.
   * @return [height, width, channels]
   */
  DLL_PUBLIC Shape PeekShape() const;

 /**
  * Sets crop window generator
  */
  inline void SetCropWindowGenerator(const CropWindowGenerator& crop_window_generator) {
    crop_window_generator_ = crop_window_generator;
  }

  inline void SetCropWindow(const CropWindow& crop_window) {
    if (!crop_window)
      return;
    crop_window_generator_ = [crop_window](const TensorShape<>& shape,
                                           const TensorLayout& shape_layout) {
      DALI_ENFORCE(shape_layout == "HW",
        make_string("Unexpected input shape layout: ", shape_layout, " vs HW"));
      crop_window.EnforceInRange(shape);
      return crop_window;
    };
  }

  inline void SetUseFastIdct(bool use_fast_idct) {
    use_fast_idct_ = use_fast_idct;
  }

  inline bool UseFastIdct() const {
    return use_fast_idct_;
  }

  virtual ~Image() = default;
  DISABLE_COPY_MOVE_ASSIGN(Image);

 protected:
  /**
   * Template method, that implements actual decoding.
   * @param image_type
   * @param encoded_buffer encoded image data
   * @param length length of the encoded buffer
   * @return [ptr to decoded image, Shape]
   */
  virtual std::pair<std::shared_ptr<uint8_t>, Shape>
  DecodeImpl(DALIImageType image_type, const uint8_t *encoded_buffer, size_t length) const = 0;

  /**
   * Template method. Reads image dimensions, without decoding the image
   * @param encoded_buffer encoded image data
   * @param length length of the encoded buffer
   * @return [height, width, depth]
   */
  virtual Shape PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const = 0;

  Image(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);

  /**
   * Gets random crop generator
   */
  inline CropWindowGenerator GetCropWindowGenerator() const {
    return crop_window_generator_;
  }

 private:
  const uint8_t *encoded_image_;
  const size_t length_;
  const DALIImageType image_type_;
  bool decoded_ = false;
  bool use_fast_idct_ = false;
  Shape shape_;
  CropWindowGenerator crop_window_generator_;
  std::shared_ptr<uint8_t> decoded_image_ = nullptr;
};


}  // namespace dali

#endif  // DALI_IMAGE_IMAGE_H_
