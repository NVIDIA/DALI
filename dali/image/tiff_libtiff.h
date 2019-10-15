// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_IMAGE_TIFF_LIBTIFF_H_
#define DALI_IMAGE_TIFF_LIBTIFF_H_

#include <tiffio.h>
#include <utility>
#include <memory>
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/image/generic_image.h"

namespace dali {

class TiffImage_Libtiff : public GenericImage {
 public:
  TiffImage_Libtiff(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);
  bool CanDecode(DALIImageType image_type) const;

 protected:
  std::pair<std::shared_ptr<uint8_t>, Image::Shape>
  DecodeImpl(DALIImageType image_type, const uint8_t *encoded_buffer, size_t length) const override;

  Image::Shape PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const override;

 private:
  span<const uint8_t> buf_;
  size_t buf_pos_;
  std::unique_ptr<TIFF, void (*)(TIFF *)> tif_ = {nullptr, &TIFFClose};

  TensorShape<3> shape_ = {0, 0, 0};
  bool is_tiled_ = false;
  uint16_t bit_depth_ = 8;
  uint16_t orientation_ = ORIENTATION_TOPLEFT;
  uint32_t rows_per_strip_ = 0xFFFFFFFF;
  uint16_t compression_ = COMPRESSION_NONE;
};

}  // namespace dali

#endif  // DALI_IMAGE_TIFF_LIBTIFF_H_
