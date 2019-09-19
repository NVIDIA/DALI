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

#ifndef DALI_IMAGE_TIFF_LIBTIFF_IMPL_H_
#define DALI_IMAGE_TIFF_LIBTIFF_IMPL_H_

#include <tiffio.h>
#include <utility>
#include <memory>
#include "dali/core/span.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/util/crop_window.h"

namespace dali {

class LibtiffImpl {
 private:
  span<const uint8_t> buf_;
  size_t buf_pos_;
  std::unique_ptr<TIFF, void (*)(TIFF *)> tif_ = {nullptr, &TIFFClose};

  kernels::TensorShape<3> shape_ = {0, 0, 0};
  bool is_tiled_ = false;
  uint16_t bit_depth_ = 8;
  uint16_t orientation_ = ORIENTATION_TOPLEFT;

 public:
  explicit LibtiffImpl(span<const uint8_t> buf);

  kernels::TensorShape<3> Dims() const;

  bool CanDecode(DALIImageType image_type) const;

  std::pair<std::shared_ptr<uint8_t>, kernels::TensorShape<3>>
  Decode(DALIImageType image_type, CropWindowGenerator crop_window_generator) const;
};

}  // namespace dali

#endif  // DALI_IMAGE_TIFF_LIBTIFF_IMPL_H_
