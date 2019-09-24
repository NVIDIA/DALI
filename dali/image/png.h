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

#ifndef DALI_IMAGE_PNG_H_
#define DALI_IMAGE_PNG_H_

#include "dali/image/generic_image.h"

namespace dali {

/**
 * PNG image decoding is performed using OpenCV, thus it's the same as Generic decoding
 */
class PngImage final : public GenericImage {
 public:
  PngImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type);

 private:
  Shape PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const override;
};

}  // namespace dali

#endif  // DALI_IMAGE_PNG_H_
