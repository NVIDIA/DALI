// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_IMAGE_JPEG2K_H_
#define DALI_IMAGE_JPEG2K_H_

#include <utility>
#include <memory>

#include "dali/core/common.h"
#include "dali/image/generic_image.h"

namespace dali {

bool CheckIsJPEG2k(const uint8_t *jpeg2k, int size);

class Jpeg2kImage : public GenericImage {
 public:
  Jpeg2kImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type)
  : GenericImage(encoded_buffer, length, image_type) {}

  ~Jpeg2kImage() override = default;

 protected:
  Shape PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const override;
};

}  // namespace dali

#endif  // DALI_IMAGE_JPEG2K_H_
