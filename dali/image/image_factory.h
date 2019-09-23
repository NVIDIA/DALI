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

#ifndef DALI_IMAGE_IMAGE_FACTORY_H_
#define DALI_IMAGE_IMAGE_FACTORY_H_

#include <memory>
#include "dali/image/image.h"

namespace dali {

class ImageFactory {
 public:
  DLL_PUBLIC static std::unique_ptr<Image>
  CreateImage(const uint8_t *encoded_image, size_t length, DALIImageType image_type);
};

}  // namespace dali

#endif  // DALI_IMAGE_IMAGE_FACTORY_H_
