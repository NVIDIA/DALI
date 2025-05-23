// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "dali/operators/image/crop/random_crop_attr.h"

namespace dali {

DALI_SCHEMA(RandomCropAttr)
    .DocStr(R"code(Random Crop attributes placeholder)code")
    .AddOptionalArg("random_aspect_ratio",
      R"code(Range from which to choose random aspect ratio (width/height).)code",
      std::vector<float>{3./4., 4./3.})
  .AddOptionalArg("random_area",
      R"code(Range from which to choose random area fraction ``A``.

The cropped image's area will be equal to ``A`` * original image's area.)code",
      std::vector<float>{0.08, 1.0})
  .AddOptionalArg("num_attempts",
      R"code(Maximum number of attempts used to choose random area and aspect ratio.)code",
      10)
  .AddRandomSeedArg();

}  // namespace dali
