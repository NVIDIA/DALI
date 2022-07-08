// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/util/exif.h"

namespace dali {
namespace imgcodec {

constexpr int ORIENTATION_HORIZONTAL = 1;
constexpr int ORIENTATION_MIRROR_HORIZONTAL = 2;
constexpr int ORIENTATION_ROTATE_180 = 3;
constexpr int ORIENTATION_MIRROR_VERTICAL = 4;
constexpr int ORIENTATION_MIRROR_HORIZONTAL_ROTATE_270 = 5;
constexpr int ORIENTATION_ROTATE_90 = 6;
constexpr int ORIENTATION_MIRROR_HORIZONTAL_ROTATE_90 = 7;
constexpr int ORIENTATION_ROTATE_270 = 8;

void SetOrientation(ImageInfo &info, uint16_t exif_orientation) {
    switch (exif_orientation) {
          case ORIENTATION_HORIZONTAL:
            info.orientation = {0, false, false};
            break;
          case ORIENTATION_MIRROR_HORIZONTAL:
            info.orientation = {0, true, false};
            break;
          case ORIENTATION_ROTATE_180:
            info.orientation = {180, false, false};
            break;
          case ORIENTATION_MIRROR_VERTICAL:
            info.orientation = {0, false, true};
            break;
          case ORIENTATION_MIRROR_HORIZONTAL_ROTATE_270:
            info.orientation = {270, true, false};
            break;
          case ORIENTATION_ROTATE_90:
            info.orientation = {90, false, false};
            break;
          case ORIENTATION_MIRROR_HORIZONTAL_ROTATE_90:
            info.orientation = {90, true, false};
            break;
          case ORIENTATION_ROTATE_270:
            info.orientation = {270, false, false};
            break;
          default:
            DALI_FAIL("Couldn't read TIFF image orientation.");
        }
}

}  // namespace imgcodec
}  // namespace dali
