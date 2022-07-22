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
#include <string>

namespace dali {
namespace imgcodec {

Orientation FromExifOrientation(ExifOrientation exif_orientation) {
  switch (exif_orientation) {
    case ExifOrientation::HORIZONTAL:
      return {0, false, false};
    case ExifOrientation::MIRROR_HORIZONTAL:
      return {0, true, false};
    case ExifOrientation::ROTATE_180:
      return {180, false, false};
    case ExifOrientation::MIRROR_VERTICAL:
      return {0, false, true};
    case ExifOrientation::MIRROR_HORIZONTAL_ROTATE_270_CW:
      return {90, true, false};  // 270 CW = 90 CCW
    case ExifOrientation::ROTATE_90_CW:
      return {270, false, false};  // 90 CW = 270 CCW
    case ExifOrientation::MIRROR_HORIZONTAL_ROTATE_90_CW:
      return {270, true, false};  // 90 CW = 270 CCW
    case ExifOrientation::ROTATE_270_CW:
      return {90, false, false};  // 270 CW = 90 CCW
    default:
      DALI_FAIL("Couldn't read image orientation.");
  }
}

}  // namespace imgcodec
}  // namespace dali
