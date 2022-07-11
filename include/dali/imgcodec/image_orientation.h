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

#ifndef DALI_IMGCODEC_IMAGE_ORIENTATION_H_
#define DALI_IMGCODEC_IMAGE_ORIENTATION_H_

namespace dali {
namespace imgcodec {

/**
 * @brief The transform that needs to be applied to rectify an image.
 * 
 * The operations are applied in the order in which they are declared.
 */
struct Orientation {
  /// @brief Rotation, CCW, in multiples of 90 degrees
  int rotate;
  /// @brief Mirror, horizontal
  bool flip_x;
  /// @brief Mirror, vertical
  bool flip_y;
};

enum ExifOrientation : uint16_t {
  ORIENTATION_HORIZONTAL = 1,
  ORIENTATION_MIRROR_HORIZONTAL = 2,
  ORIENTATION_ROTATE_180 = 3,
  ORIENTATION_MIRROR_VERTICAL = 4,
  ORIENTATION_MIRROR_HORIZONTAL_ROTATE_270_CW = 5,
  ORIENTATION_ROTATE_90_CW = 6,
  ORIENTATION_MIRROR_HORIZONTAL_ROTATE_90_CW = 7,
  ORIENTATION_ROTATE_270_CW = 8
};

Orientation FromExifOrientation(ExifOrientation exif_orientation);

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_ORIENTATION_H_
