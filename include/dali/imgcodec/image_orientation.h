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
  /// @brief Rotation angle, CCW, in degrees; only multiples of 90 are allowed.
  int rotate;
  /// @brief Mirror, horizontal
  bool flip_x;
  /// @brief Mirror, vertical
  bool flip_y;
};

enum class ExifOrientation : uint16_t {
  HORIZONTAL = 1,
  MIRROR_HORIZONTAL = 2,
  ROTATE_180 = 3,
  MIRROR_VERTICAL = 4,
  MIRROR_HORIZONTAL_ROTATE_270_CW = 5,
  ROTATE_90_CW = 6,
  MIRROR_HORIZONTAL_ROTATE_90_CW = 7,
  ROTATE_270_CW = 8
};

DLL_PUBLIC Orientation FromExifOrientation(ExifOrientation exif_orientation);

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_ORIENTATION_H_
