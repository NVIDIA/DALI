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

#include "dali/imgcodec/util/output_shape.h"

namespace dali {
namespace imgcodec {

ROI PreOrientationRoi(const ImageInfo &info, ROI roi) {
  bool swap_xy = info.orientation.rotate % 180 == 90;
  bool flip_x = info.orientation.rotate == 180 || info.orientation.rotate == 270;
  bool flip_y = info.orientation.rotate == 90 || info.orientation.rotate == 180;
  flip_x ^= info.orientation.flip_x, flip_y ^= info.orientation.flip_y;

  auto flip_axis = [&](int idx) {
    std::swap(roi.begin[idx], roi.end[idx]);
    roi.begin[idx] = info.shape[idx ^ swap_xy] - roi.begin[idx];
    roi.end[idx] = info.shape[idx ^ swap_xy] - roi.end[idx];
  };

  // Performing operations in reverse order, to cancel them out

  if (flip_x) flip_axis(1);
  if (flip_y) flip_axis(0);

  if (swap_xy) {
    std::swap(roi.begin[0], roi.begin[1]);
    std::swap(roi.end[0], roi.end[1]);
  }

  return roi;
}

}  // namespace imgcodec
}  // namespace dali
