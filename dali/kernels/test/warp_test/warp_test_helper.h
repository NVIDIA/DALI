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

#ifndef DALI_KERNELS_TEST_WARP_TEST_WARP_TEST_HELPER_H_
#define DALI_KERNELS_TEST_WARP_TEST_WARP_TEST_HELPER_H_

#include "dali/kernels/imgproc/warp/affine.h"

namespace dali {
namespace kernels {

/// @brief Apply correction of pixel centers and convert the mapping to
///        OpenCV matrix type.
inline cv::Matx<float, 2, 3> AffineToCV(const AffineMapping2D &mapping) {
  vec2 translation = mapping({0.5f, 0.5f}) - vec2(0.5f, 0.5f);
  mat2x3 tmp = mapping.transform;
  tmp.set_col(2, translation);

  cv::Matx<float, 2, 3> cv_transform;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
      cv_transform(i, j) = tmp(i, j);
  return cv_transform;
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_TEST_WARP_TEST_WARP_TEST_HELPER_H_
