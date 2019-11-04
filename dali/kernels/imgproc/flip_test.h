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

#ifndef DALI_KERNELS_IMGPROC_FLIP_TEST_H_
#define DALI_KERNELS_IMGPROC_FLIP_TEST_H_

namespace dali {
namespace kernels {

template <typename T>
bool is_flipped(const T* lhs, const T* rhs,
                size_t seq_length, size_t depth, size_t height, size_t width,
                size_t channels, bool flip_z, bool flip_y, bool flip_x) {
  for (size_t f = 0; f < seq_length; ++f) {
    for (size_t z = 0; z < depth; ++z) {
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          auto rhs_x = flip_x ? width - x - 1 : x;
          auto rhs_y = flip_y ? height - y - 1 : y;
          auto rhs_z = flip_z ? depth - z - 1 : z;
          for (size_t c = 0; c < channels; ++c) {
            if (lhs[channels * (width * (height * (depth * f + z) + y) + x) + c] !=
                rhs[channels * (width * (height * (depth * f + rhs_z) + rhs_y) + rhs_x) + c]) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_FLIP_TEST_H_
