// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_KERNELS_COMMON_SPLIT_SHAPE_H_
#define DALI_KERNELS_COMMON_SPLIT_SHAPE_H_

namespace dali {
namespace kernels {

/**
 * @brief Utility to divide a bigger shape into smaller blocks, given a desired minimum number
 *        of blocks and a minimum practical block size.
 *        The algorithm starts splitting from the outermost dimension until either the number of
 *        blocks reaches the desired minimum, or until the remaining volume is under a given threshold.
 * @remarks The algorithm makes an effort to keep a good balance of block sizes, which might result in
 *          a higher number of blocks than the minimum requested.
 * @param split_factor Output argument used to represent split factors for each dimension.
 * @param in_shape Input shape
 * @param min_nblocks Desired minimum number of blocks
 * @param min_sz Minimum practical block size
 */
template <typename SplitFactor, typename Shape>
void split_shape(SplitFactor& split_factor, const Shape& in_shape, int min_nblocks = 8,
                 int min_sz = (16 << 10)) {
  int ndim = dali::size(in_shape);
  assert(static_cast<int>(dali::size(split_factor)) == ndim);
  for (int d = 0; d < ndim; d++)
    split_factor[d] = 1;

  int64_t vol = volume(in_shape);
  for (int d = 0, nblocks = 1; d < ndim && nblocks < min_nblocks && vol > min_sz; d++) {
    int n = in_shape[d];
    int &b = split_factor[d];
    auto remaining = div_ceil(min_nblocks, nblocks);
    constexpr int kThreshold = 4;
    // ``* kThreshold`` to keep balance of block sizes,
    // only dividing by ``remaining`` when the number is small.
    if (remaining * kThreshold < n) {
      b = remaining;
      nblocks *= b;
      assert(nblocks >= min_nblocks);
      break;
    }

    b = n;
    nblocks *= b;
    vol = div_ceil(vol, b);
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_SPLIT_SHAPE_H_
