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

#include <utility>
#include "dali/core/util.h"
#include "dali/core/tensor_shape.h"

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
 * @param skip_dim_mask Bitmask representing which dimensions should not be split
 * @return product of split_factor
 */
template <typename SplitFactor, typename Shape>
int split_shape(SplitFactor& split_factor, const Shape& in_shape, int min_nblocks,
                int min_sz = 16000, uint64_t skip_dim_mask = 0) {
  int ndim = dali::size(in_shape);
  assert(static_cast<int>(dali::size(split_factor)) == ndim);
  for (int d = 0; d < ndim; d++)
    split_factor[d] = 1;

  int64_t vol = volume(in_shape);
  for (int d = 0, nblocks = 1; d < ndim && nblocks < min_nblocks && vol > min_sz; d++) {
    if (skip_dim_mask & (1_u64 << d))
      continue;
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
  return volume(split_factor);
}

/**
 * @brief returns the dimension index with a split factor > 1
 */
template <typename SplitFactor>
int LastSplitDim(const SplitFactor& split_factor) {
  int last_split_dim = -1;
  int ndim = dali::size(split_factor);
  for (int d = ndim - 1; d >= 0; d--) {
    if (split_factor[d] > 1) {
      last_split_dim = d;
      break;
    }
  }
  return last_split_dim;
}

/**
 * @brief Iterates over blocks, based on a split factor for each dimension
 * @param start start coordinates of the region
 * @param end end coordinates of the region
 * @param split_factor split factor for each dimension in the region
 * @param d Current dimension
 * @param max_split_dim last dimension with a split factor different than 1.
 * @param func Function to run for each block.
 */
template <int ndim, typename SplitFactor, typename OnBlockFunc>
void ForEachBlock(TensorShape<ndim> start, TensorShape<ndim> end, const SplitFactor& split_factor,
                  int d, int max_split_dim, OnBlockFunc&& func) {
  assert(start.size() == end.size());
  if (d > max_split_dim || d == start.size()) {
    func(start, end);
    return;
  }

  if (split_factor[d] == 1) {
    ForEachBlock(start, end, split_factor, d + 1, max_split_dim,
                 std::forward<OnBlockFunc>(func));
    return;
  }

  int64_t start_d = start[d];
  int64_t extent_d = end[d] - start_d;
  int nblocks_d = split_factor[d];
  int64_t prev_end = start_d;
  for (int b = 0; b < nblocks_d; b++) {
    start[d] = prev_end;
    end[d] = prev_end = extent_d * (b + 1) / nblocks_d + start_d;
    ForEachBlock(start, end, split_factor, d + 1, max_split_dim,
                 std::forward<OnBlockFunc>(func));
  }
}

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_COMMON_SPLIT_SHAPE_H_
