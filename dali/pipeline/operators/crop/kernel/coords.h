// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_CROP_KERNEL_COORDS_H_
#define DALI_PIPELINE_OPERATORS_CROP_KERNEL_COORDS_H_

#include <array>
#include <cstdint>
#include <iostream>

namespace dali {
namespace detail {

template <typename T, size_t N>
int64_t getOffsetImplLinear(const T& shape, std::array<int64_t, N> coords) {
  int64_t result = coords[0];
  for (size_t i = 1; i < N; i++) {
    result *= shape[i];
    result += coords[i];
  }
  return result;
}

template <int64_t... Ints>
class dali_index_sequence {};

/**
 * @brief Permute shape by given order
 *
 * @tparam order
 * @param shape
 * @return std::array<int64_t, sizeof...(order)>
 */
template <int64_t... order>
std::array<int64_t, sizeof...(order)> permuteShape(
    const std::array<int64_t, sizeof...(order)>& shape, dali_index_sequence<order...> = {}) {
  return {shape[order]...};
}

/**
 * @brief Get the offset to coordiantes after requested permutation
 *
 * For given shape {s_1, s_2, ... s_n} and coordinates (c_1, c_2, ..., c_n)
 * return offset after doing axes permuation <o_1, o_2, ..., o_n> (at position i we place axis o_i).
 *
 * calculates:
 *   c[o_1] * (s[2] * s[3] * ... * s[n])
 * + c[o_2] * (s[3] * s[4] * ... * s[n])
 * + ...
 * + c[o_{n-1}] * (s[n])
 * + c[o_n]
 *
 * Example: for given shape {N, H, W, C}, and coordiantes (n, h, w, c),
 * and order <0, 1, 2, 3> it would calculate offsets for NHWC data layout.
 * Other order will produce offsets to permutation of dimensions, for example:
 * <0, 3, 1, 2> will result in offsets to NCHW data layout.
 *
 * User may provide an optional argument of dali_index_sequence object that carries the information
 * of the order instead of specifying it explicitly in template arguments.
 *
 * getOffset<0, 3, 1, 2>({4, 1080, 1920, 3}, 0, 500, 100, 2);
 * is equivalent to:
 * getOffset({4, 1080, 1920, 3}, 0, 500, 100, 2, dali_index_sequence<0, 3, 1, 2>{});
 *
 * @tparam order Permutation of input dimensions
 * @tparam T type of container carrying shape information
 * @param shape Data layout "before" permutation
 * @param coords Coordinates "after" permutation
 * @return int64_t offset to given coordinate
 */
template <int64_t... order, typename T>
int64_t getOffset(const T& shape, std::array<int64_t, sizeof...(order)> coords,
                  dali_index_sequence<order...> = {}) {
  return detail::getOffsetImplLinear(shape, permuteShape<order...>(coords));
}

// TODO(klecki) - case where sizes are already permutated, go back to more compile time expansion?

}  // namespace detail

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_KERNEL_COORDS_H_
