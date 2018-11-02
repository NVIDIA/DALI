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

#ifndef DALI_PIPELINE_BASIC_COORDS_H_
#define DALI_PIPELINE_BASIC_COORDS_H_

#include "dali/common.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace basic {

template <typename T>
Index getPlane(const T& shape) {
  return 1;
}

template <size_t I, size_t N, typename T>
Index getPlane(const T& shape) {
  Index result = 1;
  for (size_t i = I; i < N; i++) {
    result *= shape[i];
  }
  return result;
}

template <typename T, size_t N>
Index getOffsetImpl(const T& shape, std::array<Index, N> coords) {
  return 0;
}

template <Index o, Index... os, typename T, size_t N>
Index getOffsetImpl(const T& shape, std::array<Index, N> coords) {
  return coords[o] * getPlane<N - sizeof...(os), N>(shape) + getOffsetImpl<os...>(shape, coords);
}

template <typename T, Index>
using order_to_type = T;

/**
 * @brief Get the offset to coordiantes after requested permutation
 *
 * For given shape {s_1, s_2, ... s_n} and coordinates (c_1, c_2, ..., c_n)
 * return offset after doing axes permuation <o_1, o_2, ..., o_n> (at position i we place axis o_i).
 *
 * calculates:
 *   c[o-1] * (s[o_2] * s[o_3] * ... * s[o_n])
 * + c[o-2] * (s[o_3] * s[o_4] * ... * s[o_n])
 * + ...
 * + c[o_{n-1}] * (s[o_n])
 * + c[o_n]
 *
 * TODO(klecki) formal explanation and handle default case
 *
 * Example: for given shape {N, H, W, C}, and coordiantes (n, h, w, c),
 * and order <0, 1, 2, 3> it would calculate offsets for NHWC data layout.
 * Other order will produce offsets to permutation of dimensions, for example:
 * <0, 3, 1, 2> will result in offsets to NCHW data layout.
 *
 *
 * @tparam order Permutation of input dimensions
 * @tparam T type of container carrying shape information
 * @param shape Data layout "before" permutation
 * @param coords Coordinates "before" permutation
 * @return Index offset to given coordinate
 */
template <Index... order, typename T>
Index getOffset(const T& shape, order_to_type<Index, order>... coords) {
  return getOffsetImpl<order...>(shape, std::array<Index, sizeof...(order)>{coords...});
}

template <Index... Ints>
class dali_index_sequence {};

template <typename T>
struct index_to_array {};

template <Index... Ints>
struct index_to_array<dali_index_sequence<Ints...>> {
  static constexpr std::array<Index, sizeof...(Ints)> array = {Ints...};
  static constexpr size_t N = sizeof...(Ints);
};

template <Index... order, typename T, template <Index...> class Seq>
Index getOffsetBySeq(Seq<order...> seq, const T& shape, order_to_type<Index, order>... coords) {
  return getOffset<order...>(shape, coords...);
}

DALIDataType layoutToTypeId(DALITensorLayout layout);

template <Index... order>
std::array<Index, sizeof...(order)> permuteShape(const std::array<Index, sizeof...(order)>& shape) {
  return {shape[order]...};
}

template <Index... order, template <Index...> class Seq>
std::array<Index, sizeof...(order)> permuteShapeBySeq(
    Seq<order...> seq, const std::array<Index, sizeof...(order)>& shape) {
  return {shape[order]...};
}

// TODO(klecki) - case where sizes are already permutated, implement as Horner's method?

}  // namespace basic

using basic::dali_index_sequence;
using basic::layoutToTypeId;
using basic::permuteShape;
using basic::permuteShapeBySeq;

}  // namespace dali

#endif  // DALI_PIPELINE_BASIC_COORDS_H_
