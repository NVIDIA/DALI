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

#ifndef DALI_TYPES_H
#define DALI_TYPES_H

#include "dali/core/geom/vec.h"

namespace dali {

template<size_t ndims, typename CoordinateType>
struct Box {
  static_assert(std::is_standard_layout<CoordinateType>::value,
                "Cannot create a Box of a non-standard layout type");
  using corner_t = vec<ndims, CoordinateType>;
  static_assert(std::is_pod<corner_t>::value, "Corner has to be POD");

  /**
   * Corners of the box.
   * Assumes, that `lo <= hi`, i.e. every coordinate of `lo` will be lower or equal to
   * corresponding coordinate of `hi`.
   */
  corner_t lo, hi;

  /**
   * Creates an empty box
   */
  Box() = default;


  /**
   * Creates a bounding-box using 2 points.
   * Assumes, that `lo <= hi`, i.e. every coordinate of `lo` will be lower or equal to
   * corresponding coordinate of `hi`.
   */
  Box(const corner_t &lo, const corner_t &hi) {
    this->lo = lo;
    this->hi = hi;
  }


  /**
   * Convenient overload, for 2-D case
   * @param left x-coordinate of `lo`
   * @param top y-coordinate of `lo`
   * @param right x-coordinate of `hi`
   * @param bottom y-coordinate of `hi`
   */
  Box(CoordinateType left, CoordinateType top, CoordinateType right, CoordinateType bottom) {
    lo = {left, top};
    hi = {right, bottom};
  }


  constexpr DALI_HOST_DEV corner_t extent() const {
    return hi - lo;
  }


  /**
   * @return true, if this box contains given point
   */
  constexpr DALI_HOST_DEV bool contains(const corner_t &point) const {
    for (size_t i = 0; i < ndims; i++) {
      if (this->lo[i] > point[i] || this->hi[i] <= point[i])
        return false;
    }
    return true;
  }


  /**
   * @return true, if this box contains given box
   */
  constexpr DALI_HOST_DEV bool contains(const Box<ndims, CoordinateType> &other) const {
    for (size_t i = 0; i < ndims; i++) {
      if (this->lo[i] > other.lo[i] || this->hi[i] < other.hi[i])
        return false;
    }
    return true;
  }


  /**
   * @return true, if this box overlaps another box
   */
  constexpr DALI_HOST_DEV bool overlaps(const Box &other) const {
    for (size_t i = 0; i < ndims; i++) {
      if (this->lo[i] >= other.hi[i] || this->hi[i] <= other.lo[i])
        return false;
    }
    return true;
  }


  /**
   * @return true, if this box is empty (its volume is 0)
   */
  constexpr DALI_HOST_DEV bool empty() const {
    return lo == hi;
  }
};


/**
 * @return volume of a given box
 */
template<size_t ndims, typename CoordinateType>
constexpr DALI_HOST_DEV CoordinateType volume(const Box<ndims, CoordinateType> &box) {
  return dali::volume(box.extent());
}


/**
 * @return Box, that is an intersection of two boxes
 */
template<size_t ndims, typename CoordinateType>
constexpr DALI_HOST_DEV Box<ndims, CoordinateType>
intersection(const Box<ndims, CoordinateType> &lhs, const Box<ndims, CoordinateType> &rhs) {
//  Box<ndims, CoordinateType> tmp = { max(lhs.lo, rhs.lo), min(lhs.hi, rhs.hi) };
//  return all_coords(tmp.hi >= tmp.lo) ? Box<ndims, CoordinateType>() : tmp;
  if (!lhs.overlaps(rhs)) return {};
  return Box<ndims, CoordinateType>(max(lhs.lo, rhs.lo), min(lhs.hi, rhs.hi));
}


/**
 * Two boxes are equal IFF its corners are identical
 */
template<size_t ndims, typename CoordinateType>
constexpr DALI_HOST_DEV bool
operator==(const Box<ndims, CoordinateType> &lhs, const Box<ndims, CoordinateType> &rhs) {
  return lhs.lo == rhs.lo && lhs.hi == rhs.hi;
}

}  // namespace dali

#endif //DALI_TYPES_H
