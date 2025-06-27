// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_GEOM_BOX_H_
#define DALI_CORE_GEOM_BOX_H_

#include "dali/core/geom/vec.h"

namespace dali {

template<int ndims, typename CoordinateType>
struct Box {
  static constexpr int ndim = ndims;
  // box is represented with two ndim coordinates
  static constexpr int size = ndims * 2;
  using corner_t = vec<ndims, CoordinateType>;
  static_assert(is_pod_v<corner_t>, "Corner has to be POD");

  /**
   * Corners of the box.
   * Assumes, that `lo <= hi`, i.e. every coordinate of `lo` will be lower or equal to
   * corresponding coordinate of `hi`.
   */
  corner_t lo, hi;


  constexpr Box() = default;


  /**
   * Creates a bounding-box using 2 points.
   * Assumes, that `lo <= hi`, i.e. every coordinate of `lo` will be lower or equal to
   * corresponding coordinate of `hi`.
   */
  constexpr DALI_HOST_DEV Box(const corner_t &lo, const corner_t &hi) :
          lo(lo), hi(hi) {}


  constexpr DALI_HOST_DEV corner_t extent() const {
    return hi - lo;
  }

  constexpr DALI_HOST_DEV corner_t centroid() const {
    return 0.5 * (hi + lo);
  }

  /**
   * @return true, if this box contains given point
   */
  constexpr DALI_HOST_DEV bool contains(const corner_t &point) const {
    for (int i = 0; i < ndims; i++) {
      if (!(point[i] >= lo[i] && point[i] < hi[i]))
        return false;
    }
    return true;
  }


  /**
   * @return true, if this box contains given box
   */
  constexpr DALI_HOST_DEV bool contains(const Box &other) const {
    for (int i = 0; i < ndims; i++) {
      if (!(other.lo[i] >= lo[i] && other.hi[i] <= hi[i]))
        return false;
    }
    return true;
  }


  /**
   * @return true, if this box overlaps another box
   */
  constexpr DALI_HOST_DEV bool overlaps(const Box &other) const {
    for (int i = 0; i < ndims; i++) {
      if (!(this->lo[i] < other.hi[i] && this->hi[i] > other.lo[i]))
        return false;
    }
    return true;
  }


  /**
   * @return true, if this box is empty (its volume is 0)
   */
  constexpr DALI_HOST_DEV bool empty() const {
    return any_coord(hi <= lo);
  }
};


/**
 * @return volume of a given box
 */
template<int ndims, typename CoordinateType>
constexpr DALI_HOST_DEV CoordinateType volume(const Box<ndims, CoordinateType> &box) {
  return dali::volume(box.extent());
}

/**
 * @return Intersection of two boxes or a default one when the arguments are disjoint.
 */
template <int ndims, typename CoordinateType>
constexpr DALI_HOST_DEV Box<ndims, CoordinateType> intersection(
    const Box<ndims, CoordinateType> &lhs, const Box<ndims, CoordinateType> &rhs) {
  Box<ndims, CoordinateType> tmp = {max(lhs.lo, rhs.lo), min(lhs.hi, rhs.hi)};
  return !all_coords(tmp.hi > tmp.lo) ? Box<ndims, CoordinateType>() : tmp;
}

template <int ndims, typename CoordinateType>
constexpr DALI_HOST_DEV CoordinateType intersection_over_union(
    const Box<ndims, CoordinateType> &lhs, const Box<ndims, CoordinateType> &rhs) {
  auto intersection_vol = volume(intersection(lhs, rhs));
  if (intersection_vol == 0)
    return 0.0f;

  const CoordinateType union_vol = volume(lhs) + volume(rhs) - intersection_vol;
  return intersection_vol / union_vol;
}

template<int ndims, typename CoordinateType>
constexpr DALI_HOST_DEV bool
overlaps(const Box<ndims, CoordinateType> &lhs, const Box<ndims, CoordinateType> &rhs) {
  return lhs.overlaps(rhs);
}

/**
 * Two boxes are equal when their corners are identical
 */
template<int ndims, typename CoordinateType>
constexpr DALI_HOST_DEV bool
operator==(const Box<ndims, CoordinateType> &lhs, const Box<ndims, CoordinateType> &rhs) {
  return lhs.lo == rhs.lo && lhs.hi == rhs.hi;
}


/**
 * Two boxes are equal when their corners are identical
 */
template<int ndims, typename CoordinateType>
constexpr DALI_HOST_DEV bool
operator!=(const Box<ndims, CoordinateType> &lhs, const Box<ndims, CoordinateType> &rhs) {
  return lhs.lo != rhs.lo || lhs.hi != rhs.hi;
}


template <int ndims, typename CoordinateType>
std::ostream &operator<<(std::ostream &os, const Box<ndims, CoordinateType> &box) {
  auto print_corner = [&os](const typename Box<ndims, CoordinateType>::corner_t &c) {
      for (int i = 0; i < ndims; i++)
        os << (i != 0 ? ", " : "(") << c[i];
      os << ")";
  };
  os << "{";
  print_corner(box.lo);
  os << ", ";
  print_corner(box.hi);
  os << "}";
  return os;
}

}  // namespace dali

#endif  // DALI_CORE_GEOM_BOX_H_
