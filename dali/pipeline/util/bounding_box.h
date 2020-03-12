// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
// limitations under the License.c++ copyswa

#ifndef DALI_PIPELINE_UTIL_BOUNDING_BOX_H_
#define DALI_PIPELINE_UTIL_BOUNDING_BOX_H_

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/math_util.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"
#include "dali/core/tensor_layout.h"

namespace dali {

template <int ndim>
class BoundingBox {
  static_assert(ndim == 2 || ndim == 3);
  static constexpr int coords_size = ndim * 2;

 public:
  static BoundingBox<ndim> Uniform(float min = 0.0f, float max = 1.0f) {
    BoundingBox<ndim> out;
    for (int i = 0; i < ndim; i++)
      out.bbox_bounds_[i] = min;
    for (int i = ndim; i < coords_size; i++)
      out.bbox_bounds_[i] = max;
    return out;
  }

  static BoundingBox<ndim> NoBounds() {
    return Uniform(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
  }

  static BoundingBox<ndim> FromStartAndEnd(span<const float> bbox_bounds,
                                           TensorLayout layout = {},
                                           BoundingBox<ndim> limits = Uniform()) {
    assert(bbox_bounds.size() == coords_size);
    BoundingBox<ndim> out;
    for (int d = 0; d < coords_size; d++)
      out.bbox_bounds_[d] = bbox_bounds[d];

    assert(layout.empty() || layout.size() == coords_size);

    if (!layout.empty()) {  // if layout not provided we assume `xy(z)XY(Z)`
      DALI_ENFORCE(layout.is_permutation_of(InternalLayout()),
        make_string("`", layout, "` is not a permutation of `", InternalLayout(), "`"));
      out.bbox_bounds_ = Permute(out.bbox_bounds_, layout, InternalLayout());
    }

    out.CheckBounds(limits);
    return out;
  }

  static BoundingBox<ndim> FromStartAndEnd(std::array<float, coords_size> bbox_bounds,
                                           TensorLayout layout = {},
                                           BoundingBox<ndim> limits = Uniform()) {
    return FromStartAndEnd(make_cspan(bbox_bounds), layout, limits);
  }

  static BoundingBox<ndim> FromStartAndShape(span<const float> bbox_bounds,
                                             TensorLayout layout = {},
                                             BoundingBox<ndim> limits = Uniform()) {
    BoundingBox<ndim> out;
    for (int d = 0; d < coords_size; d++)
      out.bbox_bounds_[d] = bbox_bounds[d];

    assert(layout.empty() || layout.size() == coords_size);

    if (!layout.empty()) {  // if layout not provided we assume `xy(z)WH(D)`
      DALI_ENFORCE(layout.is_permutation_of(InternalAnchorAndShapeLayout()),
                   make_string("`", layout, "` is not a permutation of `",
                               InternalAnchorAndShapeLayout(), "`"));
      out.bbox_bounds_ = Permute(out.bbox_bounds_, layout, InternalAnchorAndShapeLayout());
    }

    for (int d = 0; d < ndim; d++)
      out.bbox_bounds_[ndim + d] += out.bbox_bounds_[d];

    out.CheckBounds(limits);
    return out;
  }

  static BoundingBox<ndim> FromStartAndShape(std::array<float, coords_size> bbox_bounds,
                                             TensorLayout layout = {},
                                             BoundingBox<ndim> limits = Uniform()) {
    return FromStartAndShape(make_cspan(bbox_bounds), layout, limits);
  }

  static BoundingBox<ndim> From(span<const float> bounds, TensorLayout layout,
                                BoundingBox<ndim> limits = Uniform()) {
    assert(bounds.size() == coords_size);

    if (layout.empty())
      layout = InternalLayout();

    if (layout.is_permutation_of(InternalLayout())) {
      return FromStartAndEnd(bounds, layout, limits);
    } else if (layout.is_permutation_of(InternalAnchorAndShapeLayout())) {
      return FromStartAndShape(bounds, layout, limits);
    } else {
      DALI_FAIL(make_string("Unexpected bbox layout: ", layout));
    }
  }

  static BoundingBox<ndim> From(std::array<float, coords_size> bounds, TensorLayout layout,
                                BoundingBox<ndim> limits = Uniform()) {
    return From(make_cspan(bounds), layout, limits);
  }

  BoundingBox<ndim> Intersect(const BoundingBox<ndim>& oth) const {
    if (!Overlaps(oth)) {
      return BoundingBox<ndim>{};
    }
    std::array<float, coords_size> arr;
    for (int d = 0; d < ndim; d++)
      arr[d] = std::max(oth.bbox_bounds_[d], bbox_bounds_[d]);

    for (int d = 0; d < ndim; d++)
      arr[ndim + d] = std::min(oth.bbox_bounds_[ndim + d], bbox_bounds_[ndim + d]);

    return BoundingBox{arr};
  }

  float Volume() const {
    float vol = 1.0f;
    for (int d = 0; d < ndim; d++) {
      assert(bbox_bounds_[ndim + d] >= bbox_bounds_[d]);
      vol *= bbox_bounds_[ndim + d] - bbox_bounds_[d];
    }
    return vol;
  }

  bool Contains(span<const float> point) const {
    DALI_ENFORCE(point.size() == ndim,
      make_string("Unexpected number of dimensions: ", ndim));
    for (int d = 0; d < ndim; d++) {
      if (point[d] < bbox_bounds_[d] || point[d] > bbox_bounds_[ndim + d])
        return false;
    }
    return true;
  }

  bool Contains(std::array<float, ndim> point) const {
    return Contains(make_cspan(point));
  }

  bool Contains(const BoundingBox<ndim> other_box) const {
    for (int d = 0; d < ndim; d++) {
      if (other_box.bbox_bounds_[d] < bbox_bounds_[d] ||
          other_box.bbox_bounds_[d] > bbox_bounds_[ndim + d] ||
          other_box.bbox_bounds_[ndim + d] < bbox_bounds_[d] ||
          other_box.bbox_bounds_[ndim + d] > bbox_bounds_[ndim + d])
        return false;
    }
    return true;
  }

  bool Overlaps(const BoundingBox& other) const {
    for (int d = 0; d < ndim; d++) {
      if (bbox_bounds_[d] >= other.bbox_bounds_[ndim + d] ||
          bbox_bounds_[ndim + d] <= other.bbox_bounds_[d])
        return false;
    }
    return true;
  }

  float IntersectionOverUnion(const BoundingBox<ndim>& other) const {
    auto intersection_area = Intersect(other).Volume();
    if (intersection_area == 0) {
      return 0.0f;
    }

    const float union_area = Volume() + other.Volume() - intersection_area;
    return intersection_area / union_area;
  }

  BoundingBox<ndim> RemapTo(const BoundingBox<ndim>& other) const {
    std::array<float, coords_size> out;
    for (int d = 0; d < ndim; d++) {
      float oth_start = other.bbox_bounds_[d];
      float start = bbox_bounds_[d];
      float oth_end = other.bbox_bounds_[ndim + d];
      float end = bbox_bounds_[ndim + d];
      float rel_extent = oth_end - oth_start;
      float new_start = (std::max(oth_start, start) - oth_start) / rel_extent;
      float new_end = (std::min(oth_end, end) - oth_start) / rel_extent;
      out[d] = clamp<float>(new_start, 0.0f, 1.0f);
      out[ndim + d] = clamp<float>(new_end, 0.0f, 1.0f);
    }
    return BoundingBox{make_cspan(out)};
  }

  BoundingBox<ndim> AxisFlip(int axis) const {
    assert(axis >= 0 && axis < ndim);
    BoundingBox<ndim> out{bbox_bounds_};
    out.bbox_bounds_[axis]        = 1.0f - bbox_bounds_[ndim + axis];
    out.bbox_bounds_[ndim + axis] = 1.0f - bbox_bounds_[axis];
    return out;
  }

  BoundingBox<ndim> HorizontalFlip() const {
    return AxisFlip(0);
  }

  BoundingBox<ndim> VerticalFlip() const {
    return AxisFlip(1);
  }

  BoundingBox<ndim> DepthWiseFlip() const {
    return AxisFlip(2);
  }

  static constexpr TensorLayout InternalLayout() {
    if (ndim == 3)
      return TensorLayout{"xyzXYZ"};
    else
      return TensorLayout{"xyXY"};
  }

  static constexpr TensorLayout InternalAnchorAndShapeLayout() {
    if (ndim == 3)
      return TensorLayout{"xyzWHD"};
    else
      return TensorLayout{"xyWH"};
  }

  BoundingBox<ndim> AsStartAndEnd(TensorLayout layout = {}) const {
    if (layout.empty() || layout == InternalLayout()) {
      return *this;
    }

    auto out = Permute(bbox_bounds_, InternalLayout(), layout);
    return BoundingBox<ndim>{out};
  }

  BoundingBox<ndim> AsStartAndShape(TensorLayout layout = {}) const {
    auto start_and_shape = bbox_bounds_;
    for (int d = 0; d < ndim; d++) {
      start_and_shape[ndim + d] -= bbox_bounds_[d];
    }

    if (layout.empty() || layout == InternalAnchorAndShapeLayout()) {
      return BoundingBox<ndim>{start_and_shape};
    }
    auto out = Permute(start_and_shape, InternalAnchorAndShapeLayout(), layout);
    return BoundingBox<ndim>{out};
  }

  BoundingBox<ndim> As(TensorLayout layout) const {
    if (layout.find('W') >= 0) {
      return AsStartAndShape(layout);
    } else if (layout.find('X') >= 0) {
      return AsStartAndEnd(layout);
    } else {
      DALI_FAIL(make_string("Unexpected layout: ", layout));
    }
  }

  BoundingBox<ndim> AsCenterAndShape(TensorLayout layout = {}) const {
    auto center_and_shape = bbox_bounds_;
    for (int d = 0; d < ndim; d++) {
      center_and_shape[d]        = 0.5f * (bbox_bounds_[ndim + d] + bbox_bounds_[d]);
      center_and_shape[ndim + d] = bbox_bounds_[ndim + d] - bbox_bounds_[d];
    }

    if (layout.empty() || layout == InternalAnchorAndShapeLayout()) {
      return BoundingBox<ndim>{center_and_shape};
    }
    center_and_shape = Permute(center_and_shape, InternalAnchorAndShapeLayout(), layout);
    return BoundingBox<ndim>{center_and_shape};
  }

  std::array<float, ndim> Centroid() const {
    std::array<float, ndim> centroid;
    for (int d = 0; d < ndim; d++) {
      centroid[d] = 0.5f * (bbox_bounds_[ndim + d] + bbox_bounds_[d]);
    }
    return centroid;
  }

  float AspectRatio(int dim0, int dim1) const {
    assert(dim0 >= 0 && dim0 < ndim);
    assert(dim1 >= 0 && dim1 < ndim);
    auto start_and_shape = AsStartAndShape();
    auto extent0 = start_and_shape.bbox_bounds_[ndim + dim0];
    auto extent1 = start_and_shape.bbox_bounds_[ndim + dim1];
    return extent0 / extent1;
  }

  float MinAspectRatio() const {
    float min_ar = std::numeric_limits<float>::max();
    for (int i = 0; i < ndim; i++) {
      for (int j = i + 1; j < ndim; j++) {
        min_ar = std::min(min_ar, AspectRatio(i, j));
      }
    }
    return min_ar;
  }

  float MaxAspectRatio() const {
    float max_ar = std::numeric_limits<float>::min();
    for (int i = 0; i < ndim; i++) {
      for (int j = i + 1; j < ndim; j++) {
        max_ar = std::max(max_ar, AspectRatio(i, j));
      }
    }
    return max_ar;
  }

  bool Empty() const {
    return Volume() == 0.0f;
  }

  bool operator==(const BoundingBox<ndim>& oth) const {
    for (int d = 0; d < ndim; d++) {
      if (bbox_bounds_[d] != oth.bbox_bounds_[d])
        return false;
    }
    return true;
  }

  bool operator!=(const BoundingBox<ndim>& oth) const {
    return !operator==(oth);
  }

  float& operator[](ptrdiff_t index) noexcept {
    return bbox_bounds_[index];
  }

  const float& operator[](ptrdiff_t index) const noexcept {
    return bbox_bounds_[index];
  }

  explicit BoundingBox(std::array<float, coords_size> bbox_bounds)
    : bbox_bounds_(std::move(bbox_bounds)) {}

  explicit BoundingBox(span<const float> data) {
    assert(data.size() == coords_size);
    for (int d = 0; d < coords_size; d++)
      bbox_bounds_[d] = data[d];
  }

  BoundingBox() {
    for (int d = 0; d < coords_size; d++)
      bbox_bounds_[d] = 0.0f;
  }

 private:
  static std::array<float, coords_size> Permute(const std::array<float, coords_size>& bounds,
                                                TensorLayout orig_layout, TensorLayout new_layout) {
    DALI_ENFORCE(orig_layout.is_permutation_of(new_layout),
                 make_string("`", orig_layout, "` is not a permutation of `", new_layout, "`"));
    auto perm = GetDimIndices(orig_layout, new_layout);
    auto out = bounds;
    for (int d = 0; d < coords_size; d++)
      out[d] = bounds[perm[d]];
    return out;
  }

  void CheckBounds(const BoundingBox<ndim>& limits) const {
    for (int d = 0; d < ndim; d++) {
      auto start = bbox_bounds_[d];
      auto end = bbox_bounds_[ndim + d];
      auto lower = limits.bbox_bounds_[d];
      auto upper = limits.bbox_bounds_[ndim + d];
      DALI_ENFORCE(start >= lower && start <= upper,
                   make_string("dim", d, " start =", start, " is out of bounds. Expected [", lower,
                               ", ", upper, "]"));
      DALI_ENFORCE(end >= lower && end <= upper,
                   make_string("dim", d, " end =", end, " is out of bounds. Expected [", lower,
                               ", ", upper, "]"));
      DALI_ENFORCE(start <= end,
                   make_string("dim", d, " start should be <= end. Got start=", start, ", ", end));
    }
  }

  std::array<float, coords_size> bbox_bounds_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_BOUNDING_BOX_H_
