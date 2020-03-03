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
#include <utility>
#include <vector>
#include <limits>
#include <string>
#include "dali/core/error_handling.h"
#include "dali/core/small_vector.h"
#include "dali/core/format.h"
#include "dali/core/math_util.h"

namespace dali {

using RelCoords = SmallVector<float, 3>;
using RelBounds = SmallVector<float, 6>;  // e.g. start0, start1, ..., end0, end1, ...

class BoundingBox {
 public:
  static RelBounds Uniform(int ndim = 2, float min = 0.0f, float max = 1.0f) {
    assert(ndim > 0);
    RelBounds bounds;
    bounds.reserve(ndim * 2);
    for (int i = 0; i < ndim; i++)
      bounds.push_back(min);
    for (int i = 0; i < ndim; i++)
      bounds.push_back(max);
    return bounds;
  }

  static RelBounds NoBounds(int ndim = 2) {
    return Uniform(ndim, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
  }

  static BoundingBox FromStartAndEnd(RelBounds bbox_bounds, RelBounds limits = {}) {
    DALI_ENFORCE(bbox_bounds.size() == limits.size() || limits.empty());
    DALI_ENFORCE(bbox_bounds.size() % 2 == 0);
    int ndim = bbox_bounds.size() / 2;
    DALI_ENFORCE(ndim >= 2 && ndim <= 3);

    if (limits.empty())
      limits = NoBounds(ndim);

    CheckBounds(bbox_bounds[0], bbox_bounds[ndim],   limits[0], limits[ndim],   "left", "right");
    CheckBounds(bbox_bounds[1], bbox_bounds[ndim+1], limits[1], limits[ndim+1], "top",  "bottom");

    // handle depth dimension (if necessary)
    if (ndim == 3)
      CheckBounds(bbox_bounds[2], bbox_bounds[ndim+2], limits[2], limits[ndim+2], "front", "back");

    return BoundingBox{bbox_bounds};
  }

  static BoundingBox FromStartAndShape(RelBounds start_shape, RelBounds limits = {}) {
    DALI_ENFORCE(start_shape.size() == limits.size() || limits.empty());
    DALI_ENFORCE(start_shape.size() % 2 == 0);
    int ndim = start_shape.size() / 2;
    DALI_ENFORCE(ndim >= 2 && ndim <= 3, make_string("Unexpected number of dimensions: ", ndim));

    auto bbox_bounds = start_shape;
    for (int i = 0; i < ndim; i++) {
      bbox_bounds[ndim+i] += bbox_bounds[i];
    }

    return FromStartAndEnd(bbox_bounds, limits);
  }

  static BoundingBox FromLtrb(float l, float t, float r, float b,
                              RelBounds limits = {}) {
    return FromStartAndEnd(RelBounds{l, t, r, b}, limits);
  }

  static BoundingBox FromXywh(float x, float y, float w, float h,
                              RelBounds limits = {}) {
    return FromStartAndShape(RelBounds{x, y, w, h}, limits);
  }

  BoundingBox Intersect(const BoundingBox& oth) const {
    DALI_ENFORCE(ndim() == oth.ndim());
    RelBounds out;
    if (!Overlaps(oth)) {
      return BoundingBox{out};
    }
    out.reserve(ndim() * 2);
    for (int i = 0; i < ndim(); i++)
      out.push_back(std::max(oth.bbox_bounds_[i], bbox_bounds_[i]));

    for (int i = 0; i < ndim(); i++)
      out.push_back(std::min(oth.bbox_bounds_[ndim() + i], bbox_bounds_[ndim() + i]));

    return BoundingBox{out};
  }

  float Volume() const {
    if (bbox_bounds_.empty())
      return 0.0f;

    float vol = 1.0f;
    for (int i = 0; i < ndim(); i++) {
      assert(bbox_bounds_[ndim()+i] >= bbox_bounds_[i]);
      vol *= bbox_bounds_[ndim()+i] - bbox_bounds_[i];
    }
    return vol;
  }

  float Area() const {
    DALI_ENFORCE(ndim() == 2,
      make_string("Requested area on ", ndim(), "-dimensional bounding box. Use Volume() instead"));
    return Volume();
  }

  bool Contains(RelCoords point) const {
    DALI_ENFORCE(static_cast<int>(point.size()) == ndim(),
      make_string("Unexpected number of dimensions: ", ndim()));
    for (int i = 0; i < ndim(); i++) {
      if (point[i] < bbox_bounds_[i] || point[i] > bbox_bounds_[ndim()+i])
        return false;
    }
    return true;
  }

  bool Contains(float x, float y) const {
    DALI_ENFORCE(ndim() == 2,
      make_string("Unexpected number of dimensions: ", ndim()));
    return Contains(RelCoords{x, y});
  }

  bool Overlaps(const BoundingBox& other) const {
    for (int i = 0; i < ndim(); i++) {
      float this_start = bbox_bounds_[i], this_end = bbox_bounds_[ndim() + i],
            oth_start = other.bbox_bounds_[i], oth_end = other.bbox_bounds_[ndim() + i];
      if (this_start >= oth_end || this_end <= oth_start)
        return false;
    }
    return true;
  }

  float IntersectionOverUnion(const BoundingBox& other) const {
    auto intersection_area = Intersect(other).Volume();
    if (intersection_area == 0) {
      return 0.0f;
    }

    const float union_area = Volume() + other.Volume() - intersection_area;
    return intersection_area / union_area;
  }

  BoundingBox RemapTo(const BoundingBox& other) const {
    RelBounds out;
    out.resize(ndim() * 2);
    for (int i = 0; i < ndim(); i++) {
      float oth_start = other.bbox_bounds_[i], start = bbox_bounds_[i];
      float oth_end = other.bbox_bounds_[ndim()+i], end = bbox_bounds_[ndim()+i];
      float rel_extent = oth_end - oth_start;
      float new_start = (std::max(oth_start, start) - oth_start) / rel_extent;
      float new_end = (std::min(oth_end, end) - oth_start) / rel_extent;
      out[i] = clamp<float>(new_start, 0.0f, 1.0f);
      out[ndim() + i] = clamp<float>(new_end, 0.0f, 1.0f);
    }
    return BoundingBox{out};
  }

  BoundingBox HorizontalFlip() const {
    RelBounds out = bbox_bounds_;
    out[0]        = 1.0f - bbox_bounds_[ndim()];
    out[ndim()]   = 1.0f - bbox_bounds_[0];
    return BoundingBox{out};
  }

  BoundingBox VerticalFlip() const {
    RelBounds out = bbox_bounds_;
    out[1]        = 1.0f - bbox_bounds_[ndim()+1];
    out[ndim()+1] = 1.0f - bbox_bounds_[1];
    return BoundingBox{out};
  }

  RelBounds AsStartAndEnd() const {
    return bbox_bounds_;
  }

  RelBounds AsStartAndShape() const {
    auto out = bbox_bounds_;
    for (int i = 0; i < ndim(); i++) {
      out[ndim()+i] -= bbox_bounds_[i];
    }
    return out;
  }

  RelBounds AsCenterAndShape() const {
    auto out = bbox_bounds_;
    for (int i = 0; i < ndim(); i++) {
      out[i]        = 0.5f * (bbox_bounds_[ndim()+i] + bbox_bounds_[i]);
      out[ndim()+i] = bbox_bounds_[ndim()+i] - bbox_bounds_[i];
    }
    return out;
  }

  RelBounds Centroid() const {
    RelBounds centroid;
    centroid.reserve(ndim());
    for (int i = 0; i < ndim(); i++) {
      centroid.push_back(0.5f * (bbox_bounds_[ndim()+i] + bbox_bounds_[i]));
    }
    return centroid;
  }

  float AspectRatio(int dim0, int dim1) const {
    DALI_ENFORCE(dim0 >= 0 && dim0 < ndim());
    DALI_ENFORCE(dim1 >= 0 && dim1 < ndim());
    auto start_and_shape = AsStartAndShape();
    auto extent0 = start_and_shape[ndim()+dim0];
    auto extent1 = start_and_shape[ndim()+dim1];
    return extent0 / extent1;
  }

  float MinAspectRatio() const {
    float min_ar = std::numeric_limits<float>::max();
    for (int i = 0; i < ndim(); i++) {
      for (int j = i + 1; j < ndim(); j++) {
        min_ar = std::min(min_ar, AspectRatio(i, j));
      }
    }
    return min_ar;
  }

  float MaxAspectRatio() const {
    float max_ar = std::numeric_limits<float>::min();
    for (int i = 0; i < ndim(); i++) {
      for (int j = i + 1; j < ndim(); j++) {
        max_ar = std::max(max_ar, AspectRatio(i, j));
      }
    }
    return max_ar;
  }

  int ndim() const {
    int bbox_bounds_size = bbox_bounds_.size();
    assert(bbox_bounds_size % 2 == 0);
    return bbox_bounds_size / 2;
  }

  bool Empty() const {
    return bbox_bounds_.empty() || Volume() == 0.0f;
  }

  bool operator==(const BoundingBox& oth) const {
    if (ndim() != oth.ndim()) {
      return false;
    }

    for (int i = 0; i < ndim(); i++) {
      if (bbox_bounds_[i] != oth.bbox_bounds_[i])
        return false;
    }

    return true;
  }

  bool operator!=(const BoundingBox& oth) const {
    return !operator==(oth);
  }

  explicit BoundingBox(RelBounds bbox_bounds = {})
    : bbox_bounds_(std::move(bbox_bounds)) {}

 private:
  static void CheckBounds(float start, float end, float lower, float upper,
                          const string& name_start, const string& name_end) {
    DALI_ENFORCE(start >= lower && start <= upper,
      make_string(name_start, "=", start, " is out of bounds [", lower, ", ", upper, "]"));
    DALI_ENFORCE(end >= lower && end <= upper,
      make_string(name_end, "=", end, " is out of bounds [", lower, ", ", upper, "]"));
      assert(start <= end);
    DALI_ENFORCE(start <= end,
      make_string(name_start, " should be <= ", name_end, ". Got ",
                  name_start, "=", start, ", ", name_end, "=", end));
  }

  RelBounds bbox_bounds_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_UTIL_BOUNDING_BOX_H_
